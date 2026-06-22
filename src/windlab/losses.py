"""Loss functions used by the training pipeline."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import NDArray

from windlab.registry import LOSSES

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ModelOutput = Mapping[str, Any] | torch.Tensor
LossFn = Callable[[ModelOutput, torch.Tensor, torch.Tensor | None], torch.Tensor]
LossTermBuilder = Callable[[dict[str, Any], "LossBuildContext"], LossFn]
SUPPORTED_FOURIER_MASK_MODES = frozenset({"strict_real_only", "all_points"})
SUPPORTED_PATCH_WISE_STRUCTURAL_MASK_MODES = frozenset(
    {"strict_real_only", "all_points"}
)


class LossTermConfigProtocol(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    @property
    def params(self) -> Mapping[str, Any]: ...


class LossConfigProtocol(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def terms(self) -> Sequence[LossTermConfigProtocol]: ...


@dataclass(frozen=True)
class LossBuildContext:
    train_targets: FloatArray | None = None
    train_mask: BoolArray | None = None
    annealing_steps: int = 1
    total_train_steps: int = 1


@dataclass(frozen=True)
class _InlineLossTermConfig:
    name: str
    weight: float
    params: dict[str, Any]


@dataclass(frozen=True)
class _InlineLossConfig:
    name: str
    terms: list[_InlineLossTermConfig]


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    squared_error = (prediction - target) ** 2
    if mask is None:
        return squared_error.mean()
    active = squared_error[mask.bool()]
    if active.numel() == 0:
        raise ValueError("Mask selects no elements for masked_mse_loss.")
    return active.mean()


def _extract_prediction_and_aux(
    output: ModelOutput,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    if isinstance(output, torch.Tensor):
        return output, {}
    if "prediction" not in output:
        raise KeyError("Model output mapping must contain 'prediction'.")
    prediction = output["prediction"]
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Model output 'prediction' must be a torch.Tensor.")
    aux = output.get("aux", {})
    if aux is None:
        return prediction, {}
    if not isinstance(aux, Mapping):
        raise TypeError("Model output 'aux' must be a mapping when provided.")
    return prediction, aux


def _default_loss_config() -> _InlineLossConfig:
    return _InlineLossConfig(
        name="composite",
        terms=[_InlineLossTermConfig(name="mse", weight=1.0, params={})],
    )


def _select_active_forecast_sequences(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    mask_mode: str,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction shape "
            f"{prediction.shape} does not match target shape {target.shape}."
        )
    if prediction.ndim != 4:
        raise ValueError(
            f"{loss_name} expects tensors shaped "
            "[batch, horizon, airport, target]."
        )
    if mask is not None and mask.shape != target.shape:
        raise ValueError(
            "Mask shape "
            f"{mask.shape} does not match target shape {target.shape}."
        )

    prediction_sequences = prediction.permute(0, 2, 3, 1)
    target_sequences = target.permute(0, 2, 3, 1)
    if mask is None or mask_mode == "all_points":
        sequence_mask = torch.ones_like(
            prediction_sequences[..., 0],
            dtype=torch.bool,
        )
    else:
        sequence_mask = mask.bool().permute(0, 2, 3, 1).all(dim=-1)
    return prediction_sequences[sequence_mask], target_sequences[sequence_mask]


def _symmetric_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(p.dtype).eps
    p = p.clamp_min(epsilon)
    q = q.clamp_min(epsilon)
    kl_pq = functional.kl_div(p.log(), q, reduction="batchmean")
    kl_qp = functional.kl_div(q.log(), p, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


def _dirichlet_kl(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    beta = torch.ones(
        (1, num_classes),
        device=alpha.device,
        dtype=alpha.dtype,
    )
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_beta = torch.sum(beta, dim=-1, keepdim=True)
    ln_b = torch.lgamma(sum_alpha) - torch.sum(
        torch.lgamma(alpha), dim=-1, keepdim=True
    )
    ln_b_uniform = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(
        sum_beta
    )
    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)
    return (
        torch.sum((alpha - beta) * (digamma_alpha - digamma_sum), dim=-1, keepdim=True)
        + ln_b
        + ln_b_uniform
    )


def _uncertainty_cross_entropy(
    labels: torch.Tensor,
    logits: torch.Tensor,
    *,
    num_classes: int,
    step_count: int,
    annealing_steps: int,
) -> torch.Tensor:
    if labels.numel() == 0:
        raise ValueError("HCAN classification labels are empty.")
    evidence = functional.softplus(logits)
    alpha = evidence + 1.0
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    residual_evidence = alpha - 1.0
    one_hot = functional.one_hot(labels, num_classes=num_classes).to(dtype=logits.dtype)
    belief = residual_evidence / sum_alpha
    confidence = 1.0 - belief
    term_a = torch.sum(
        confidence * one_hot * (torch.digamma(sum_alpha) - torch.digamma(alpha)),
        dim=-1,
        keepdim=True,
    )
    annealing_coef = min(1.0, step_count / max(annealing_steps, 1))
    adjusted_alpha = residual_evidence * (1.0 - one_hot) + 1.0
    term_b = annealing_coef * _dirichlet_kl(adjusted_alpha, num_classes)
    return torch.mean(term_a + term_b)


class CompositeLoss:
    """Weighted sum of registry-built loss terms."""

    def __init__(self, terms: list[tuple[float, LossFn]]) -> None:
        if not terms:
            raise ValueError("CompositeLoss requires at least one term.")
        self.terms = terms

    def __call__(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, _ = _extract_prediction_and_aux(output)
        total = prediction.sum() * 0.0
        for weight, term in self.terms:
            total = total + weight * term(output, target, mask)
        return total


class MaskedMSELossTerm:
    """Plain masked MSE over the forecast output contract."""

    def __call__(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, _ = _extract_prediction_and_aux(output)
        return masked_mse_loss(prediction, target, mask)


class HCANAuxiliaryLossTerm:
    """HCAN hierarchical objective, including the direct forecast term."""

    def __init__(
        self,
        *,
        train_targets: FloatArray | None = None,
        train_mask: BoolArray | None = None,
        annealing_steps: int = 1,
    ) -> None:
        self._train_targets = train_targets
        self._train_mask = train_mask
        self._annealing_steps = annealing_steps
        self._step_count = 0
        self._boundary_cache: dict[int, np.ndarray[Any, np.dtype[np.float64]]] = {}

    def __call__(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, aux = _extract_prediction_and_aux(output)
        hcan_aux = aux.get("hcan")
        if hcan_aux is None:
            raise ValueError(
                "Configured hcan_auxiliary loss requires model output aux['hcan']."
            )
        if not isinstance(hcan_aux, Mapping):
            raise TypeError("HCAN auxiliary payload must be a mapping.")
        current_step = self._step_count
        if torch.is_grad_enabled():
            self._step_count += 1
        direct_loss = masked_mse_loss(prediction, target, mask)
        return self._compose_hcan_loss(
            direct_loss=direct_loss,
            target=target,
            mask=mask,
            hcan_aux=hcan_aux,
            step_count=current_step,
        )

    def _compose_hcan_loss(
        self,
        *,
        direct_loss: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        hcan_aux: Mapping[str, Any],
        step_count: int,
    ) -> torch.Tensor:
        coarse_prediction = self._require_tensor(hcan_aux, "coarse_prediction")
        coarse_logits = self._require_tensor(hcan_aux, "coarse_logits")
        fine_prediction = self._require_tensor(hcan_aux, "fine_prediction")
        fine_logits = self._require_tensor(hcan_aux, "fine_logits")
        fine_logits_switch = self._require_tensor(hcan_aux, "fine_logits_switch")

        num_coarse = self._require_positive_int(hcan_aux, "num_coarse")
        num_fine = self._require_positive_int(hcan_aux, "num_fine")
        if num_fine != 2 * num_coarse:
            raise ValueError(
                "HCAN auxiliary payload requires num_fine == 2 * num_coarse."
            )

        active_mask = self._resolve_active_mask(target, mask)
        coarse_labels, coarse_regression_target = self._build_level_targets(
            target,
            num_classes=num_coarse,
        )
        fine_labels, fine_regression_target = self._build_level_targets(
            target,
            num_classes=num_fine,
        )

        coarse_logits_active = coarse_logits[active_mask]
        fine_logits_active = fine_logits[active_mask]
        fine_logits_switch_active = fine_logits_switch[active_mask]
        coarse_labels_active = coarse_labels[active_mask]
        fine_labels_active = fine_labels[active_mask]

        coarse_loss = _uncertainty_cross_entropy(
            coarse_labels_active,
            coarse_logits_active,
            num_classes=num_coarse,
            step_count=max(step_count, 1),
            annealing_steps=self._annealing_steps,
        )
        fine_loss = _uncertainty_cross_entropy(
            fine_labels_active,
            fine_logits_active,
            num_classes=num_fine,
            step_count=max(step_count, 1),
            annealing_steps=self._annealing_steps,
        )

        coarse_predicted_relative = coarse_prediction.gather(
            dim=-1,
            index=coarse_labels.unsqueeze(-1),
        ).squeeze(-1)
        fine_predicted_relative = fine_prediction.gather(
            dim=-1,
            index=fine_labels.unsqueeze(-1),
        ).squeeze(-1)
        coarse_regression_loss = masked_mse_loss(
            coarse_predicted_relative,
            coarse_regression_target,
            active_mask,
        )
        fine_regression_loss = masked_mse_loss(
            fine_predicted_relative,
            fine_regression_target,
            active_mask,
        )
        acl_loss = _symmetric_kl_divergence(
            torch.softmax(coarse_logits_active, dim=-1),
            torch.softmax(fine_logits_switch_active, dim=-1),
        )

        lambda_cls = self._require_non_negative_float(hcan_aux, "lambda_cls")
        lambda_reg = self._require_non_negative_float(hcan_aux, "lambda_reg")
        lambda_acl = self._require_non_negative_float(hcan_aux, "lambda_acl")
        lambda_direct = self._require_non_negative_float(hcan_aux, "lambda_direct")
        return (
            lambda_direct * direct_loss
            + lambda_cls * coarse_loss
            + lambda_reg * coarse_regression_loss
            + lambda_cls * fine_loss
            + lambda_reg * fine_regression_loss
            + lambda_acl * acl_loss
        )

    def _resolve_active_mask(
        self,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(target, dtype=torch.bool)
        active_mask = mask.bool()
        if not torch.any(active_mask):
            raise ValueError("Mask selects no elements for masked_mse_loss.")
        return active_mask

    def _build_level_targets(
        self,
        target: torch.Tensor,
        *,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundaries = torch.as_tensor(
            self._fit_boundaries(num_classes),
            device=target.device,
            dtype=target.dtype,
        )
        labels = torch.bucketize(target, boundaries[1:-1], right=False)
        lower_bounds = boundaries.index_select(0, labels.reshape(-1)).reshape_as(target)
        regression_target = target - lower_bounds
        return labels.to(dtype=torch.long), regression_target

    def _fit_boundaries(
        self,
        num_classes: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        cached = self._boundary_cache.get(num_classes)
        if cached is not None:
            return cached
        if self._train_targets is None:
            raise ValueError(
                "HCAN loss requires train_targets to fit hierarchy boundaries."
            )
        flat_values = self._train_targets
        if self._train_mask is not None:
            observed_values = flat_values[self._train_mask]
        else:
            observed_values = flat_values.reshape(-1)
        if observed_values.size == 0:
            raise ValueError(
                "HCAN loss cannot fit boundaries from an empty training target set."
            )
        sorted_values = np.sort(observed_values.astype(np.float64, copy=False))
        last_index = sorted_values.size - 1
        boundaries = np.zeros(num_classes + 1, dtype=np.float64)
        for group_index in range(num_classes + 1):
            data_index = int(np.floor(last_index * group_index / num_classes))
            boundaries[group_index] = sorted_values[data_index]
        self._boundary_cache[num_classes] = boundaries
        return boundaries

    def _require_tensor(self, payload: Mapping[str, Any], key: str) -> torch.Tensor:
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"HCAN auxiliary field {key!r} must be a torch.Tensor.")
        return value

    def _require_positive_int(self, payload: Mapping[str, Any], key: str) -> int:
        value = payload.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"HCAN auxiliary field {key!r} must be a positive integer."
            )
        return value

    def _require_non_negative_float(
        self,
        payload: Mapping[str, Any],
        key: str,
    ) -> float:
        value = payload.get(key)
        if not isinstance(value, (int, float)) or float(value) < 0.0:
            raise ValueError(
                f"HCAN auxiliary field {key!r} must be a non-negative float."
            )
        return float(value)


class FourierAmplitudeCorrelationLossTerm:
    """Paper-aligned FACL term adapted to 1D forecast horizons."""

    def __init__(
        self,
        *,
        mode: str,
        total_train_steps: int,
        alpha: float | None = None,
        mask_mode: str = "strict_real_only",
    ) -> None:
        if mode not in {"paper_random", "fal", "fcl"}:
            raise ValueError("mode must be 'paper_random', 'fal', or 'fcl'.")
        if total_train_steps <= 0:
            raise ValueError("total_train_steps must be positive.")
        if mask_mode not in SUPPORTED_FOURIER_MASK_MODES:
            raise ValueError(
                "mask_mode must be 'strict_real_only' or 'all_points'."
            )
        if mode == "paper_random":
            if alpha is None or not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    "alpha must be provided in [0.0, 1.0] for paper_random mode."
                )
        elif alpha is not None:
            raise ValueError("alpha is only valid when mode='paper_random'.")
        self.mode = mode
        self.total_train_steps = total_train_steps
        self.alpha = alpha
        self.mask_mode = mask_mode
        self.step_count = 0

    def __call__(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, _ = _extract_prediction_and_aux(output)
        active_prediction, active_target = self._select_active_sequences(
            prediction,
            target,
            mask,
        )
        if active_prediction.numel() == 0:
            return prediction.sum() * 0.0

        fft_prediction = torch.fft.fft(active_prediction, dim=-1, norm="ortho")
        fft_target = torch.fft.fft(active_target, dim=-1, norm="ortho")
        fal_loss = functional.mse_loss(fft_prediction.abs(), fft_target.abs())
        fcl_loss = self._fcl(fft_prediction, fft_target)
        scale = prediction.new_tensor(math.sqrt(float(active_prediction.shape[-1])))

        if self.mode == "fal":
            return scale * fal_loss
        if self.mode == "fcl":
            return scale * fcl_loss

        current_step = self.step_count
        if torch.is_grad_enabled():
            self.step_count += 1
        probability = self._current_probability(current_step)
        if not torch.is_grad_enabled():
            return scale * (probability * fcl_loss + (1.0 - probability) * fal_loss)
        if random.random() > probability:
            return scale * fal_loss
        return scale * fcl_loss

    def _select_active_sequences(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _select_active_forecast_sequences(
            prediction,
            target,
            mask,
            mask_mode=self.mask_mode,
            loss_name="FourierAmplitudeCorrelationLossTerm",
        )

    def _current_probability(self, step: int) -> float:
        alpha = 0.0 if self.alpha is None else self.alpha
        constant_tail_steps = int(self.total_train_steps * alpha)
        linear_steps = self.total_train_steps - constant_tail_steps
        if linear_steps <= 0:
            return 0.0
        if step >= linear_steps:
            return 0.0
        if linear_steps == 1:
            return 1.0
        return max(0.0, 1.0 - (step / float(linear_steps - 1)))

    def _fcl(
        self,
        fft_prediction: torch.Tensor,
        fft_target: torch.Tensor,
    ) -> torch.Tensor:
        numerator = (torch.conj(fft_prediction) * fft_target).sum().real
        denominator = torch.sqrt(
            ((fft_target.abs() ** 2).sum()) * ((fft_prediction.abs() ** 2).sum())
        )
        epsilon = torch.finfo(fft_prediction.real.dtype).eps
        if bool((denominator <= epsilon).detach().cpu().item()):
            return fft_prediction.real.sum() * 0.0
        return 1.0 - numerator / denominator.clamp_min(epsilon)


class PatchWiseStructuralLossTerm:
    """Patch-wise Structural Loss adapted to the forecast output contract."""

    def __init__(
        self,
        *,
        patch_len_threshold: int,
        mask_mode: str,
    ) -> None:
        if patch_len_threshold <= 0:
            raise ValueError("patch_len_threshold must be positive.")
        if mask_mode not in SUPPORTED_PATCH_WISE_STRUCTURAL_MASK_MODES:
            raise ValueError(
                "mask_mode must be 'strict_real_only' or 'all_points'."
            )
        self.patch_len_threshold = patch_len_threshold
        self.mask_mode = mask_mode
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

    def __call__(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, _ = _extract_prediction_and_aux(output)
        if not torch.is_grad_enabled():
            return prediction.sum() * 0.0

        active_prediction, active_target = _select_active_forecast_sequences(
            prediction,
            target,
            mask,
            mask_mode=self.mask_mode,
            loss_name="PatchWiseStructuralLossTerm",
        )
        if active_prediction.numel() == 0:
            return prediction.sum() * 0.0

        true_patches, pred_patches = self._fourier_based_adaptive_patching(
            active_target,
            active_prediction,
        )
        corr_loss, var_loss, mean_loss = self._patch_wise_structural_loss(
            true_patches,
            pred_patches,
        )
        alpha, beta, gamma = self._gradient_based_dynamic_weighting(
            active_target,
            active_prediction,
            corr_loss,
            var_loss,
            mean_loss,
        )
        return alpha * corr_loss + beta * var_loss + gamma * mean_loss

    def _create_patches(
        self,
        values: torch.Tensor,
        *,
        patch_len: int,
        stride: int,
    ) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("PatchWiseStructuralLossTerm expects [sequence, horizon].")
        unfolded = values.unsqueeze(1).unfold(2, patch_len, stride)
        return unfolded.reshape(values.shape[0], 1, unfolded.shape[2], patch_len)

    def _fourier_based_adaptive_patching(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = true.shape[-1]
        true_fft = torch.fft.rfft(true.unsqueeze(-1), dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        if frequency_list.numel() == 0:
            raise ValueError("PatchWiseStructuralLossTerm received an empty horizon.")
        frequency_list = frequency_list.clone()
        frequency_list[:1] = 0.0
        top_index = int(torch.argmax(frequency_list).detach().cpu().item())
        if top_index <= 0:
            candidate_patch_len = min(self.patch_len_threshold, max(2, horizon // 2))
        else:
            period = max(2, horizon // top_index)
            candidate_patch_len = min(self.patch_len_threshold, period // 2)
        patch_len = min(horizon, max(2, candidate_patch_len))
        stride = max(1, patch_len // 2)
        return (
            self._create_patches(true, patch_len=patch_len, stride=stride),
            self._create_patches(pred, patch_len=patch_len, stride=stride),
        )

    def _patch_wise_structural_loss(
        self,
        true_patch: torch.Tensor,
        pred_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        epsilon = torch.finfo(true_patch.dtype).eps
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)

        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var.clamp_min(epsilon))
        pred_patch_std = torch.sqrt(pred_patch_var.clamp_min(epsilon))

        true_pred_patch_cov = torch.mean(
            (true_patch - true_patch_mean) * (pred_patch - pred_patch_mean),
            dim=-1,
            keepdim=True,
        )
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (
            true_patch_std * pred_patch_std + 1e-5
        )
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_log_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_log_softmax, true_patch_softmax).sum(
            dim=-1
        ).mean()
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()
        return linear_corr_loss, var_loss, mean_loss

    def _gradient_based_dynamic_weighting(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        corr_loss: torch.Tensor,
        var_loss: torch.Tensor,
        mean_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        epsilon = torch.finfo(pred.dtype).eps
        true_series = true.unsqueeze(1)
        pred_series = pred.unsqueeze(1)
        true_mean = torch.mean(true_series, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred_series, dim=-1, keepdim=True)
        true_var = torch.var(true_series, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred_series, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var.clamp_min(epsilon))
        pred_std = torch.sqrt(pred_var.clamp_min(epsilon))
        true_pred_cov = torch.mean(
            (true_series - true_mean) * (pred_series - pred_mean),
            dim=-1,
            keepdim=True,
        )
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)

        corr_gradient = torch.autograd.grad(
            corr_loss,
            pred,
            retain_graph=True,
            create_graph=False,
        )[0]
        var_gradient = torch.autograd.grad(
            var_loss,
            pred,
            retain_graph=True,
            create_graph=False,
        )[0]
        mean_gradient = torch.autograd.grad(
            mean_loss,
            pred,
            retain_graph=True,
            create_graph=False,
        )[0]
        gradient_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0
        gradient_avg_norm = gradient_avg.norm().detach().clamp_min(epsilon)
        alpha = gradient_avg_norm / corr_gradient.norm().detach().clamp_min(epsilon)
        beta = gradient_avg_norm / var_gradient.norm().detach().clamp_min(epsilon)
        gamma = gradient_avg_norm / mean_gradient.norm().detach().clamp_min(epsilon)
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()
        return alpha, beta, gamma


def _build_mse_term(
    params: dict[str, Any],
    context: LossBuildContext,
) -> LossFn:
    _ = context
    if params:
        raise ValueError("mse loss term does not accept params.")
    return MaskedMSELossTerm()


def _build_hcan_auxiliary_term(
    params: dict[str, Any],
    context: LossBuildContext,
) -> LossFn:
    if params:
        raise ValueError("hcan_auxiliary loss term does not accept params.")
    return HCANAuxiliaryLossTerm(
        train_targets=context.train_targets,
        train_mask=context.train_mask,
        annealing_steps=context.annealing_steps,
    )


def _build_fourier_amplitude_correlation_term(
    params: dict[str, Any],
    context: LossBuildContext,
) -> LossFn:
    mode = params.get("mode")
    alpha = params.get("alpha")
    mask_mode = params.get("mask_mode", "strict_real_only")
    if not isinstance(mode, str):
        raise ValueError("fourier_amplitude_correlation requires a string mode.")
    if alpha is not None and not isinstance(alpha, (int, float)):
        raise ValueError("fourier_amplitude_correlation alpha must be numeric.")
    if not isinstance(mask_mode, str):
        raise ValueError(
            "fourier_amplitude_correlation mask_mode must be a string."
        )
    return FourierAmplitudeCorrelationLossTerm(
        mode=mode,
        alpha=None if alpha is None else float(alpha),
        mask_mode=mask_mode,
        total_train_steps=context.total_train_steps,
    )


def _build_patch_wise_structural_term(
    params: dict[str, Any],
    context: LossBuildContext,
) -> LossFn:
    _ = context
    patch_len_threshold = params.get("patch_len_threshold")
    mask_mode = params.get("mask_mode")
    if not isinstance(patch_len_threshold, int):
        raise ValueError(
            "patch_wise_structural patch_len_threshold must be an integer."
        )
    if not isinstance(mask_mode, str):
        raise ValueError("patch_wise_structural mask_mode must be a string.")
    return PatchWiseStructuralLossTerm(
        patch_len_threshold=patch_len_threshold,
        mask_mode=mask_mode,
    )


def build_forecast_loss(
    loss_config: LossConfigProtocol | None = None,
    *,
    train_targets: FloatArray | None = None,
    train_mask: BoolArray | None = None,
    annealing_steps: int = 1,
    total_train_steps: int = 1,
) -> CompositeLoss:
    config = _default_loss_config() if loss_config is None else loss_config
    if config.name != "composite":
        raise ValueError(f"Unsupported loss config name: {config.name!r}.")
    context = LossBuildContext(
        train_targets=train_targets,
        train_mask=train_mask,
        annealing_steps=annealing_steps,
        total_train_steps=total_train_steps,
    )
    weighted_terms: list[tuple[float, LossFn]] = []
    for term_config in config.terms:
        if term_config.weight <= 0.0:
            continue
        builder = cast(LossTermBuilder, LOSSES.get(term_config.name))
        term = builder(dict(term_config.params), context)
        weighted_terms.append((float(term_config.weight), term))
    return CompositeLoss(weighted_terms)


if "mse" not in LOSSES.keys():
    LOSSES.register("mse", cast(Any, _build_mse_term))
if "hcan_auxiliary" not in LOSSES.keys():
    LOSSES.register("hcan_auxiliary", cast(Any, _build_hcan_auxiliary_term))
if "fourier_amplitude_correlation" not in LOSSES.keys():
    LOSSES.register(
        "fourier_amplitude_correlation",
        cast(Any, _build_fourier_amplitude_correlation_term),
    )
if "patch_wise_structural" not in LOSSES.keys():
    LOSSES.register(
        "patch_wise_structural",
        cast(Any, _build_patch_wise_structural_term),
    )
