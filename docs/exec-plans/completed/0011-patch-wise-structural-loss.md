## Goal

Integrate Patch-wise Structural Loss from the paper as a reusable loss term in
the unified training framework, with explicit mask handling and focused
15min-compatible experiment configs.

## Scope

- extend loss config validation with a new `patch_wise_structural` term;
- implement the paper's adaptive patching, patch-level structural components,
  and dynamic weighting inside the existing loss registry path;
- keep the single trainer and single evaluator entry points unchanged;
- add focused configs, docs, and tests for the new loss-backed method.

## Non-goals

- no paper-specific trainer, evaluator, dataset, or training script;
- no new forecasting backbone pretending to be the paper method;
- no real-data or long GPU experiments.

## Steps

1. Add config-schema validation for PS-loss params and document the adaptation
   choice that this paper is integrated as a generic loss term, not a model.
2. Implement the PS loss term in `windlab.losses`, including explicit
   project-level mask semantics and training-only behavior that matches the
   official code's MSE-only validation path.
3. Add minimal experiment YAMLs, focused unit coverage, and one tiny smoke test,
   then run only the related lint/type/test scope before closing the plan.
