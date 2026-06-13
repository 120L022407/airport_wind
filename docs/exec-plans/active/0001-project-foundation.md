# 0001 Project Foundation

> Status: Active. Move to `docs/exec-plans/completed/` when finished.

## 1. Goal

Build a small, maintainable, configuration-driven airport wind-speed
forecasting framework.

The datasets are already preprocessed and already divided into train,
validation, and test splits. Do not recreate cleaning, interpolation, or
chronological splitting.

This milestone provides:

- one training entry point and one evaluation entry point;
- YAML configurations grouped by model;
- a lightweight registry;
- loaders for current data sources;
- train-only normalization and split-local windows;
- one baseline training/evaluation path;
- focused CPU tests;
- minimum reproducibility artifacts.

## 2. Data source of truth

The authoritative human-readable data definition is `docs/data.md`.

It documents:

- `series`;
- `series_15min`;
- `EC`;
- filenames, shapes, ordering, units, masks, resolutions, channels, and grids.

Runtime `metadata.json` files must also be validated by loaders.

Do not duplicate data details in this plan or `ARCHITECTURE.md`.

Read `docs/data.md` only when changing data loading, data configuration,
normalization, windows, masks, variable selection, or source alignment.

## 3. Data location

Research data must not be committed to Git.

Recommended server layout:

```text
/mnt/md0/sh/datasets/airport_wind/
├── series/
├── series_15min/
└── EC/
```

Optional local layout:

```text
airport_wind_lab/data/
├── series/
├── series_15min/
└── EC/
```

Set the root with:

```bash
export AIRPORT_WIND_DATA_ROOT=/mnt/md0/sh/datasets/airport_wind
```

Experiment YAML:

```yaml
data:
  root: ${AIRPORT_WIND_DATA_ROOT}
```

Do not hardcode machine-specific paths in Python.

`data/README.md` only explains placement and links to `docs/data.md`.
Ignore local `data/` in Git.

## 4. Non-goals

This milestone does not:

- regenerate, interpolate, or repartition data;
- implement ECMWF or several models at once;
- create experiment-specific Trainer classes or copied training scripts;
- create dynamic plugins or deep configuration inheritance;
- create full AST/import architecture checks;
- run full GPU training or test on real research data;
- implement all future ramp or turning-point methods.

## 5. Target structure

```text
airport_wind_lab/
├── AGENTS.md
├── ARCHITECTURE.md
├── README.md
├── Makefile
├── pyproject.toml
├── .gitignore
├── config/<model>/<experiment>.yaml
├── data/README.md
├── docs/
│   ├── data.md
│   ├── decisions/
│   └── exec-plans/{active,completed}/
├── scripts/{train.py,evaluate.py}
├── src/windlab/
│   ├── config.py
│   ├── registry.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── losses.py
│   ├── metrics.py
│   ├── utils.py
│   ├── data/{loaders.py,series.py,normalization.py,windows.py}
│   └── models/<first-model>.py
├── tests/
│   ├── test_config.py
│   ├── test_loaders.py
│   ├── test_normalization.py
│   ├── test_windows.py
│   ├── test_registry.py
│   ├── test_model_shapes.py
│   ├── test_metrics.py
│   └── test_smoke.py
└── outputs/
```

Create model files and configuration directories only when implemented.
Ignore data, outputs, arrays, checkpoints, and logs in Git.

## 6. Entry points and configuration

Training:

```bash
python scripts/train.py --config config/<model>/<experiment>.yaml
```

Evaluation:

```bash
python scripts/evaluate.py --run-dir outputs/<run-name>
```

Entry points only parse arguments, load configuration, call reusable
`windlab` functions, and report results.

Do not create model- or experiment-specific train/evaluate scripts.

Configurations are grouped by model, for example:

```text
config/
├── gru/baseline_hourly.yaml
├── itransformer/baseline_hourly.yaml
└── fusion/obs_era5_grid.yaml
```

Each YAML is a complete experiment and may define:

- experiment name, seed, and output root;
- data root, source, selected airports, variables, and targets;
- input and forecast lengths;
- normalization;
- model, loss, optimizer, and trainer settings;
- metrics and evaluation-mask behavior.

Do not introduce deep base/override inheritance.

Shell scripts may select a YAML and override only device or output location.

## 7. Registry

Implement one lightweight registry for:

- data builders;
- models;
- losses;
- metrics.

Keys represent component types such as `obs_series`, `gru`, `mse`, and `mae`,
not experiment names.

Use explicit dictionaries, decorators, or small registration functions.
Do not implement dynamic scanning or external plugins.

Duplicate and unknown keys must raise clear errors.

## 8. Data modules

`data/loaders.py`:

- load `series`, `series_15min`, and `EC`;
- read metadata and original masks;
- validate files, dimensions, shapes, and metadata;
- never change time resolution.

`data/series.py`:

- organize existing train, validation, and test splits;
- select configured airports, variables, and targets;
- attach optional masks and ERA5 arrays;
- validate alignment;
- never recreate splits.

`data/normalization.py`:

- fit statistics on train only;
- reuse the same state for validation and test;
- save and restore the state;
- record variables and normalization axes.

`data/windows.py`:

- construct windows independently inside each split;
- prevent windows from crossing split boundaries;
- align target masks with target times;
- expose explicit, tested batch shapes.

Different resolutions must not be silently repeated, broadcast, or
interpolated. Alignment requires explicit configuration and a focused test.

## 9. Model, training, and evaluation

Models are added through the registry and model-specific YAML files.

Models must accept documented batch fields, return a documented prediction
shape, avoid file loading/normalization, and avoid experiment-name branches.

Minimum output:

```text
{"prediction": tensor, "aux": dict}
```

`trainer.py` provides one generic flow for construction, optimization,
validation, checkpoint selection, and artifact saving.

Do not create experiment-specific Trainer subclasses.

`evaluator.py` loads saved configuration, checkpoint, and normalization state;
generates predictions; applies masks; and saves metrics.

## 10. Required artifacts

Every successful run must save:

```text
outputs/<run-name>/
├── config.yaml
├── checkpoint.pt
├── metrics.json
└── normalization.npz
```

`normalization.npz` may be omitted only when the complete state is stored in
`checkpoint.pt`.

- `config.yaml`: fully resolved configuration.
- `checkpoint.pt`: model, optimizer, epoch, best validation metric, seed, and
  normalization state when not separate.
- `metrics.json`: validation metrics, optional test metrics, metric names,
  real-observation-only status, and nonstandard thresholds.

Predictions, logs, and a last checkpoint are optional.

## 11. Testing policy

Tests use synthetic or tiny fixture data and run on CPU.

Required coverage:

- configuration loading;
- all three loader paths and metadata/mask validation;
- train-only normalization and restoration;
- split-local windows and mask alignment;
- registry errors;
- model shapes;
- metric mask behavior;
- one minimal smoke path.

Use the smallest relevant test scope:

- documentation: no tests;
- configuration: configuration tests;
- data: affected data tests;
- model: affected shape tests;
- metric: affected metric tests;
- trainer/evaluator: relevant tests plus smoke test.

Run the full suite and lint only when completing this milestone or after a
broad coordinated change.

Do not run unrelated tests after every small change.
Do not run full GPU training.

## 12. Implementation

1. Establish directories, `docs/data.md`, `data/README.md`, `.gitignore`,
   `pyproject.toml`, `Makefile`, and the two entry-point skeletons.
2. Implement configuration loading, registry, one baseline YAML, and tests.
3. Implement loaders, split organization, normalization, windows, and tests.
4. Implement one baseline model, generic Trainer/Evaluator, minimum
   loss/metrics, required artifacts, and CPU smoke test.

Later models and fusion methods are outside this plan unless explicitly added.

## 13. Completion

This milestone is complete when:

- one YAML experiment runs through the single train/evaluate paths;
- registry and loaders work;
- normalization fits train only;
- windows remain inside their split;
- masks restrict validation/test metrics when configured;
- model shape and smoke tests pass;
- required artifacts are saved;
- the final full suite passes.

These criteria apply only when explicitly completing this milestone.
Do not recheck them after every normal code change.

After completion:

```bash
mv docs/exec-plans/active/0001-project-foundation.md    docs/exec-plans/completed/0001-project-foundation.md
```
