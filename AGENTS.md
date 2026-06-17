- # Project purpose
  
  This repository is a maintainable research framework for airport wind-speed
  forecasting.
  
  The framework will eventually support:
  
  - multi-airport observation sequences;
  - ERA5 point and grid data;
  - 24-hour input to 24-hour forecasting;
  - multiple forecasting models;
  - ramp and turning-point evaluation.
  
  # Current development stage
  
  The project is being built from scratch.
  
  Do not copy code or architecture from previous projects unless explicitly
  requested.
  
  The current priority is to establish architecture, interfaces, configuration,
  tests, and repository rules before implementing multiple training models.

  ## Default experiment preference

  Unless a task explicitly requires another source or resolution, new training
  baselines and model-to-model comparisons should default to:

  - data source: `series_15min`;
  - task: past 24 hours to next 24 hours, i.e. `input_steps=96` and
    `forecast_steps=96`;
  - target airport: `ZGSZ`.

  Hourly (`series`) configs remain supported for comparison and backward
  compatibility, but they are no longer the default experimental starting point.
  
  # Core rules
  
  1. There must be exactly one Python training entry point.
  2. There must be exactly one Python evaluation entry point.
  3. New experiments must be represented by configuration, not copied scripts.
  4. Do not create experiment-specific Trainer classes.
  5. Data shapes, variable names, units, masks, and time splits must be explicit.
  6. Wind speed uses `m/s` internally.
  7. Normalization statistics must only be fitted on the training split.
  8. Train, validation, and test data must preserve chronological order.
  9. Validation and test metrics must support evaluation only on real observed
     points.
  10. Tests must use synthetic or tiny fixture data, not real research data.
  
  # Architecture rules
  
  Generic modules must not contain experiment-name branches.
  
  Models, data builders, losses, and metrics must use registries or documented
  interfaces.
  
  Do not add model-specific or experiment-specific `train_*.py` or
  `evaluate_*.py` entry points.
  
  Shell scripts may select a configuration and override operational settings,
  but must not become a second configuration system.
  
  # Codex workflow
  
  Create an execution plan only when a task changes:
  
  - public data shapes, units, or mask semantics;
  - configuration schema;
  - registry interfaces;
  - training or evaluation entry points;
  - dependency boundaries between major modules;
  - required experiment artifact formats;
  - coordinated behavior across three or more production modules.
  
  An execution plan is not required for:
  
  - documentation-only changes;
  - isolated bug fixes;
  - adding or updating tests;
  - changes limited to one production module and its tests;
  - adding a model through an existing registry interface;
  - adding an experiment YAML through existing interfaces;
  - formatting, naming, logging, comment, or message changes.
  
  For tasks requiring an execution plan:
  
  1. Read `AGENTS.md` and the relevant sections of `ARCHITECTURE.md`.
  2. Create a concise plan under `docs/exec-plans/active/`.
  3. Wait for user approval only when changing architecture or public contracts.
  4. Move the plan to `docs/exec-plans/completed/` after implementation.
  
  # Data documentation
  
  `docs/data.md` is the human-readable source of truth for supported dataset
  layouts, shapes, variable ordering, units, masks, and time resolutions.
  
  Runtime metadata files are the executable source of truth and must be validated
  against the documented data definitions.
  
  Read `docs/data.md` only when changing:
  
  - data loading or data configuration;
  - airport or variable selection;
  - normalization;
  - window construction;
  - mask handling;
  - alignment between data sources.
  
  Do not read `docs/data.md` for isolated model, loss, metric, logging, or
  documentation changes unless the task depends on data semantics.
  
  # Validation policy
  
  Use the smallest relevant test scope:
  
  - documentation-only changes: no tests;
  - configuration changes: configuration tests only;
  - data changes: relevant loader, normalization, or window tests;
  - model changes: affected model shape tests;
  - metric changes: affected metric tests;
  - training or evaluation pipeline changes: relevant tests plus the smoke test;
  - milestone completion or broad coordinated changes: full test suite and lint.
  
  Do not run unrelated tests after every small change.
  
  # Local execution restrictions
  
  Do not run research training locally.
  
  Local execution is limited to:
  
  - lint and formatting checks;
  - unit tests;
  - configuration validation;
  - model shape tests;
  - tiny synthetic CPU smoke tests;
  - at most one or a few training steps required to verify code connectivity.
  
  Do not locally run:
  
  - real research datasets;
  - multi-epoch training;
  - hyperparameter searches;
  - benchmark experiments;
  - GPU training;
  - long-running jobs.
  
  Actual model training is performed manually by the user on the server.
  
  # Communication
  
  Communicate with the user in Chinese unless the user explicitly requests
  another language.
  
  Keep explanations and completion reports concise.
  
  Code identifiers, configuration keys, commands, filenames, and error messages
  may remain in English.
  
  # Completion report
  
  After implementation, report:
  
  - files changed;
  - behavior changed;
  - tests executed;
  - test results;
  - remaining risks.
