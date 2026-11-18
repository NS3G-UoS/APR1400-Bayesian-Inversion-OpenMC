# APR1400 Bayesian Inversion Pipeline

This repository now follows a physically motivated four-dimensional design for APR1400 inverse analysis.

## Scientific objective

The project is designed to recover an unknown design parameter from real stochastic OpenMC transport calculations using a heteroscedastic Gaussian-process surrogate and a deterministic Bayesian posterior with no Markov-chain Monte Carlo. The active inverse workflow is therefore a direct, grid-based posterior evaluation over the feasible design domain, with each case validated by genuinely blind leave-one-out inversion against real OpenMC output.

The active design variables are:

- alpha_gd: effective gadolinium loading multiplier; in the code this is the parameter used to build the Gd₂O₃ mixture via w = 0.08 × α, so it is not treated as a literal Gd₂O₃ mass fraction unless the chemistry model is explicitly written that way.
- fuel_temp: fuel temperature in K
- moderator_temp: moderator temperature in K
- boron_ppm: soluble boron concentration in ppm

The current workflow is no longer a single alpha-only sweep. Instead, Phase I generates OpenMC forward responses over a feasible 4D design matrix, and Phase II fits surrogate models using those four inputs.

## Current design philosophy

The active forward model is a 4D DOE over the feasible domain:

- alpha_gd: 0.25 to 1.25
- fuel_temp: 294 to 1000 K
- moderator_temp: 294 to 600 K
- boron_ppm: 0 to 2000 ppm
- constraint: fuel_temp >= moderator_temp

This is implemented as a design-of-experiments workflow rather than a simple alpha scan. The design generator intentionally avoids physically infeasible regions and adds replicate points in the highest-noise region for uncertainty checks.

In full mode, the DOE also injects targeted slice points that independently vary moderator temperature and boron concentration at fixed anchors. This improves GP identifiability for those dimensions and reduces boundary-hitting length-scale warnings.

## Core workflow

1. Phase I: generate OpenMC cases for a 4D feasible design matrix.
2. Phase II: fit Gaussian-process and comparison surrogates on the 4 features.
3. Use genuinely blind leave-one-out validation to recover the inverse relationship and compare model performance.

This is not a synthetic or noise-free reconstruction. Each held-out stochastic OpenMC result is removed from training, the GP is refitted, and the posterior is recomputed without MCMC. The workflow therefore tests whether the unknown design parameters can be recovered from real Monte Carlo transport data in a way that mirrors the paper objective.

Legacy alpha-only runs are kept only in the archive for historical reference and are not the active design path.

## Main scripts

- `openmc_apr1400_phase1_runner_final.py`: Phase I OpenMC runner for the 4D design and legacy fallback support.
- `phase2_konly_analysis/src/phase2_analysis.py`: Phase II analysis for 4D and alpha-only datasets.
- `phase2_konly_analysis/src/generate_multidim_openmc_design.py`: builds the feasible 4-feature DOE matrix.
- `phase2_konly_analysis/src/multidim_gp_analysis.py`: prototype 4D GP analysis and diagnostic plotting.
- `run_apr1400_pipeline.py`: orchestration entrypoint for the legacy run layout; the modern design path is driven by the 4D Phase I input matrix.

## Repository structure

```text
.
├── README.md
├── run_apr1400_pipeline.py
├── openmc_apr1400_phase1_runner_final.py
├── generatecrosssection.py
├── config.yaml
├── archive/
│   └── legacy_root/
├── phase2_konly_analysis/
│   ├── README.md
│   ├── inputs/
│   ├── multidim_design/
│   │   └── openmc_4d_design_matrix.csv
│   ├── multidim_gp_outputs/
│   └── src/
│       ├── phase2_analysis.py
│       ├── generate_multidim_openmc_design.py
│       └── multidim_gp_analysis.py
├── runs/
│   └── YYYYMMDD_HHMMSS_<mode>_<suffix>/
│       ├── phase1/
│       │   ├── raw_data/
│       │   ├── processed_data/
│       │   ├── reports/
│       │   └── private_truth/
│       ├── phase2/
│       │   ├── results/
│       │   └── plots/
│       └── pipeline_summary_<mode>.json
└── llm_responses/
```

The active project path is the 4D design workflow. `archive/legacy_root/` contains earlier alpha-only experiments that are no longer the default delivery path.

## Running Phase I and Phase II

The project supports two execution modes:

- pilot: small 4D design for quick testing
- full: realistic benchmark 4D design for the complete analysis

### One-command end-to-end run

```bash
python run_apr1400_pipeline.py --pilot
```

```bash
python run_apr1400_pipeline.py --full

python run_apr1400_pipeline.py --full --workers 4 --threads 1 --n-unique-full 24 --particles 6000 --batches 60 --inactive 20 --cv-folds 10

```

This command runs Phase I first, validates the generated dataset, and then runs Phase II on the same output folder so the two stages are directly linked.

The validation path in the active code is intentionally conservative: the default Phase II workflow is blind leave-one-out recovery, not a 10-fold CV. The `--cv-folds` flag is accepted only for compatibility with pipeline tooling; it does not replace the genuine LOO holdout logic used in the primary inverse evaluation.

Pilot mode now defaults to lightweight OpenMC settings for fast error detection:

- particles: 200
- batches: 12
- inactive: 3

Full mode keeps a realistic benchmark default:

- particles: 6000
- batches: 60
- inactive: 20
- n-unique-full: 24

You can override these defaults explicitly with `--particles`, `--batches`, and `--inactive`.

Validation behavior follows mode intent: pilot accepts a compact 4D design with narrower alpha coverage, while full expects broad support suitable for no-extrapolation inverse analysis.

### Phase I only

The Phase I runner supports the active multidimensional DOE and can also be used directly:

```bash
python openmc_apr1400_phase1_runner_final.py --multidim --pilot --run-output-dir /path/to/output_dir
```

```bash
python openmc_apr1400_phase1_runner_final.py --multidim --full --run-output-dir /path/to/output_dir
```

In `--full` mode, the internal design includes targeted moderator and boron slices automatically. In `--pilot` mode, the design remains compact and does not add those augmentation slices.

### Use a saved 4D design CSV

```bash
python openmc_apr1400_phase1_runner_final.py --design-csv phase2_konly_analysis/multidim_design/openmc_4d_design_matrix.csv --run-output-dir /path/to/output_dir
```

### Phase II only

Phase II consumes the Phase I CSV and performs the inverse-analysis workflow.

```bash
python phase2_konly_analysis/src/phase2_analysis.py \
  --dataset-csv /path/to/openmc_dataset.csv \
  --output-dir /path/to/phase2/results \
  --plots-dir /path/to/phase2/plots
```

If the dataset is blind and does not contain `alpha_gd`, pass the matching secret lookup file:

```bash
python phase2_konly_analysis/src/phase2_analysis.py \
  --dataset-csv /path/to/blind_dataset.csv \
  --secret-alpha-lookup /path/to/blind_alpha_lookup.json
```

Phase II range checks are mode-aware:

- `--pilot`: allows a narrower alpha span from the pilot DOE
- `--full`: enforces broad alpha support across the design range for robust no-extrapolation posterior fitting

The 4D run is the recommended path for paper-quality modeling and forward simulations.

## 4D GP and model structure

The active surrogate model is built on the 4D feature space.

- `alpha_gd`
- `fuel_temp`
- `moderator_temp`
- `boron_ppm`

The GP uses an ARD Matérn kernel with standardization across inputs, which makes the fitted length scales interpretable as feature sensitivity indicators.

Phase I base 4D points are generated using Sobol quasi-random sampling (space-filling) and include edge-concentration diagnostics per feature. Phase II prints explicit GP diagnostics when length scales approach optimization bounds.

The Phase II comparison still includes weighted polynomial baselines for reference, but the main scientific design is the 4D GP response surface.

## Output layout per run

Each run writes a timestamped directory under `runs/`:

```text
runs/YYYYMMDD_HHMMSS_<mode>_<suffix>/
├── phase1/
│   ├── raw_data/
│   ├── processed_data/
│   │   ├── openmc_dataset.csv
│   │   ├── blind_observations.json
│   │   └── ...
│   ├── reports/
│   ├── logs/
│   ├── plots/
│   └── private_truth/
├── phase2/
│   ├── results/
│   └── plots/
├── pipeline_summary_<mode>.json
└── failure_report.json (only on failure)
```

The key dataset columns for the active design are:

```text
alpha_gd, fuel_temp, moderator_temp, boron_ppm, keff, keff_std, success, case_dir, statepoint, ...
```

## Environment setup

From the repository root:

```bash
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
```

Optional chain files may also be needed depending on the OpenMC library configuration.

### Build a cross_sections.xml file

```bash
python generatecrosssection.py --data-dir /path/to/endf_hdf5_library --output /path/to/endf_hdf5_library/cross_sections.xml
```

Then export the resulting XML path before running the pipeline.

## Notes on legacy artifacts

`archive/legacy_root/` contains obsolete alpha-only code and analysis artifacts that are kept for provenance only. They are preserved for comparison and debugging, but the current supported scientific workflow is the 4D design.
