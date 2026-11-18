# Phase II 4D inverse analysis

This package contains the active Phase II surrogate-analysis workflow for the APR1400 inverse design.

The current scientific design is a 4-dimensional forward space:

- alpha_gd
- fuel_temp
- moderator_temp
- boron_ppm

The main Phase II runner is:

```bash
python src/phase2_analysis.py --dataset-csv /path/to/openmc_dataset.csv
```

For prototype/model-development diagnostics, use:

```bash
python src/multidim_gp_analysis.py
```

This package also includes a design generator for the feasible DOE:

```bash
python src/generate_multidim_openmc_design.py
```

The generated design is saved under `multidim_design/` as `openmc_4d_design_matrix.csv` and is intended to be consumed directly by the OpenMC Phase I runner.

Environment variables used by OpenMC setup may include:

```bash
OPENMC_CROSS_SECTIONS
OPENMC_CHAIN_FILE_FAST
OPENMC_CHAIN_FILE_THERMAL
```

The older alpha-only analysis files and archived results remain in `archive/legacy_root/` and are no longer the primary project path.