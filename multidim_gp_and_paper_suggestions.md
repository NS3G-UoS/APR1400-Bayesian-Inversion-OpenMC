# Multidimensional GP implementation and paper-change summary

## Scope implemented

The project now includes a new four-feature GP analysis scaffold in [phase2_konly_analysis/src/multidim_gp_analysis.py](phase2_konly_analysis/src/multidim_gp_analysis.py).

### Included changes

- Added a 4-feature GP design with:
  - alpha_gd
  - fuel_temp
  - moderator_temp
  - boron_ppm
- Enforced the feasible design region:
  - fuel_temp >= moderator_temp
- Built a feasible-region design generator rather than a full box sample
- Switched the base design sampler to Sobol quasi-random points for better space-filling behavior
- Added edge-concentration diagnostics (near lower/upper bound occupancy by feature)
- Added replication in the noisiest region to support uncertainty checks
- Standardized all continuous inputs before fitting
- Used a composite GP kernel: linear trend plus ARD Matérn-5/2 residual
- Added feature-importance reporting from inverse length scales
- Added plot outputs for:
  - the 4D design geometry
  - feature length scales / variable importance
  - a GP prediction slice
- Added JSON summaries with kernel diagnostics, feature ranges, replicates, and library-note guidance

## Physical definitions adopted for the manuscript

The following definitions should be stated explicitly in the paper to avoid ambiguity:

- Fuel temperature: use a single Doppler-effective, volume-averaged fuel temperature in K, not a centerline temperature and not a melting-limit value.
- Moderator temperature: use a single bulk coolant / moderator temperature in K at the design pressure, e.g. fixed 2,250 psia.
- alpha_gd: define as an effective absorber-strength multiplier used to build the Gd₂O₃ mixture in the code, with the implemented relation w = 0.08 × α. This is the correct interpretation of the current transport model; it is not a literal Gd₂O₃ mass fraction unless the chemistry model is explicitly written that way.
- alpha_gd lower bound: use the implemented design bound consistently in the manuscript (current active bound is 0.25).

## Design-space bounds and feasible-region rule

The current GP design uses:

- alpha_gd: 0.25 to 1.25
- fuel_temp: 294 to 1000 K
- moderator_temp: 294 to 600 K
- boron_ppm: 0 to 2000 ppm

The implementation enforces:

- fuel_temp >= moderator_temp
- values are sampled only in the feasible region, rather than across the full rectangular box

This is the correct engineering interpretation for a realistic reactor model.

In addition, the implementation now reports edge occupancy diagnostics. If a large fraction of points cluster near any bound, that is surfaced in logs so the DOE can be adjusted before expensive full campaigns.

## Cross-section library coverage note

The paper should explicitly note the library support before using 1000 K design points. The planned design should rely on the available library temperatures, with the likely confirmed set being:

- 294 K
- 600 K
- 900 K
- 1200 K

The 1000 K case should be treated as an interpolated condition unless the library explicitly supports it.

The Phase I workflow now includes a design-level temperature coverage check against discovered neutron and thermal library ranges before transport execution. This provides fail-fast behavior if a proposed DOE exceeds available data support.

## Variable-importance interpretation for the GP

In the composite GP (linear trend + ARD Matérn residual), each feature gets its own Matérn length scale in the residual component. The model learns variable sensitivity through those scales:

- short length scale: strong local sensitivity to the feature
- long length scale: weaker or smoother influence
- the paper should describe feature importance using the inverse length-scale proxy, i.e. relative importance ~ 1 / length_scale

The output file feature_length_scales.csv contains the relevant numbers in a machine-readable form, and the plot feature_length_scales.png is the journal-friendly summary.

## Baseline-comparison guidance

Weighted linear and polynomial baselines can be retained for benchmarking, but they should not be framed as expected winners in a nonlinear, anisotropic 4D physics-response space. Recommended framing:

- GP is the primary surrogate for inference and uncertainty quantification
- weighted linear / quadratic / cubic are reduced-order comparators
- conclusions should be based on reported cross-validated error and calibration metrics, not on an a priori claim that one model must always win

This preserves research rigor and avoids over-claiming. In some smooth regimes a low-order model can match or approach GP error, which is a valid scientific finding.

## Kernel and warning interpretation for the manuscript

The current surrogate uses a composite kernel to separate global and local structure:

- global component: linear trend
- local component: Matérn-5/2 residual with ARD length scales

This remains fully Matérn-based for local nonlinear behavior while improving numerical stability and interpretability in smooth regimes.

For full-size datasets, warnings about Matérn length scales near the upper bound should be interpreted as potential smooth/weakly-sensitive dimensions rather than automatic model failure. These diagnostics should be reported transparently in supplementary material.

## Speed-versus-fidelity policy (hardware aware)

Current defaults are tuned for practical execution on limited workstation hardware:

- pilot mode: small/faster transport settings for integration checks
- full mode: larger but still tractable defaults with explicit overrides

The full DOE default is set to a moderate unique-point count to balance runtime with surrogate fidelity. Report this as a resource-aware design choice and include sensitivity checks where feasible.

## Suggested paper wording updates

### Suggested wording for the model definition

The revised manuscript should describe alpha_gd as an effective absorber-strength calibration parameter rather than a literal Gd2O3 weight fraction, unless the design is explicitly chemistry-based.

The surrogate is a Gaussian process with a linear trend and Matérn-5/2 ARD residual kernel, enabling both global trend capture and local nonlinear correction.

### Suggested wording for the thermal state

Fuel and moderator temperatures should be described as effective operating-state temperatures in the reduced core model, not as centerline or melting-limit values.

### Suggested wording for the design space

The paper should state that the GP is trained over a physically feasible operating region, not the full rectangular design box, and that the constraint fuel_temp >= moderator_temp is enforced. This region is a feasible operating superset, not a strictly physical operating envelope; for example, points such as 1000 K fuel with 294 K moderator are still inside the current design rule and should be described as feasible-but-not-fully-physical operating states unless a stronger ΔT-based parametrization is adopted.

The paper should also note that base design points are generated with a space-filling Sobol strategy in the feasible region, with diagnostics to monitor bound concentration. The active design is therefore a practical benchmark region rather than a fully physics-constrained operating map.

### Suggested wording for uncertainty and design planning

The paper should also state that the experimental design includes replicate observations in the highest-uncertainty region to quantify local MC noise and improve GP stability. The design should report the number of unique design points, the number of replicates, the jitter used in the surrogate, and whether the forward response is k∞ or reactivity.

When reporting warnings, distinguish between transport/data-support failures (critical) and surrogate smoothness diagnostics (informational for full-size datasets). The grouped-fold requirement is also important: replicated design points must remain together in the same validation fold so that a held-out replicate does not leak its twin into training.

The GP’s value should be assessed using predictive uncertainty, not only MAE. Report held-out coverage of k∞ intervals and a proper probabilistic scoring metric such as interval score or negative log predictive density. For the kernel specifically, do not interpret long Matérn length scales as evidence that a feature is irrelevant when a strong linear trend is present; the linear component can dominate the response while the residual smoothness remains broad.

Finally, the paper should describe the actual validation run used: the default inverse evaluation is blind leave-one-out recovery on the real OpenMC cases; the 10-fold flag is a compatibility convenience and is not the primary validation design that supports the paper claim.

## Limitations and reporting constraints for the manuscript

The reduced full-mode benchmark should be described as a resource-aware, transport-backed 4D study rather than an exhaustive reactor-design campaign. The practical limits that should be disclosed are:

- The full 4D OpenMC design is computationally expensive: in the current implementation, the full benchmark uses a moderate DOE size and transport settings chosen to preserve feasibility on typical workstation hardware.
- The study is therefore limited by Monte Carlo wall time and available CPU resources, not by a theoretical lack of design points alone.
- The run should be reported as a benchmark-scale inverse study, with explicit note that larger designs or higher-fidelity particle counts would increase runtime substantially.
- The active design is physically feasible but not intended to cover every possible operating condition in the APR1400 domain; it is a targeted surrogate-design region.
- Inverse recovery quality may be more limited at the edges of the design domain, especially when the training set is sparse near the bound or the GP length scales approach the optimization limits.
- The deterministic Bayesian posterior is grid-based and therefore depends on the chosen prior support and the fidelity of the forward surrogate; it is not a full nonparametric posterior over an unbounded space.
- For the manuscript, it is appropriate to state that the study demonstrates credible recovery of an effective absorber-strength parameter under realistic stochastic transport noise, but not a universal proof that all four input variables are separately identifiable under all operating conditions.
- The active validation mode is genuine blind leave-one-out inversion of the actual OpenMC cases. The 10-fold flag is only a compatibility input and should not be described as the primary validation design.
- The design should also state the number of unique points, replicate points, jitter/noise handling, and whether the forward response is k∞ or reactivity, since this affects all uncertainty statements.

These limitations are important because the reduced benchmark is designed for practical execution while preserving the key scientific objective: a 4D, heteroscedastic-GP, no-MCMC inverse workflow with genuine blind LOO validation.

## Recommended next step

The next practical phase is to run the full OpenMC design matrix with the four features above, using the feasible-region Sobol design plus targeted replication, then report:

- Phase I execution success rates and failure taxonomy
- cross-section temperature-support checks
- GP versus baseline model metrics (MAE/RMSE/coverage) under identical folds
- kernel diagnostics with clear interpretation policy
- explicit runtime and benchmark-size notes tied to the resource-aware full-mode defaults

That is the point at which the 4D GP analysis becomes a fully transport-backed inverse model rather than a design and analysis scaffold.



All OpenMC calculations were performed at a fixed moderator pressure of 2250 psia (15.5 MPa), including the cold-state design points near 294 K. Pressure was not treated as an inferred variable in the current surrogate model; therefore, conclusions are conditional on that constant-pressure assumption.