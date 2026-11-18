#!/usr/bin/env python3
"""Multidimensional GP workflow for APR1400 inverse study.

This file adds a robust 4-feature GP analysis scaffold that mirrors the paper's
inverse-program logic but extends the forward model beyond the original single
scalar alpha_gd view. The implementation is intentionally conservative:

- alpha_gd remains the effective absorber-strength parameter
- fuel_temp and moderator_temp are treated as physically meaningful thermal state variables
- boron_ppm enters as a soluble boron / poison term
- the design is constrained to the feasible domain fuel_temp >= moderator_temp
- all continuous features are standardized before fitting
- the kernel uses an ARD Matérn-5/2 kernel for interpretable length scales
- replication is included in the highest-noise region for uncertainty checks

This script is intended to be run with a proper OpenMC-generated CSV containing
columns:
    alpha_gd, fuel_temp, moderator_temp, boron_ppm, keff, keff_std

If such a CSV is not available, the script falls back to a synthetic demo dataset
that preserves the intended experimental design and demonstrates the data
processing, GP fitting, and interpretability outputs.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "phase2_konly_analysis" / "multidim_gp_outputs"

FEATURE_RANGES = {
    "alpha_gd": (0.25, 1.25),
    "fuel_temp": (294.0, 1000.0),
    "moderator_temp": (294.0, 600.0),
    "boron_ppm": (0.0, 2000.0),
}

MODEL_FEATURES = ["alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm"]
MAX_LOGICAL_DELTA_T = 1000.0
GP_LENGTH_SCALE_BOUNDS = (1e-2, 300.0)
GP_AMPLITUDE_BOUNDS = (1e-6, 1e3)


def feasible_points(points: np.ndarray) -> np.ndarray:
    """Keep only points satisfying fuel_temp >= moderator_temp."""
    fuel = points[:, 1]
    mod = points[:, 2]
    mask = fuel >= mod
    return points[mask]


def generate_feasible_design(
    n_unique: int = 48,
    n_replicates: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a feasible design for the 4-feature GP study.

    We use Sobol quasi-random sampling for better space-filling properties than
    pseudo-random draws, then enforce fuel_temp >= moderator_temp by swapping.
    """
    m = int(np.ceil(np.log2(max(1, n_unique))))
    sobol = qmc.Sobol(d=4, scramble=True, seed=random_seed)
    pts = sobol.random_base2(m)
    if len(pts) < n_unique:
        pts = np.vstack([pts, sobol.random(n_unique - len(pts))])
    pts = pts[:n_unique]
    l_bounds = np.array([FEATURE_RANGES[k][0] for k in MODEL_FEATURES], dtype=float)
    u_bounds = np.array([FEATURE_RANGES[k][1] for k in MODEL_FEATURES], dtype=float)
    scaled = qmc.scale(pts, l_bounds=l_bounds, u_bounds=u_bounds)

    points: List[List[float]] = []
    for row in scaled:
        alpha, fuel, mod, boron = [float(v) for v in row]
        if fuel < mod:
            fuel, mod = mod, fuel
        points.append([alpha, fuel, mod, boron])

    design = pd.DataFrame(points, columns=MODEL_FEATURES)
    design["replicate"] = False

    noisy_region = (
        (design["alpha_gd"] >= 0.75)
        & (design["boron_ppm"] >= 1200.0)
        & (design["fuel_temp"] >= 700.0)
        & (design["moderator_temp"] >= 294.0)
    )
    noisy_candidates = design.loc[noisy_region].copy()
    if len(noisy_candidates) == 0:
        noisy_candidates = design.sample(min(n_replicates, len(design)), random_state=random_seed)

    replicates = noisy_candidates.head(min(n_replicates, len(noisy_candidates))).copy()
    replicates["replicate"] = True
    design = pd.concat([design, replicates], ignore_index=True)
    design = design.reset_index(drop=True)
    design["design_id"] = np.arange(len(design))
    design["remarks"] = "feasible_region_sample"
    design.loc[design["replicate"], "remarks"] = "replicate_noisy_region"

    return design


def synthetic_keff_model(df: pd.DataFrame) -> pd.DataFrame:
    """Create a smooth synthetic response for demonstration and code verification.

    This intentionally mirrors the physically expected response shape: as alpha_gd,
    boron, and delta-temperature increase, keff generally decreases, while moderator
    temperature drives a secondary reactivity trend.
    """
    x = df[MODEL_FEATURES].copy()
    delta_t = x["fuel_temp"] - x["moderator_temp"]
    keff = (
        1.26
        - 0.075 * x["alpha_gd"]
        - 0.000039 * x["boron_ppm"]
        - 0.00018 * delta_t
        + 0.00014 * x["moderator_temp"]
        + 0.000025 * x["fuel_temp"]
    )
    keff = keff.astype(float)
    keff_std = 0.0006 + 0.00018 * (x["boron_ppm"] / 2000.0) + 0.00012 * (x["alpha_gd"] / 1.25)
    if "replicate" in df.columns:
        # intentionally enlarged uncertainty in the noisiest/collocated region
        keff_std = keff_std + 0.0002 * df["replicate"].astype(float)
    out = df.copy()
    out["keff"] = keff.to_numpy() + np.random.default_rng(123).normal(0.0, keff_std.to_numpy(), size=len(out))
    out["keff_std"] = keff_std.to_numpy()
    return out


def make_demo_dataset() -> pd.DataFrame:
    design = generate_feasible_design(n_unique=48, n_replicates=10)
    dataset = synthetic_keff_model(design)
    return dataset


def fit_multidim_gp(df: pd.DataFrame, response_col: str = "keff", noise_col: str = "keff_std") -> Dict[str, object]:
    """Fit an ARD Matérn-5/2 GP to the structured four-feature dataset.

    The model learns which variables matter most through the fitted kernel length
    scales. Shorter length scales indicate higher sensitivity; longer scales
    indicate weaker or smoother effects.
    """
    required = MODEL_FEATURES + [response_col, noise_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for multidimensional GP: {missing}")

    X = df[MODEL_FEATURES].to_numpy(dtype=float)
    y = df[response_col].to_numpy(dtype=float)
    sigma = df[noise_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(sigma)):
        raise ValueError("Input data contains NaN or Inf values for the GP fit.")
    if np.any(sigma <= 0):
        raise ValueError("keff_std must be positive for heteroscedastic GP fitting.")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_X.fit_transform(X)
    ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    alpha_noise = (sigma / np.std(y)) ** 2 + 1e-10

    kernel = ConstantKernel(1.0, GP_AMPLITUDE_BOUNDS) * (
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-6, 1e3))
        + Matern(
            length_scale=np.ones(X.shape[1], dtype=float),
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
            nu=2.5,
        )
    )
    n_restarts = 5 if len(df) <= 20 else 8
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_noise,
        normalize_y=False,
        n_restarts_optimizer=n_restarts,
        random_state=42,
    )
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(Xs, ys)
    conv = [str(w.message) for w in ws if issubclass(w.category, ConvergenceWarning)]

    if not hasattr(gp, "kernel_"):
        raise RuntimeError("GaussianProcessRegressor did not produce an optimized kernel.")
    def _find_ls_kernel(k):
        if hasattr(k, "length_scale") and hasattr(k, "length_scale_bounds"):
            return k
        if hasattr(k, "k1"):
            f = _find_ls_kernel(k.k1)
            if f is not None:
                return f
        if hasattr(k, "k2"):
            f = _find_ls_kernel(k.k2)
            if f is not None:
                return f
        return None

    ls_kernel = _find_ls_kernel(gp.kernel_)
    if ls_kernel is None:
        raise RuntimeError("Could not locate Matern length_scale component in GP kernel.")
    length_scales = np.asarray(ls_kernel.length_scale, dtype=float)

    raw_feature_strength = 1.0 / np.maximum(length_scales, 1e-12)
    total_strength = raw_feature_strength.sum()
    feature_importance = raw_feature_strength / total_strength
    importance_df = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "length_scale": length_scales,
            "inverse_length_scale": raw_feature_strength,
            "relative_importance": feature_importance,
        }
    ).sort_values("relative_importance", ascending=False)

    return {
        "gp": gp,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "X": X,
        "y": y,
        "sigma": sigma,
        "feature_importance": importance_df,
        "length_scales": length_scales,
        "kernel_summary": str(gp.kernel_),
        "gp_diagnostics": {
            "n_restarts_optimizer": int(n_restarts),
            "convergence_warnings": conv,
            "length_scale_bounds": [float(GP_LENGTH_SCALE_BOUNDS[0]), float(GP_LENGTH_SCALE_BOUNDS[1])],
            "near_upper_features": [
                MODEL_FEATURES[i] for i, v in enumerate(length_scales)
                if np.isclose(float(v), float(GP_LENGTH_SCALE_BOUNDS[1]), rtol=0.0, atol=0.02 * float(GP_LENGTH_SCALE_BOUNDS[1]))
            ],
        },
    }


def save_length_scale_plot(importance_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(importance_df["feature"], importance_df["relative_importance"], color="tab:blue")
    ax.set_ylabel("Relative importance proxy (1 / length scale)")
    ax.set_xlabel("Feature")
    ax.set_title("GP learned feature importance from ARD Matérn length scales")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_design_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    xs = df["alpha_gd"].to_numpy()
    ys = df["fuel_temp"].to_numpy()
    zs = df["boron_ppm"].to_numpy()
    ax.scatter(xs, ys, zs, c=df["moderator_temp"].to_numpy(), cmap="viridis", s=35)
    ax.set_xlabel("alpha_gd")
    ax.set_ylabel("fuel_temp (K)")
    ax.set_zlabel("boron_ppm")
    ax.set_title("Feasible 4-feature design with replicated high-noise region")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_prediction_slice(gp_info: Dict[str, object], out_path: Path) -> None:
    gp = gp_info["gp"]
    scaler_X = gp_info["scaler_X"]
    X = gp_info["X"]
    y = gp_info["y"]

    alpha_grid = np.linspace(0.25, 1.25, 101)
    fuel_grid = np.linspace(294.0, 1000.0, 101)
    moderator_median = np.median(X[:, 2])
    boron_median = np.median(X[:, 3])

    grid = np.array(
        [(a, f, moderator_median, boron_median) for a, f in zip(alpha_grid, fuel_grid)],
        dtype=float,
    )
    scaled = scaler_X.transform(grid)
    mu, sigma = gp.predict(scaled, return_std=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(alpha_grid, mu, color="tab:blue", lw=2, label="GP mean")
    ax.fill_between(alpha_grid, mu - 1.96 * sigma, mu + 1.96 * sigma, color="tab:blue", alpha=0.2, label="95% uncertainty")
    ax.set_xlabel("alpha_gd")
    ax.set_ylabel("Predicted keff")
    ax.set_title("4D GP slice at median moderator_temp and boron_ppm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_demo() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    design = generate_feasible_design(n_unique=48, n_replicates=10, random_seed=42)
    design.to_csv(OUTPUT_DIR / "multidim_gp_design.csv", index=False)
    save_design_plot(design, OUTPUT_DIR / "multidim_gp_design.png")

    demo = synthetic_keff_model(design)
    demo.to_csv(OUTPUT_DIR / "multidim_gp_demo_dataset.csv", index=False)

    gp_info = fit_multidim_gp(demo)
    importance_df = gp_info["feature_importance"]
    importance_df.to_csv(OUTPUT_DIR / "feature_length_scales.csv", index=False)
    save_length_scale_plot(importance_df, OUTPUT_DIR / "feature_length_scales.png")
    save_prediction_slice(gp_info, OUTPUT_DIR / "gp_prediction_slice.png")

    summary = {
        "features": MODEL_FEATURES,
        "ranges": FEATURE_RANGES,
        "design_rows": int(len(design)),
        "replicate_rows": int(design["replicate"].sum()),
        "length_scales": gp_info["length_scales"].tolist(),
        "feature_importance": importance_df.to_dict(orient="records"),
        "kernel_summary": gp_info["kernel_summary"],
        "gp_diagnostics": gp_info["gp_diagnostics"],
        "feasibility_constraint": "fuel_temp >= moderator_temp",
        "library_coverage_note": "Check cross-section library availability: 294, 600, 900, and 1200 K should be confirmed before using 1000 K points; 1000 K may require interpolation between 900 and 1200 K.",
    }
    (OUTPUT_DIR / "multidim_gp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved multidimensional GP planning outputs to: {OUTPUT_DIR}")
    print("Feature importance summary:")
    print(importance_df.to_string(index=False))
    diag = gp_info["gp_diagnostics"]
    print(f"GP restarts used: {diag['n_restarts_optimizer']}")
    if diag.get("near_upper_features"):
        print(f"[warn] GP length-scale near upper bound for: {diag['near_upper_features']}")
    for msg in diag.get("convergence_warnings", []):
        print(f"[warn] {msg}")


if __name__ == "__main__":
    run_demo()
