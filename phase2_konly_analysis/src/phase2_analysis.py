#!/usr/bin/env python3
import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def cumulative_trapezoid(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    return np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)))


PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PHASE2_ROOT / "results"
PLOT_DIR = PHASE2_ROOT / "plots"
PILOT_ALPHA_GRID = [0.25, 0.50, 0.75, 1.00, 1.25]
FULL_EXPECTED_CASES = 65  # legacy 1-D check only; the 4-D path bypasses it
PILOT_EXPECTED_CASES = 5
ALPHA_MIN, ALPHA_MAX, N_GRID = 0.25, 1.25, 5001  # Uniform(ALPHA_MIN, ALPHA_MAX) prior; spacing = 0.0002
MEASUREMENT_SD = 0.00050
TALLY_REL_MEASUREMENT_SD = 0.005  # 0.5% relative measurement uncertainty on Gd fission share

MODEL_SPECS = {
    # --- core models used in the paper (default) ---
    "gp": {"label": "GP", "degree": None, "kind": "gp"},
    "wlin": {"label": "Weighted linear", "degree": 1, "kind": "poly"},
    "wquad": {"label": "Weighted quadratic", "degree": 2, "kind": "poly"},
    "wquad_loo": {"label": "Regularized quadratic (LOO)", "degree": 2, "kind": "poly_loo"},
    # --- extended comparators (enable with --extended or --models) ---
    "gpn": {"label": "GP + nugget", "degree": None, "kind": "gp_nugget"},
    "wcubic_r": {"label": "Weighted quad + cubes", "degree": "cubic_reduced", "kind": "poly_loo"},
    "wcubic": {"label": "Weighted cubic (35 terms)", "degree": 3, "kind": "poly_loo"},
}
CORE_MODELS = ["gp", "wlin", "wquad"]
EXTENDED_MODELS = ["wquad_loo", "gpn", "wcubic_r", "wcubic"]
VALID_EXPERIMENTS = {"learning", "edges", "active"}
POLY_LOO_LAMBDAS = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
CI_LEVELS = (0.50, 0.90, 0.95)
OBS_SCENARIOS = {
    "obs50pcm": MEASUREMENT_SD,
    "obs0": 0.0,
}
STRICT_4D_FEATURES = ["alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm"]
GP_LENGTH_SCALE_BOUNDS_1D = (0.05, 300.0)
GP_LENGTH_SCALE_BOUNDS_4D = (1e-2, 300.0)
GP_AMPLITUDE_BOUNDS = (1e-6, 1e3)
RIDGE_ALPHA = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase II k-only & joint Gd-tally inversion runner")
    parser.add_argument("--pilot", action="store_true", help="Use the 5-point pilot dataset (0.25, 0.50, 0.75, 1.00, 1.25).")
    parser.add_argument("--full", action="store_true", help="Use the full Phase I dataset (default).")
    parser.add_argument("--dataset-csv", required=True, help="Path to Phase I openmc_dataset.csv for the same run folder.")
    parser.add_argument("--secret-alpha-lookup", default=None, help="Optional path to blind_alpha_lookup.json when alpha_gd is absent.")
    parser.add_argument("--output-dir", default=str(RESULT_DIR), help="Directory for Phase II CSV/JSON outputs.")
    parser.add_argument("--plots-dir", default=str(PLOT_DIR), help="Directory for Phase II plot outputs.")
    parser.add_argument("--cv-folds", type=int, default=10, help="Optional grouped CV for replicate-aware evaluation; default Phase II validation remains leave-one-out.")
    parser.add_argument("--models", default=",".join(CORE_MODELS), help=f"Comma list from {list(MODEL_SPECS)}; default = core paper models.")
    parser.add_argument("--extended", action="store_true", help="Add the extended comparators (wquad_loo, GP+nugget, reduced cubic, full cubic) to --models.")
    parser.add_argument("--experiments", default="", help="Comma list of extra experiments: learning,edges,active (written to <output-dir>/extensions).")
    parser.add_argument("--exp-repeats", type=int, default=12, help="Learning-curve & active-learning repeats.")
    parser.add_argument("--exp-sizes", default="12,16,20,28,36,44,52", help="Learning-curve & active-learning training sizes.")
    parser.add_argument("--exp-test-groups", type=int, default=8, help="Sparse alpha groups held out per learning-curve repeat.")
    parser.add_argument("--edge-frac", type=float, default=0.15, help="Tail fraction of each feature held out in the edge-extrapolation test.")
    parser.add_argument("--exp-seed", type=int, default=7)
    parser.add_argument("--exp-measurement-sd", type=float, default=MEASUREMENT_SD)
    parser.add_argument("--tally-rel-sd", type=float, default=TALLY_REL_MEASUREMENT_SD, help="Relative measurement uncertainty for gd_fission_share.")
    args = parser.parse_args(argv)
    if args.pilot and args.full:
        parser.error("Choose either --pilot or --full, not both.")
    return args


def validate_dataset_range(df, pilot_mode: bool = False):
    if "alpha_gd" not in df.columns:
        raise ValueError("Dataset is missing alpha_gd; Phase II cannot be run without alpha labels.")
    alpha = pd.to_numeric(df["alpha_gd"], errors="coerce")
    if alpha.isna().any():
        raise ValueError("alpha_gd contains missing values; this indicates an incomplete or stale Phase I export.")
    lo, hi = float(alpha.min()), float(alpha.max())
    if lo < ALPHA_MIN - 1e-9 or hi > ALPHA_MAX + 1e-9:
        raise ValueError(
            f"Dataset alpha range [{lo:.4f}, {hi:.4f}] is outside the required Phase II support "
            f"[{ALPHA_MIN:.2f}, {ALPHA_MAX:.2f}]. Re-run Phase I across 0.25–1.25 before fitting the posterior."
        )
    if pilot_mode:
        return alpha
    if hi - lo < (ALPHA_MAX - ALPHA_MIN) * 0.9:
        raise ValueError(
            f"Dataset does not span the required posterior support; alpha range is [{lo:.4f}, {hi:.4f}] "
            f"but should cover [{ALPHA_MIN:.2f}, {ALPHA_MAX:.2f}] for a valid no-extrapolation inverse fit."
        )
    return alpha


def load_analysis_frame(data_path: Path, secret_path: Path | None):
    if not data_path.exists():
        raise FileNotFoundError(f"Phase II dataset not found: {data_path}")
    df = pd.read_csv(data_path)

    if "alpha_gd" in df.columns:
        return df

    if secret_path is None:
        raise ValueError(
            "Response-only dataset is being used, but no alpha lookup path was provided. "
            "Pass --secret-alpha-lookup with the matching run's blind_alpha_lookup.json."
        )
    if not secret_path.exists():
        raise FileNotFoundError(f"Secret alpha lookup file not found: {secret_path}")

    with open(secret_path, "r", encoding="utf-8") as f:
        secret = json.load(f)
    alpha_values = secret.get("alpha_by_row")
    if alpha_values is None or len(alpha_values) != len(df):
        raise ValueError(
            f"Secret alpha lookup is missing or mismatched for {data_path}. "
            f"Dataset rows={len(df)}, alpha_lookup rows={0 if alpha_values is None else len(alpha_values)}."
        )
    df = df.copy()
    df["alpha_gd"] = np.asarray(alpha_values, dtype=float)
    return df


def validate_mode_case_count(df: pd.DataFrame, pilot_mode: bool):
    multidim_features = {"alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm"}
    if multidim_features.issubset(df.columns):
        return
    expected = PILOT_EXPECTED_CASES if pilot_mode else FULL_EXPECTED_CASES
    actual = int(len(df))
    if actual != expected:
        mode_tag = "pilot" if pilot_mode else "full"
        raise ValueError(
            f"Mode/data mismatch for {mode_tag}: expected exactly {expected} successful rows, found {actual}. "
            "Regenerate Phase I in the same mode and rerun."
        )


def gp_feature_columns(train: pd.DataFrame):
    if all(col in train.columns for col in STRICT_4D_FEATURES):
        return STRICT_4D_FEATURES
    raise ValueError(
        "Phase II requires the 4-feature dataset columns: alpha_gd, fuel_temp, moderator_temp, boron_ppm. "
        "Legacy alpha-only mode is no longer supported in this strict 4D workflow."
    )


def has_gd_tally(df: pd.DataFrame) -> bool:
    if not {"gd_fission_share", "gd_fission_share_std"}.issubset(df.columns):
        return False
    vals = pd.to_numeric(df["gd_fission_share"], errors="coerce")
    stds = pd.to_numeric(df["gd_fission_share_std"], errors="coerce")
    return bool(vals.notna().all() and stds.notna().all() and (stds > 0).all())


def _recommended_restarts(n_rows: int, n_features: int) -> int:
    if n_rows <= 8:
        return 3
    if n_rows <= 20:
        return 5
    if n_rows <= 60:
        return 7
    return 9


def _kernel_bound_diagnostics(gp: GaussianProcessRegressor, n_features: int) -> Dict[str, object]:
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
        return {
            "length_scale": [],
            "lower_bound": float("nan"),
            "upper_bound": float("nan"),
            "near_upper_features": [],
            "near_lower_features": [],
        }

    ls = np.asarray(ls_kernel.length_scale, dtype=float)
    lb, ub = ls_kernel.length_scale_bounds
    ls = ls.reshape(-1)
    near_upper = np.isclose(ls, float(ub), rtol=0.0, atol=max(1e-9, 0.02 * float(ub)))
    near_lower = np.isclose(ls, float(lb), rtol=0.0, atol=max(1e-9, 0.02 * float(lb)))
    if n_features == 1 and ls.size == 1:
        names = ["alpha_gd"]
    else:
        names = STRICT_4D_FEATURES[: ls.size]
    return {
        "length_scale": ls.tolist(),
        "lower_bound": float(lb),
        "upper_bound": float(ub),
        "near_upper_features": [names[i] for i, hit in enumerate(near_upper) if bool(hit)],
        "near_lower_features": [names[i] for i, hit in enumerate(near_lower) if bool(hit)],
    }


def fit_gp(train: pd.DataFrame, target_col: str = "keff", noise_col: str = "keff_std"):
    feature_cols = gp_feature_columns(train)
    x = train[feature_cols].to_numpy(dtype=float)
    y = train[target_col].to_numpy(dtype=float)
    ysd = float(y.std())
    if not np.isfinite(ysd) or ysd <= 0:
        ysd = float(np.max(np.abs(y - y.mean()))) + 1e-9
    ymu = float(y.mean())
    Y = (y - ymu) / ysd
    alpha_noise = (train[noise_col].to_numpy(dtype=float) / ysd) ** 2 + 1e-12

    n_restarts = _recommended_restarts(n_rows=len(train), n_features=len(feature_cols))

    scaler = StandardScaler()
    X = scaler.fit_transform(x)
    kernel = ConstantKernel(1.0, GP_AMPLITUDE_BOUNDS) * (
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-6, 1e3))
        + Matern(
            length_scale=np.ones(x.shape[1], dtype=float),
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS_4D,
            nu=2.5,
        )
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_noise,
        normalize_y=False,
        n_restarts_optimizer=n_restarts,
        random_state=24681357,
    )
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(X, Y)
    conv = [str(w.message) for w in ws if issubclass(w.category, ConvergenceWarning)]
    feature_medians = {col: float(np.median(train[col].to_numpy(dtype=float))) for col in feature_cols}
    diagnostics = _kernel_bound_diagnostics(gp, n_features=len(feature_cols))
    diagnostics["convergence_warnings"] = conv
    diagnostics["n_restarts_optimizer"] = int(n_restarts)
    return gp, {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "ymu": ymu,
        "ysd": ysd,
        "feature_medians": feature_medians,
        "diagnostics": diagnostics,
    }


def predict(gp, scales, grid, condition_values: Dict[str, float] | None = None):
    feature_cols = scales["feature_cols"]
    ymu = float(scales["ymu"])
    ysd = float(scales["ysd"])

    X_pred = np.empty((len(grid), len(feature_cols)), dtype=float)
    X_pred[:, 0] = np.asarray(grid, dtype=float)
    condition_lookup = {} if condition_values is None else {str(k): float(v) for k, v in condition_values.items()}
    for idx, col in enumerate(feature_cols[1:], start=1):
        if condition_values is not None and col in condition_lookup:
            X_pred[:, idx] = condition_lookup[col]
        else:
            X_pred[:, idx] = float(scales["feature_medians"][col])
    X_scaled = scales["scaler"].transform(X_pred)
    m, s = gp.predict(X_scaled, return_std=True)
    return m * ysd + ymu, s * ysd


def fit_weighted_poly(train: pd.DataFrame, degree: int, target_col: str = "keff", noise_col: str = "keff_std"):
    degree = int(degree)
    if degree < 1 or degree > 2:
        raise ValueError("fit_weighted_poly supports degree 1-2 (fixed ridge); use fit_weighted_poly_loo for regularized/cubic.")
    feature_cols = gp_feature_columns(train)
    X = train[feature_cols].to_numpy(dtype=float)
    y = train[target_col].to_numpy(dtype=float)
    sigma = train[noise_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(f"{noise_col} must be finite and > 0 for weighted polynomial fits")

    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(Xs)
    weights = 1.0 / np.maximum(sigma ** 2, 1e-16)

    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    ridge.fit(Phi, y, sample_weight=weights)
    pred = ridge.predict(Phi)
    resid = y - pred
    dof = max(1, len(y) - Phi.shape[1])
    resid_sd = float(np.sqrt(np.sum(weights * resid * resid) / np.sum(weights) * (len(y) / dof)))

    return {
        "degree": degree,
        "feature_cols": feature_cols,
        "x_scaler": x_scaler,
        "poly": poly,
        "ridge": ridge,
        "resid_sd": max(resid_sd, 1e-10),
    }


def predict_weighted_poly(model, grid, condition_values: Dict[str, float] | None = None):
    feature_cols = list(model["feature_cols"])
    n = len(grid)
    X_pred = np.empty((n, len(feature_cols)), dtype=float)
    X_pred[:, 0] = np.asarray(grid, dtype=float)

    cond = {} if condition_values is None else {str(k): float(v) for k, v in condition_values.items()}
    for idx, col in enumerate(feature_cols[1:], start=1):
        X_pred[:, idx] = float(cond.get(col, 0.0))

    Xs = model["x_scaler"].transform(X_pred)
    Phi = model["poly"].transform(Xs)
    mu = model["ridge"].predict(Phi)
    sigma = np.full(n, float(model["resid_sd"]), dtype=float)
    return mu, sigma


def fit_gp_nugget(train: pd.DataFrame, target_col: str = "keff", noise_col: str = "keff_std"):
    """Same kernel as fit_gp (4-D) plus a learned WhiteKernel (model-discrepancy nugget)."""
    feature_cols = gp_feature_columns(train)
    x = train[feature_cols].to_numpy(dtype=float)
    y = train[target_col].to_numpy(dtype=float)
    ymu, ysd = float(y.mean()), float(y.std())
    if not np.isfinite(ysd) or ysd <= 0:
        ysd = float(np.max(np.abs(y - ymu))) + 1e-9
    Y = (y - ymu) / ysd
    alpha_noise = (train[noise_col].to_numpy(dtype=float) / ysd) ** 2 + 1e-12
    scaler = StandardScaler()
    X = scaler.fit_transform(x)
    kernel = ConstantKernel(1.0, GP_AMPLITUDE_BOUNDS) * (
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-6, 1e3))
        + Matern(length_scale=np.ones(x.shape[1]), length_scale_bounds=GP_LENGTH_SCALE_BOUNDS_4D, nu=2.5)
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_noise,
        normalize_y=False,
        n_restarts_optimizer=_recommended_restarts(n_rows=len(train), n_features=len(feature_cols)),
        random_state=24681357,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, Y)
    return gp, {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "ymu": ymu,
        "ysd": ysd,
        "feature_medians": {col: float(np.median(train[col].to_numpy(dtype=float))) for col in feature_cols},
    }


class _QuadPlusCubes:
    """Full quadratic + pure cubic terms x_i^3 (19 terms in 4-D): a cubic comparator without 35 coefficients."""

    def __init__(self):
        self.q = PolynomialFeatures(degree=2, include_bias=True)

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        return np.hstack([self.q.fit_transform(X), X ** 3])


def fit_weighted_poly_loo(train: pd.DataFrame, degree, lambdas=POLY_LOO_LAMBDAS, target_col: str = "keff", noise_col: str = "keff_std"):
    """Weighted polynomial of any degree (int) or 'cubic_reduced' with LOO-selected ridge penalty."""
    feature_cols = gp_feature_columns(train)
    x = train[feature_cols].to_numpy(dtype=float)
    y = train[target_col].to_numpy(dtype=float)
    sigma = train[noise_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(f"{noise_col} must be finite and > 0 for weighted polynomial fits")
    w = 1.0 / np.maximum(sigma ** 2, 1e-16)
    wn = w / w.mean()
    sc = StandardScaler()
    poly = _QuadPlusCubes() if degree == "cubic_reduced" else PolynomialFeatures(degree=int(degree), include_bias=True)
    Phi = poly.fit_transform(sc.fit_transform(x))
    A, b = Phi * np.sqrt(wn)[:, None], y * np.sqrt(wn)
    pen = np.ones(Phi.shape[1])
    pen[0] = 0.0
    best = None
    for lam in lambdas:
        G = A.T @ A + lam * np.diag(pen)
        Ginv_At = np.linalg.solve(G, A.T)
        H = A @ Ginv_At
        coef = Ginv_At @ b
        r = (b - A @ coef) / np.maximum(1.0 - np.diag(H), 1e-6)
        score = float(np.mean(r ** 2))
        if best is None or score < best[0]:
            best = (score, lam, coef, r)
    _, lam, coef, r = best
    resid_sd = float(np.sqrt(np.sum(r ** 2) / np.sum(wn)))
    return {
        "feature_cols": feature_cols,
        "sc": sc,
        "poly": poly,
        "coef": coef,
        "lam": float(lam),
        "resid_sd": max(resid_sd, 1e-10),
        "feature_medians": {col: float(np.median(train[col].to_numpy(dtype=float))) for col in feature_cols},
    }


def predict_weighted_poly_loo(model, grid, condition_values: Dict[str, float] | None = None):
    feature_cols = list(model["feature_cols"])
    X_pred = np.empty((len(grid), len(feature_cols)), dtype=float)
    X_pred[:, 0] = np.asarray(grid, dtype=float)
    cond = {} if condition_values is None else {str(k): float(v) for k, v in condition_values.items()}
    for idx, col in enumerate(feature_cols[1:], start=1):
        X_pred[:, idx] = float(cond.get(col, model["feature_medians"][col]))
    mu = model["poly"].transform(model["sc"].transform(X_pred)) @ model["coef"]
    return mu, np.full(len(grid), float(model["resid_sd"]), dtype=float)


def _posterior_summary(grid, dens):
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))))
    cdf /= cdf[-1]

    def q(p):
        return float(np.interp(p, cdf, grid))

    out = {
        "map": float(grid[np.argmax(dens)]),
        "mean": float(np.trapezoid(grid * dens, grid)),
        "median": q(0.5),
    }
    for lvl in CI_LEVELS:
        lo = q((1.0 - lvl) / 2.0)
        hi = q(1.0 - (1.0 - lvl) / 2.0)
        tag = f"{int(round(100 * lvl)):02d}"
        out[f"ci{tag}_low"] = lo
        out[f"ci{tag}_high"] = hi
        out[f"width_ci{tag}"] = hi - lo
    return out


def fit_predictor(train, model_key: str, target_col: str = "keff", noise_col: str = "keff_std"):
    """Fit the surrogate ONCE for a given observable and return pred(grid, condition_values) -> (mu, sigma_model)."""
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unsupported model_key={model_key}.")
    spec = MODEL_SPECS[model_key]
    kind = spec["kind"]
    if kind == "gp":
        gp, sc = fit_gp(train, target_col=target_col, noise_col=noise_col)
        return lambda grid, cond=None: predict(gp, sc, grid, condition_values=cond)
    if kind == "gp_nugget":
        gp, sc = fit_gp_nugget(train, target_col=target_col, noise_col=noise_col)
        return lambda grid, cond=None: predict(gp, sc, grid, condition_values=cond)
    if kind == "poly":
        poly = fit_weighted_poly(train, degree=int(spec["degree"]), target_col=target_col, noise_col=noise_col)
        return lambda grid, cond=None: predict_weighted_poly(poly, grid, condition_values=cond)
    if kind == "poly_loo":
        poly = fit_weighted_poly_loo(train, spec["degree"], target_col=target_col, noise_col=noise_col)
        return lambda grid, cond=None: predict_weighted_poly_loo(poly, grid, condition_values=cond)
    raise ValueError(f"Unsupported model kind={kind}.")


def joint_posterior(obs_specs: List[Dict[str, object]], condition_values: Dict[str, float] | None = None):
    """Compute 1D Bayesian posterior over alpha_Gd combining one or more independent observables."""
    grid = np.linspace(ALPHA_MIN, ALPHA_MAX, N_GRID)
    logp = np.zeros_like(grid)
    for spec in obs_specs:
        mu, sigma_model = spec["predictor"](grid, condition_values)
        var = float(spec["obs_mc_sd"]) ** 2 + float(spec["meas_sd"]) ** 2 + sigma_model ** 2
        logp += -0.5 * ((float(spec["obs_val"]) - mu) ** 2 / var + np.log(var))
    dens = np.exp(logp - logp.max())
    dens /= np.trapezoid(dens, grid)
    return grid, dens, _posterior_summary(grid, dens)


def posterior(train, obs_keff, obs_mc_sd, model_key="gp", measurement_sd=MEASUREMENT_SD, condition_values: Dict[str, float] | None = None, predictor=None):
    if predictor is None:
        predictor = fit_predictor(train, model_key)
    obs_specs = [{
        "predictor": predictor,
        "obs_val": float(obs_keff),
        "obs_mc_sd": float(obs_mc_sd),
        "meas_sd": float(measurement_sd),
    }]
    return joint_posterior(obs_specs, condition_values=condition_values)


def grouped_cv_folds(df: pd.DataFrame, n_folds: int = 5, group_mode: str = "alpha") -> List[np.ndarray]:
    if len(df) == 0:
        return []
    if group_mode == "alpha" and "alpha_gd" in df.columns:
        groups = df["alpha_gd"].round(6)
    elif all(col in df.columns for col in STRICT_4D_FEATURES):
        groups = df[STRICT_4D_FEATURES].round(12).astype(str).agg(lambda row: tuple(row), axis=1)
    elif "alpha_gd" in df.columns:
        groups = df["alpha_gd"].round(12)
    else:
        groups = np.arange(len(df))

    n_folds = max(2, min(int(n_folds), len(df)))
    unique_groups = pd.Index(groups).drop_duplicates()
    if len(unique_groups) < 2:
        return [np.arange(len(df), dtype=int)]

    gkf = GroupKFold(n_splits=min(n_folds, len(unique_groups)))
    indices = np.arange(len(df), dtype=int)
    fold_idx = []
    for train_idx, test_idx in gkf.split(X=indices, y=np.zeros(len(df)), groups=groups.to_numpy()):
        fold_idx.append(test_idx)
    return fold_idx


def grouped_splits(df: pd.DataFrame, fold_indices: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    all_idx = np.arange(len(df), dtype=int)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for test_idx in fold_indices:
        test_idx = np.asarray(test_idx, dtype=int)
        mask = np.ones(len(df), dtype=bool)
        mask[test_idx] = False
        train_idx = all_idx[mask]
        splits.append((train_idx, test_idx))
    return splits


def _holdout_record(
    predictor_k,
    row,
    idx,
    model_key: str,
    measurement_sd: float,
    grid,
    n_train: int,
    predictor_tally=None,
    tally_rel_sd: float = TALLY_REL_MEASUREMENT_SD,
    extra=None,
):
    """One blind inversion of a held-out row (k-only or joint k + Gd fission share)."""
    condition_values = {col: float(row[col]) for col in STRICT_4D_FEATURES if col != "alpha_gd"}
    mu_grid, _ = predictor_k(grid, condition_values)
    dk_dalpha = float(np.polyfit(grid, mu_grid, deg=1)[0])
    k_span = float(np.max(mu_grid) - np.min(mu_grid))

    obs_specs = [{
        "predictor": predictor_k,
        "obs_val": float(row["keff"]),
        "obs_mc_sd": float(row["keff_std"]),
        "meas_sd": float(measurement_sd),
    }]
    if predictor_tally is not None:
        tally_val = float(row["gd_fission_share"])
        tally_meas_sd = float(tally_rel_sd) * abs(tally_val) if measurement_sd > 0.0 else 0.0
        obs_specs.append({
            "predictor": predictor_tally,
            "obs_val": tally_val,
            "obs_mc_sd": float(row["gd_fission_share_std"]),
            "meas_sd": tally_meas_sd,
        })

    _, _, stats = joint_posterior(obs_specs, condition_values=condition_values)
    alpha_true = float(row["alpha_gd"])
    k_mu, k_sd = predictor_k(np.array([alpha_true]), condition_values)
    fwd_res = float(k_mu[0] - row["keff"])
    out = {
        "model": model_key,
        "obs_mode": "k_plus_tally" if predictor_tally is not None else "k_only",
        "holdout_id": int(idx),
        "holdout_alpha": alpha_true,
        **stats,
        "dk_dalpha": dk_dalpha,
        "k_span_alpha_grid": k_span,
        "error_mean": float(stats["mean"] - alpha_true),
        "error_map": float(stats["map"] - alpha_true),
        "error_median": float(stats["median"] - alpha_true),
        "abs_error_mean": float(abs(stats["mean"] - alpha_true)),
        "abs_error_map": float(abs(stats["map"] - alpha_true)),
        "sq_error_mean": float((stats["mean"] - alpha_true) ** 2),
        "rel_error_mean": float(abs(stats["mean"] - alpha_true) / max(abs(alpha_true), 1e-12)),
        "rel_error_map": float(abs(stats["map"] - alpha_true) / max(abs(alpha_true), 1e-12)),
        "fwd_res_pcm": 1.0e5 * fwd_res,
        "fwd_z": float(fwd_res / np.sqrt(k_sd[0] ** 2 + float(row["keff_std"]) ** 2)),
        "n_train": int(n_train),
    }
    for lvl in CI_LEVELS:
        tag = f"{int(round(100 * lvl)):02d}"
        out[f"covered_{tag}"] = bool(out[f"ci{tag}_low"] <= alpha_true <= out[f"ci{tag}_high"])
    if extra:
        out.update(extra)
    return out


def evaluate_holdout_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_key: str,
    measurement_sd: float = MEASUREMENT_SD,
    use_tally: bool = False,
    tally_rel_sd: float = TALLY_REL_MEASUREMENT_SD,
    extra=None,
):
    """Fit once on `train`, then blind-invert every row of `test`."""
    grid = np.linspace(ALPHA_MIN, ALPHA_MAX, N_GRID)
    predictor_k = fit_predictor(train, model_key, target_col="keff", noise_col="keff_std")
    predictor_tally = None
    if use_tally:
        predictor_tally = fit_predictor(
            train, model_key, target_col="gd_fission_share", noise_col="gd_fission_share_std"
        )
    return [
        _holdout_record(
            predictor_k,
            row,
            idx,
            model_key,
            measurement_sd,
            grid,
            len(train),
            predictor_tally=predictor_tally,
            tally_rel_sd=tally_rel_sd,
            extra=extra,
        )
        for idx, row in test.iterrows()
    ]


def loo_table(
    df,
    model_key: str,
    measurement_sd: float = MEASUREMENT_SD,
    use_tally: bool = False,
    tally_rel_sd: float = TALLY_REL_MEASUREMENT_SD,
):
    rows = []
    for idx in df.index:
        rows += evaluate_holdout_set(
            df.drop(index=idx),
            df.loc[[idx]],
            model_key,
            measurement_sd=measurement_sd,
            use_tally=use_tally,
            tally_rel_sd=tally_rel_sd,
        )
    return pd.DataFrame(rows)


def evaluate_splits(
    df: pd.DataFrame,
    model_key: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    measurement_sd: float = MEASUREMENT_SD,
    use_tally: bool = False,
    tally_rel_sd: float = TALLY_REL_MEASUREMENT_SD,
):
    rows: List[Dict[str, float]] = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx].copy()
        test = df.iloc[np.asarray(test_idx, dtype=int)]
        rows += evaluate_holdout_set(
            train,
            test,
            model_key,
            measurement_sd=measurement_sd,
            use_tally=use_tally,
            tally_rel_sd=tally_rel_sd,
            extra={"fold_id": int(fold_id)},
        )
    return pd.DataFrame(rows)


def audit_grouped_splits(df: pd.DataFrame, splits: List[Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    groups = df["alpha_gd"].round(6).reset_index(drop=True)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        if overlap:
            raise AssertionError(f"Grouped-CV leakage in fold {fold_id}: alpha in both train and test: {sorted(overlap)}")
        rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_test_alpha_groups": int(groups.iloc[test_idx].nunique()),
                "test_alphas": ";".join(f"{a:.3f}" for a in sorted(set(groups.iloc[test_idx]))),
                "alpha_overlap": 0,
            }
        )
    return pd.DataFrame(rows)


def summarise_experiment(rows: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    return (
        rows.groupby(by)
        .agg(
            n_test=("holdout_id", "count"),
            mae=("abs_error_mean", "mean"),
            cov50=("covered_50", "mean"),
            cov90=("covered_90", "mean"),
            cov95=("covered_95", "mean"),
            w95=("width_ci95", "mean"),
            fwd_rmse_pcm=("fwd_res_pcm", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            z_rms=("fwd_z", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .reset_index()
    )


def run_learning_curve(df, model_keys, measurement_sd, repeats, sizes, n_test_groups, seed):
    rng = np.random.default_rng(seed)
    grp = df["alpha_gd"].round(6)
    counts = grp.value_counts()
    sparse = counts[counts <= 2].index.to_numpy()
    n_test_groups = min(int(n_test_groups), len(sparse))
    rows: List[Dict[str, float]] = []
    for rep_id in range(repeats):
        te_groups = rng.choice(sparse, size=n_test_groups, replace=False)
        test = df[grp.isin(te_groups)]
        pool = df[~grp.isin(te_groups)]
        for n_tr in [n for n in sizes if n <= len(pool)]:
            train = pool.sample(n=n_tr, random_state=int(rng.integers(1 << 31)))
            for key in model_keys:
                rows += evaluate_holdout_set(train, test, key, measurement_sd, extra={"repeat": rep_id})
        print(f"[phase2][learning] repeat {rep_id + 1}/{repeats} done (test rows={len(test)}, pool={len(pool)})", flush=True)
    return pd.DataFrame(rows)


def _select_maximin_subset(pool: pd.DataFrame, n_init: int, rng: np.random.Generator) -> List[int]:
    """Greedy maximin space-filling initial design in standardized 4D input space."""
    X = StandardScaler().fit_transform(pool[STRICT_4D_FEATURES].to_numpy(dtype=float))
    n = len(pool)
    n_init = min(n_init, n)
    selected = [int(rng.integers(n))]
    min_dist_sq = np.sum((X - X[selected[0]]) ** 2, axis=1)
    while len(selected) < n_init:
        min_dist_sq[selected] = -1.0
        nxt = int(np.argmax(min_dist_sq))
        selected.append(nxt)
        d_new = np.sum((X - X[nxt]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, d_new)
    return [int(pool.index[i]) for i in selected]


def _acquire_gp_uncertainty(train: pd.DataFrame, candidates: pd.DataFrame, batch_size: int) -> List[int]:
    """Acquire candidates with highest epistemic variance ratio (sigma_GP^2 / sigma_MC^2)."""
    gp, sc = fit_gp(train, target_col="keff", noise_col="keff_std")
    Xs = sc["scaler"].transform(candidates[STRICT_4D_FEATURES].to_numpy(dtype=float))
    _, s_norm = gp.predict(Xs, return_std=True)
    epistemic_var = (s_norm * float(sc["ysd"])) ** 2
    mc_var = np.maximum(candidates["keff_std"].to_numpy(dtype=float) ** 2, 1e-16)
    # Mutual information acquisition score: 0.5 * log(1 + epistemic_var / mc_var)
    scores = 0.5 * np.log1p(epistemic_var / mc_var)
    top_k = np.argsort(scores)[::-1][:batch_size]
    return [int(candidates.index[i]) for i in top_k]


def _acquire_d_optimal_poly(train: pd.DataFrame, candidates: pd.DataFrame, batch_size: int) -> List[int]:
    """Sequential D-optimal acquisition for regularized weighted quadratic (wquad_loo)."""
    poly_fit = fit_weighted_poly_loo(train, degree=2, target_col="keff", noise_col="keff_std")
    sc = poly_fit["sc"]
    poly = poly_fit["poly"]
    lam = float(poly_fit["lam"])

    Phi_tr = poly.transform(sc.transform(train[STRICT_4D_FEATURES].to_numpy(dtype=float)))
    w_tr = 1.0 / np.maximum(train["keff_std"].to_numpy(dtype=float) ** 2, 1e-16)
    wn_tr = w_tr / w_tr.mean()

    pen = np.ones(Phi_tr.shape[1])
    pen[0] = 0.0
    # Information matrix M = Phi^T W Phi + lam * I
    M = (Phi_tr * wn_tr[:, None]).T @ Phi_tr + max(lam, 1e-6) * np.diag(pen)

    Phi_cand = poly.transform(sc.transform(candidates[STRICT_4D_FEATURES].to_numpy(dtype=float)))
    w_cand = (1.0 / np.maximum(candidates["keff_std"].to_numpy(dtype=float) ** 2, 1e-16)) / w_tr.mean()

    chosen_local: List[int] = []
    for _ in range(min(batch_size, len(candidates))):
        # Leverage score w(x) * phi(x)^T M^{-1} phi(x) = log det increase
        Minv_PhiT = np.linalg.solve(M, Phi_cand.T)
        leverages = w_cand * np.sum(Phi_cand.T * Minv_PhiT, axis=0)
        if chosen_local:
            leverages[chosen_local] = -1.0
        best_i = int(np.argmax(leverages))
        chosen_local.append(best_i)
        phi_vec = Phi_cand[best_i : best_i + 1, :]
        M = M + w_cand[best_i] * (phi_vec.T @ phi_vec)

    return [int(candidates.index[i]) for i in chosen_local]


def run_active_learning(df, measurement_sd, repeats, sizes, n_test_groups, seed):
    """Offline pool-based active learning comparing Active GP, D-Optimal wquad_loo, and Random sampling."""
    rng = np.random.default_rng(seed)
    grp = df["alpha_gd"].round(6)
    counts = grp.value_counts()
    sparse = counts[counts <= 2].index.to_numpy()
    n_test_groups = min(int(n_test_groups), len(sparse))

    target_sizes = sorted({int(s) for s in sizes if int(s) >= 8})
    if not target_sizes:
        target_sizes = [12, 16, 20, 28, 36, 44, 52]

    rows: List[Dict[str, float]] = []
    strategies = [
        ("active_gp", "gp"),
        ("d_optimal_wquad_loo", "wquad_loo"),
        ("random_gp", "gp"),
        ("random_wquad_loo", "wquad_loo"),
        ("random_wquad", "wquad"),
    ]

    for rep_id in range(repeats):
        te_groups = rng.choice(sparse, size=n_test_groups, replace=False)
        test = df[grp.isin(te_groups)]
        pool = df[~grp.isin(te_groups)].copy()
        valid_sizes = [s for s in target_sizes if s <= len(pool)]
        if not valid_sizes:
            continue

        n_init = valid_sizes[0]
        init_idx = _select_maximin_subset(pool, n_init=n_init, rng=rng)

        # Maintain trajectories for active_gp, d_optimal, and random
        trajectories = {
            "active_gp": list(init_idx),
            "d_optimal": list(init_idx),
            "random": list(init_idx),
        }

        for step_i, target_n in enumerate(valid_sizes):
            if step_i > 0:
                n_add = target_n - len(trajectories["active_gp"])
                for traj_name in trajectories:
                    cur_idx = trajectories[traj_name]
                    rem = pool.drop(index=cur_idx)
                    if len(rem) == 0 or n_add <= 0:
                        continue
                    k_add = min(n_add, len(rem))
                    if traj_name == "active_gp":
                        new_idx = _acquire_gp_uncertainty(pool.loc[cur_idx], rem, k_add)
                    elif traj_name == "d_optimal":
                        new_idx = _acquire_d_optimal_poly(pool.loc[cur_idx], rem, k_add)
                    else:
                        new_idx = [int(i) for i in rng.choice(rem.index.to_numpy(), size=k_add, replace=False)]
                    trajectories[traj_name].extend(new_idx)

            for strat_label, eval_model in strategies:
                traj_key = (
                    "active_gp"
                    if strat_label == "active_gp"
                    else ("d_optimal" if strat_label == "d_optimal_wquad_loo" else "random")
                )
                train_sub = pool.loc[trajectories[traj_key]]
                rows += evaluate_holdout_set(
                    train_sub,
                    test,
                    eval_model,
                    measurement_sd=measurement_sd,
                    extra={"repeat": rep_id, "strategy": strat_label},
                )

        print(f"[phase2][active] repeat {rep_id + 1}/{repeats} done (pool={len(pool)}, test={len(test)})", flush=True)

    return pd.DataFrame(rows)


def run_edge_extrapolation(df, model_keys, measurement_sd, frac):
    rows: List[Dict[str, float]] = []
    info: List[Dict[str, object]] = []
    for feat in STRICT_4D_FEATURES:
        lo, hi = df[feat].quantile(frac), df[feat].quantile(1.0 - frac)
        for side, mask in (("low", df[feat] < lo), ("high", df[feat] > hi)):
            if int(mask.sum()) < 2:
                info.append({"feature": feat, "side": side, "n_test": int(mask.sum()), "skipped": True})
                continue
            test, train = df[mask], df[~mask]
            info.append(
                {
                    "feature": feat, "side": side, "n_test": int(len(test)), "n_train": int(len(train)),
                    "test_range": f"{test[feat].min():.4g}-{test[feat].max():.4g}", "skipped": False,
                }
            )
            for key in model_keys:
                rows += evaluate_holdout_set(train, test, key, measurement_sd, extra={"feature": feat, "side": side})
    return pd.DataFrame(info), pd.DataFrame(rows)


def loo_summary(loo_df):
    out = {
        "model": str(loo_df["model"].iloc[0]),
        "obs_mode": str(loo_df["obs_mode"].iloc[0]) if "obs_mode" in loo_df.columns else "k_only",
        "n_holdouts": int(len(loo_df)),
        "mae_mean": float(loo_df["abs_error_mean"].mean()),
        "rmse_mean": float(np.sqrt(loo_df["sq_error_mean"].mean())),
        "mre_mean": float(loo_df["rel_error_mean"].mean()),
        "mre_map": float(loo_df["rel_error_map"].mean()),
        "mean_interval_width_95": float(loo_df["width_ci95"].mean()),
    }
    for lvl in CI_LEVELS:
        tag = f"{int(round(100 * lvl)):02d}"
        out[f"coverage_{tag}"] = float(loo_df[f"covered_{tag}"].mean())
        out[f"mean_interval_width_{tag}"] = float(loo_df[f"width_ci{tag}"].mean())
    return out


def alpha_metrics(loo_df):
    s = loo_df.sort_values("holdout_alpha").reset_index(drop=True).copy()
    c_abs = s["abs_error_mean"].to_numpy()
    c_sq = s["sq_error_mean"].to_numpy()
    c_cov95 = s["covered_95"].astype(float).to_numpy()
    c_w95 = s["width_ci95"].to_numpy()
    n = np.arange(1, len(s) + 1, dtype=float)
    s["mae_cum"] = np.cumsum(c_abs) / n
    s["rmse_cum"] = np.sqrt(np.cumsum(c_sq) / n)
    s["coverage95_cum"] = np.cumsum(c_cov95) / n
    s["mean_width95_cum"] = np.cumsum(c_w95) / n
    return s


def binned_summary_from_long(loo_long: pd.DataFrame, n_bins: int = 5):
    bins = np.linspace(ALPHA_MIN, ALPHA_MAX, n_bins + 1)
    labels = [f"[{bins[i]:.2f}, {bins[i + 1]:.2f}{')' if i < n_bins - 1 else ']'}" for i in range(n_bins)]
    out = loo_long.copy()
    out["alpha_bin"] = pd.cut(
        out["holdout_alpha"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    out.loc[out["holdout_alpha"] >= bins[-1], "alpha_bin"] = labels[-1]

    rows: List[Dict[str, float]] = []
    grouped = out.groupby(["model", "alpha_bin"], observed=False)
    for (model, alpha_bin), g in grouped:
        if len(g) == 0:
            continue
        rows.append(
            {
                "model": str(model),
                "alpha_bin": str(alpha_bin),
                "n_holdouts": int(len(g)),
                "mae_mean": float(g["abs_error_mean"].mean()),
                "rmse_mean": float(np.sqrt(g["sq_error_mean"].mean())),
                "coverage_95": float(g["covered_95"].mean()),
                "mean_interval_width_95": float(g["width_ci95"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "alpha_bin"]).reset_index(drop=True)


def paired_tests_from_long(
    loo_long: pd.DataFrame,
    ref: str = "gp",
    others: Tuple[str, ...] | None = None,
    metrics: Tuple[str, ...] = ("abs_error_mean", "abs_error_map"),
):
    from scipy.stats import wilcoxon

    rows: List[Dict[str, float]] = []
    if others is None:
        others = tuple(m for m in sorted(loo_long["model"].unique()) if m != ref)
    rng = np.random.default_rng(42)
    for metric in metrics:
        wide = loo_long.pivot_table(index="holdout_id", columns="model", values=metric, aggfunc="first")
        for m in others:
            d = (wide[m] - wide[ref]).dropna().to_numpy(dtype=float)
            w = wilcoxon(d, alternative="greater", zero_method="wilcox")
            if d.size > 1:
                boot = np.array([np.mean(rng.choice(d, size=d.size, replace=True)) for _ in range(2000)], dtype=float)
                ci_low, ci_high = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            else:
                ci_low = ci_high = float(d.mean()) if d.size == 1 else float("nan")
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{ref} vs {m}",
                    "n": int(d.size),
                    "mean_diff_other_minus_gp": float(d.mean()),
                    "bootstrap_mean_diff_low95": ci_low,
                    "bootstrap_mean_diff_high95": ci_high,
                    "median_diff_other_minus_gp": float(np.median(d)),
                    "wilcoxon_stat": float(w.statistic),
                    "wilcoxon_p_one_sided": float(w.pvalue),
                    "gp_wins": int((d > 0).sum()),
                    "gp_ties": int((d == 0).sum()),
                    "gp_losses": int((d < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def null_baseline_metrics(holdout_df: pd.DataFrame) -> Dict[str, float]:
    prior_mean = 0.5 * (ALPHA_MIN + ALPHA_MAX)
    err = np.abs(holdout_df["holdout_alpha"].to_numpy(dtype=float) - prior_mean)
    return {
        "null_prior_mean_alpha": float(prior_mean),
        "null_mae": float(np.mean(err)),
        "null_rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def identifiability_report(loo_df: pd.DataFrame) -> Dict[str, float | str]:
    truth = loo_df["holdout_alpha"].to_numpy(dtype=float)
    post_mean = loo_df["mean"].to_numpy(dtype=float)
    alpha_span = float(ALPHA_MAX - ALPHA_MIN)
    slope = float(np.polyfit(truth, post_mean, deg=1)[0]) if len(loo_df) >= 2 else float("nan")
    corr = float(np.corrcoef(truth, post_mean)[0, 1]) if len(loo_df) >= 2 else float("nan")
    width95 = float(loo_df["width_ci95"].mean())
    width_rel = width95 / max(alpha_span, 1e-12)
    mae = float(loo_df["abs_error_mean"].mean())
    rel_err = float(loo_df["rel_error_mean"].mean())
    neg_dir_fraction = float((loo_df["dk_dalpha"] < 0.0).mean()) if "dk_dalpha" in loo_df.columns else float("nan")
    med_abs_dk = float(np.median(np.abs(loo_df["dk_dalpha"]))) if "dk_dalpha" in loo_df.columns else float("nan")
    med_k_span = float(np.median(loo_df["k_span_alpha_grid"])) if "k_span_alpha_grid" in loo_df.columns else float("nan")

    if width_rel <= 0.20:
        tightness = "tight"
    elif width_rel <= 0.40:
        tightness = "moderate"
    else:
        tightness = "broad"

    return {
        "direction_correlation_mean_vs_truth": corr,
        "magnitude_slope_mean_vs_truth": slope,
        "direction_negative_fraction": neg_dir_fraction,
        "magnitude_median_abs_dk_dalpha": med_abs_dk,
        "median_k_span_over_alpha_grid": med_k_span,
        "posterior_tightness_label": tightness,
        "mean_width95": width95,
        "mean_width95_fraction_of_prior": width_rel,
        "mae_mean": mae,
        "mre_mean": rel_err,
    }


def write_goal_report(loo_df: pd.DataFrame, out_path: Path, obs_mode: str = "k_only"):
    prior_span = float(ALPHA_MAX - ALPHA_MIN)
    tight_threshold = 0.20 * prior_span
    report = {
        "goal": f"4D OpenMC criticality sweep -> heteroscedastic GP -> deterministic Bayesian posterior ({obs_mode}) -> blind LOO inversion",
        "obs_mode": obs_mode,
        "n_holdouts": int(len(loo_df)),
        "posterior_mean_mae": float(loo_df["abs_error_mean"].mean()),
        "posterior_mean_relative_error": float(loo_df["rel_error_mean"].mean()),
        "rediscovery_within_10pct": float((loo_df["rel_error_mean"] <= 0.10).mean()),
        "rediscovery_within_20pct": float((loo_df["rel_error_mean"] <= 0.20).mean()),
        "coverage95": float(loo_df["covered_95"].mean()),
        "mean_width95": float(loo_df["width_ci95"].mean()),
        "tight_width95_threshold": float(tight_threshold),
        "tight_width95_fraction": float((loo_df["width_ci95"] <= tight_threshold).mean()),
        "direction_negative_fraction": float((loo_df["dk_dalpha"] < 0.0).mean()),
        "magnitude_median_abs_dk_dalpha": float(np.median(np.abs(loo_df["dk_dalpha"]))),
        "median_k_span_over_alpha_grid": float(np.median(loo_df["k_span_alpha_grid"])),
    }

    low_mask = loo_df["holdout_alpha"] <= 0.55
    if low_mask.any():
        low = loo_df.loc[low_mask]
        report["low_alpha_subset"] = {
            "n": int(len(low)),
            "bias_mean": float((low["mean"] - low["holdout_alpha"]).mean()),
            "mae_mean": float(low["abs_error_mean"].mean()),
            "coverage95": float(low["covered_95"].mean()),
            "note": "Descriptive only; small-bin estimate.",
        }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_model_comparison_overlay(
    summary_df: pd.DataFrame,
    out_dir: Path,
    pilot_mode: bool = False,
    scenario_tag: str = "obs50pcm",
    obs_mode: str = "k_only",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "pilot" if pilot_mode else "full"

    order = [k for k in MODEL_SPECS if k in set(summary_df["model"])]
    labels = [MODEL_SPECS[k]["label"] for k in order]
    s = summary_df.set_index("model").loc[order].reset_index()

    x = np.arange(len(order), dtype=float)
    mae = s["mae_mean"].to_numpy(dtype=float)
    rmse = s["rmse_mean"].to_numpy(dtype=float)
    width95 = s["mean_interval_width_95"].to_numpy(dtype=float)
    cov95 = 100.0 * s["coverage_95"].to_numpy(dtype=float)
    null_mae = 0.25

    fig, (ax_err, ax_cov) = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    w = 0.24
    ax_err.bar(x - w, mae, width=w, label="MAE", color="tab:blue")
    ax_err.bar(x, rmse, width=w, label="RMSE", color="tab:orange")
    ax_err.bar(x + w, width95, width=w, label="Mean 95% width", color="tab:green")
    ax_err.axhline(null_mae, color="black", linestyle="--", linewidth=1.2, label="Prior-mean MAE ref (0.25)")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels, rotation=12)
    ax_err.set_ylabel("alpha_Gd error / interval width")
    ax_err.set_title(f"Accuracy & uncertainty ({obs_mode})")
    ax_err.grid(True, axis="y", alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=8)

    ax_cov.bar(x, cov95, width=0.55, color="tab:purple")
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(labels, rotation=12)
    ax_cov.set_ylim(0.0, 100.0)
    ax_cov.set_ylabel("Coverage (%)")
    ax_cov.set_title("95% interval coverage")
    ax_cov.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"LOO model comparison ({mode_tag}, {scenario_tag}, {obs_mode})", y=1.02)

    fig.tight_layout()
    fig.savefig(out_dir / f"loo_model_comparison_overlay_{mode_tag}_{scenario_tag}_{obs_mode}.png", dpi=220)
    if scenario_tag == "obs50pcm" and obs_mode == "k_only":
        fig.savefig(out_dir / f"loo_model_comparison_overlay_{mode_tag}_{scenario_tag}.png", dpi=220)
        fig.savefig(out_dir / f"loo_model_comparison_overlay_{mode_tag}.png", dpi=220)
        fig.savefig(out_dir / "loo_model_comparison_overlay.png", dpi=220)
    plt.close(fig)


def predict_feature_curve(gp: GaussianProcessRegressor, scales: Dict[str, object], feature_name: str, df: pd.DataFrame, n_grid: int = 200):
    feature_cols = list(scales["feature_cols"])
    if feature_name not in feature_cols:
        raise ValueError(f"Feature {feature_name} not present in fitted model columns: {feature_cols}")

    other_features = [col for col in feature_cols if col != feature_name]
    fixed_values = {col: float(np.median(df[col].to_numpy(dtype=float))) for col in other_features}
    feature_grid = np.linspace(float(df[feature_name].min()), float(df[feature_name].max()), n_grid)

    base = np.array([fixed_values.get(col, float(np.median(df[col].to_numpy(dtype=float)))) for col in feature_cols], dtype=float)
    X_pred = np.tile(base, (len(feature_grid), 1))
    feature_idx = feature_cols.index(feature_name)
    X_pred[:, feature_idx] = feature_grid
    X_scaled = scales["scaler"].transform(X_pred)
    mu, _ = gp.predict(X_scaled, return_std=True)
    return feature_grid, mu * float(scales["ysd"]) + float(scales["ymu"])


def cross_validated_keff_predictions(df: pd.DataFrame, model_key: str = "gp") -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for idx, row in df.iterrows():
        train = df.drop(index=idx)
        if model_key == "gp":
            gp, sc = fit_gp(train)
            feature_cols = list(sc["feature_cols"])
            X_row = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
            X_scaled = sc["scaler"].transform(X_row)
            mu_pred, _ = gp.predict(X_scaled, return_std=True)
            pred_keff = float(mu_pred[0] * float(sc["ysd"]) + float(sc["ymu"]))
        else:
            raise ValueError(f"Unsupported cross-validation model: {model_key}")
        rows.append(
            {
                "holdout_id": int(idx),
                "alpha_gd": float(row["alpha_gd"]),
                "observed_keff": float(row["keff"]),
                "predicted_keff": pred_keff,
                "residual_keff": pred_keff - float(row["keff"]),
            }
        )
    return pd.DataFrame(rows)


def save_phase2_plots(df, loo_by_model: Dict[str, pd.DataFrame], out_dir: Path, pilot_mode: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "pilot" if pilot_mode else "full"

    gp, gp_scales = fit_gp(df)
    cv = cross_validated_keff_predictions(df, model_key="gp")

    parity_fig, parity_ax = plt.subplots(figsize=(7, 5.5))
    parity_ax.scatter(cv["observed_keff"], cv["predicted_keff"], s=28, alpha=0.8, color="tab:blue")
    lo = min(cv["observed_keff"].min(), cv["predicted_keff"].min())
    hi = max(cv["observed_keff"].max(), cv["predicted_keff"].max())
    parity_ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x")
    parity_ax.set_xlabel("Observed k-effective")
    parity_ax.set_ylabel("Cross-validated GP prediction")
    parity_ax.set_title(f"GP parity plot (leave-one-out; {mode_tag} run)")
    parity_ax.grid(True, alpha=0.3)
    parity_ax.legend()
    parity_fig.tight_layout()
    parity_fig.savefig(out_dir / f"gp_parity_cv_{mode_tag}.png", dpi=220)
    parity_fig.savefig(out_dir / "gp_parity_cv.png", dpi=220)
    plt.close(parity_fig)

    residual_fig, residual_ax = plt.subplots(figsize=(7, 5.0))
    residual_ax.scatter(cv["predicted_keff"], cv["residual_keff"], s=25, alpha=0.8, color="tab:orange")
    residual_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    residual_ax.set_xlabel("Predicted k-effective")
    residual_ax.set_ylabel("Residual = predicted - observed")
    residual_ax.set_title(f"GP residuals (leave-one-out; {mode_tag} run)")
    residual_ax.grid(True, alpha=0.3)
    residual_fig.tight_layout()
    residual_fig.savefig(out_dir / f"gp_residuals_{mode_tag}.png", dpi=220)
    residual_fig.savefig(out_dir / "gp_residuals.png", dpi=220)
    plt.close(residual_fig)

    for feature_name in gp_scales["feature_cols"]:
        grid, pred = predict_feature_curve(gp, gp_scales, feature_name, df)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(grid, pred, "k-", lw=2)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("k-effective")
        ax.set_title(f"GP partial dependence: {feature_name} ({mode_tag} run)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"gp_partial_dependence_{feature_name}_{mode_tag}.png", dpi=220)
        fig.savefig(out_dir / f"gp_partial_dependence_{feature_name}.png", dpi=220)
        plt.close(fig)

    for model_key, holdout in loo_by_model.items():
        spec = MODEL_SPECS[model_key]
        if {"holdout_alpha", "mean", "map", "median"}.issubset(holdout.columns):
            yerr = np.vstack(
                [
                    np.maximum(holdout["mean"] - holdout["ci95_low"], 0.0),
                    np.maximum(holdout["ci95_high"] - holdout["mean"], 0.0),
                ]
            )
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.errorbar(
                holdout["holdout_alpha"],
                holdout["mean"],
                yerr=yerr,
                fmt="o",
                capsize=3,
                alpha=0.85,
                label="posterior mean ± 95% CI",
            )
            lo = min(float(holdout["holdout_alpha"].min()), float(holdout["mean"].min()))
            hi = max(float(holdout["holdout_alpha"].max()), float(holdout["mean"].max()))
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="identity")
            ax.set_xlabel("holdout true alpha_Gd")
            ax.set_ylabel("posterior mean alpha_Gd")
            ax.set_title(f"{spec['label']} alpha parity ({mode_tag} run)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"posterior_holdout_alpha_{model_key}_{mode_tag}.png", dpi=200)
            if model_key == "gp":
                fig.savefig(out_dir / f"posterior_holdout_alpha_{mode_tag}.png", dpi=200)
                fig.savefig(out_dir / "posterior_holdout_alpha.png", dpi=200)
            plt.close(fig)

    print(f"Saved Phase II plots to {out_dir} for {mode_tag} mode")


def main():
    args = parse_args()
    pilot_mode = bool(args.pilot)
    mode_tag = "pilot" if pilot_mode else "full"
    data_path = Path(args.dataset_csv).expanduser().resolve()
    secret_path = Path(args.secret_alpha_lookup).expanduser().resolve() if args.secret_alpha_lookup else None
    result_dir = Path(args.output_dir).expanduser().resolve()
    plot_dir = Path(args.plots_dir).expanduser().resolve()

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    unknown_exp = [e for e in experiments if e not in VALID_EXPERIMENTS]
    if unknown_exp:
        raise ValueError(f"Unknown experiment(s) {unknown_exp}; choose from {sorted(VALID_EXPERIMENTS)}")

    print(f"[phase2] mode={mode_tag}")
    print(f"[phase2] dataset={data_path}")
    if secret_path:
        print(f"[phase2] secret_alpha_lookup={secret_path}")
    print(f"[phase2] output_dir={result_dir}")
    print(f"[phase2] plots_dir={plot_dir}")

    df = load_analysis_frame(data_path, secret_path)
    if "success" in df.columns:
        df = df[df["success"].astype(bool)].copy()

    tally_available = has_gd_tally(df)
    obs_modes = ["k_only", "k_plus_tally"] if tally_available else ["k_only"]
    print(f"[phase2] Gd fission share tally detected: {tally_available} -> running obs_modes={obs_modes}")

    fold_plan = grouped_cv_folds(df, n_folds=args.cv_folds, group_mode="alpha")
    print(
        f"[phase2] alpha-grouped CV plan: {len(fold_plan)} folds for {df['alpha_gd'].round(6).nunique()} unique alpha groups"
    )
    validate_dataset_range(df, pilot_mode=pilot_mode)
    validate_mode_case_count(df, pilot_mode=pilot_mode)

    gp_full, gp_scales = fit_gp(df)
    _ = gp_full
    gp_diag = gp_scales.get("diagnostics", {})

    active_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.extended:
        active_models += [m for m in EXTENDED_MODELS if m not in active_models]
    unknown = [m for m in active_models if m not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {list(MODEL_SPECS)}")
    if "gp" not in active_models:
        raise ValueError("'gp' must be among the models (it is the reference for the paired tests).")
    print(f"[phase2] models enabled: {', '.join(MODEL_SPECS[m]['label'] for m in active_models)}")

    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    default_summary_path = None
    default_loo_by_model: Dict[str, pd.DataFrame] = {}

    df_cv = df.reset_index(drop=True)
    cv_splits = grouped_splits(df_cv, grouped_cv_folds(df_cv, n_folds=args.cv_folds, group_mode="alpha"))
    audit_df = audit_grouped_splits(df_cv, cv_splits)
    audit_df.to_csv(result_dir / f"grouped_alpha_cv_fold_audit_{mode_tag}.csv", index=False)

    all_obs_summaries: List[pd.DataFrame] = []

    for obs_mode in obs_modes:
        use_tally = (obs_mode == "k_plus_tally")
        for scenario_tag, measurement_sd in OBS_SCENARIOS.items():
            summary_rows: List[Dict[str, float]] = []
            loo_by_model: Dict[str, pd.DataFrame] = {}

            for model_key in active_models:
                loo_df = loo_table(
                    df,
                    model_key,
                    measurement_sd=measurement_sd,
                    use_tally=use_tally,
                    tally_rel_sd=args.tally_rel_sd,
                )
                loo_by_model[model_key] = loo_df
                loo_path = result_dir / f"loo_inverse_recovery_{model_key}_{mode_tag}_{scenario_tag}_{obs_mode}.csv"
                loo_df.to_csv(loo_path, index=False)

                alpha_path = result_dir / f"loo_alpha_metrics_{model_key}_{mode_tag}_{scenario_tag}_{obs_mode}.csv"
                alpha_metrics(loo_df).to_csv(alpha_path, index=False)

                if model_key == "gp" and scenario_tag == "obs50pcm":
                    write_goal_report(loo_df, result_dir / f"goal_report_{mode_tag}_{obs_mode}.json", obs_mode=obs_mode)
                    if obs_mode == "k_only":
                        write_goal_report(loo_df, result_dir / f"goal_report_{mode_tag}.json", obs_mode=obs_mode)

                summary_rows.append(loo_summary(loo_df))

                # Unique-alpha subset
                unique_idx = df.assign(_a=df["alpha_gd"].round(6)).drop_duplicates(subset=["_a"]).index
                unique_df = df.loc[unique_idx].copy().drop(columns=["_a"], errors="ignore")
                unique_loo = loo_table(
                    unique_df.reset_index(drop=True),
                    model_key=model_key,
                    measurement_sd=measurement_sd,
                    use_tally=use_tally,
                    tally_rel_sd=args.tally_rel_sd,
                )
                unique_loo.to_csv(
                    result_dir / f"loo_inverse_recovery_{model_key}_{mode_tag}_{scenario_tag}_{obs_mode}_unique_alpha.csv",
                    index=False,
                )

                # Alpha-grouped CV
                grouped_eval = evaluate_splits(
                    df_cv,
                    model_key=model_key,
                    splits=cv_splits,
                    measurement_sd=measurement_sd,
                    use_tally=use_tally,
                    tally_rel_sd=args.tally_rel_sd,
                )
                grouped_eval.to_csv(
                    result_dir / f"grouped_alpha_cv_{model_key}_{mode_tag}_{scenario_tag}_{obs_mode}.csv", index=False
                )
                pd.DataFrame([loo_summary(grouped_eval)]).to_csv(
                    result_dir / f"grouped_alpha_cv_summary_{model_key}_{mode_tag}_{scenario_tag}_{obs_mode}.csv",
                    index=False,
                )

            summary_df = pd.DataFrame(summary_rows).sort_values("model")
            summary_df["scenario"] = scenario_tag
            summary_path = result_dir / f"loo_model_summary_{mode_tag}_{scenario_tag}_{obs_mode}.csv"
            summary_df.to_csv(summary_path, index=False)
            all_obs_summaries.append(summary_df)

            loo_long = pd.concat([loo_by_model[m] for m in active_models], ignore_index=True)
            loo_long.to_csv(result_dir / f"loo_inverse_recovery_models_{mode_tag}_{scenario_tag}_{obs_mode}.csv", index=False)

            paired_df = paired_tests_from_long(loo_long)
            paired_df.to_csv(result_dir / f"paired_tests_{mode_tag}_{scenario_tag}_{obs_mode}.csv", index=False)

            binned_summary_from_long(loo_long, n_bins=5).to_csv(
                result_dir / f"loo_model_summary_binned_alpha_{mode_tag}_{scenario_tag}_{obs_mode}.csv", index=False
            )

            save_model_comparison_overlay(
                summary_df, out_dir=plot_dir, pilot_mode=pilot_mode, scenario_tag=scenario_tag, obs_mode=obs_mode
            )

            if obs_mode == "k_only" and scenario_tag == "obs50pcm":
                default_summary_path = summary_path
                default_loo_by_model = loo_by_model
                summary_df.to_csv(result_dir / f"loo_model_summary_{mode_tag}.csv", index=False)
                loo_long.to_csv(result_dir / f"loo_inverse_recovery_models_{mode_tag}.csv", index=False)
                paired_df.to_csv(result_dir / f"paired_tests_{mode_tag}.csv", index=False)
                null_df = pd.DataFrame([null_baseline_metrics(loo_long[loo_long["model"] == "gp"])])
                null_df.to_csv(result_dir / f"null_baseline_{mode_tag}.csv", index=False)
                binned_summary_from_long(loo_long, n_bins=5).to_csv(
                    result_dir / f"loo_model_summary_binned_alpha_{mode_tag}.csv", index=False
                )

    combined_comparison = pd.concat(all_obs_summaries, ignore_index=True)
    combined_comparison.to_csv(result_dir / f"loo_konly_vs_tally_comparison_{mode_tag}.csv", index=False)

    phase2_summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode_tag,
        "dataset_csv": str(data_path),
        "n_rows_used": int(len(df)),
        "tally_available": tally_available,
        "obs_modes_evaluated": obs_modes,
        "alpha_range": [float(df["alpha_gd"].min()), float(df["alpha_gd"].max())],
        "models": active_models,
        "n_grid": N_GRID,
        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,
        "measurement_scenarios": OBS_SCENARIOS,
        "default_summary_csv": str(default_summary_path) if default_summary_path else "",
        "gp_diagnostics": gp_diag,
        "identifiability_k_only": identifiability_report(default_loo_by_model["gp"]) if "gp" in default_loo_by_model else {},
    }
    summary_json = result_dir / f"phase2_summary_{mode_tag}.json"
    summary_json.write_text(json.dumps(phase2_summary, indent=2), encoding="utf-8")

    save_phase2_plots(df, default_loo_by_model, out_dir=plot_dir, pilot_mode=pilot_mode)

    if experiments:
        ext_dir = result_dir / "extensions"
        ext_dir.mkdir(parents=True, exist_ok=True)
        df_exp = df.reset_index(drop=True)
        sizes = [int(v) for v in args.exp_sizes.split(",") if v.strip()]

        if "learning" in experiments:
            lc = run_learning_curve(df_exp, active_models, args.exp_measurement_sd, args.exp_repeats, sizes, args.exp_test_groups, args.exp_seed)
            lc.to_csv(ext_dir / f"learning_curve_rows_{mode_tag}.csv", index=False)
            summarise_experiment(lc, ["model", "n_train"]).to_csv(ext_dir / f"learning_curve_summary_{mode_tag}.csv", index=False)

        if "edges" in experiments:
            info, er = run_edge_extrapolation(df_exp, active_models, args.exp_measurement_sd, args.edge_frac)
            info.to_csv(ext_dir / f"edge_design_{mode_tag}.csv", index=False)
            er.to_csv(ext_dir / f"edge_rows_{mode_tag}.csv", index=False)
            summarise_experiment(er, ["feature", "side", "model"]).to_csv(ext_dir / f"edge_summary_{mode_tag}.csv", index=False)

        if "active" in experiments:
            al = run_active_learning(df_exp, args.exp_measurement_sd, args.exp_repeats, sizes, args.exp_test_groups, args.exp_seed)
            al.to_csv(ext_dir / f"active_learning_rows_{mode_tag}.csv", index=False)
            summarise_experiment(al, ["strategy", "n_train"]).to_csv(ext_dir / f"active_learning_summary_{mode_tag}.csv", index=False)

        print(f"[phase2] extension experiments {experiments} written to {ext_dir}")

    print(f"Completed Phase II inversion using {data_path}")
    print(f"Saved combined k-only vs tally comparison to {result_dir / f'loo_konly_vs_tally_comparison_{mode_tag}.csv'}")


if __name__ == "__main__":
    main()