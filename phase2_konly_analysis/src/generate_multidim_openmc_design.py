#!/usr/bin/env python3
"""Generate a physically feasible 4-feature OpenMC design matrix.

This script creates a DOE for the 4-feature GP study with the constraints:
- alpha_gd in [0.25, 1.25]
- fuel_temp in [294, 1000] K
- moderator_temp in [294, 600] K
- boron_ppm in [0, 2200] ppm
- fuel_temp >= moderator_temp
- additional replication in the highest-noise / strongest-absorber region

The design is written to a CSV and to a companion JSON summary. The output is
intended to be used as the OpenMC planning dataset before full transport runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "phase2_konly_analysis" / "multidim_design"
FEATURE_RANGES = {
    "alpha_gd": (0.25, 1.25),
    "fuel_temp": (294.0, 1000.0),
    "moderator_temp": (294.0, 600.0),
    "boron_ppm": (0.0, 2000.0),
}

TARGETED_ALPHA_ANCHORS = (0.40, 1.10)
TARGETED_FUEL_ANCHORS = (700.0, 900.0)
TARGETED_MODERATOR_SWEEP = (300.0, 400.0, 500.0, 580.0)
TARGETED_BORON_SWEEP = (100.0, 500.0, 1000.0, 1600.0, 2100.0)
TARGETED_BORON_FIXED = 1000.0
TARGETED_MODERATOR_FIXED = 500.0
EDGE_PROXIMITY_FRACTION = 0.05
EDGE_CONCENTRATION_WARN_RATIO = 0.25


def _edge_band(value: float, lo: float, hi: float, frac: float) -> str:
    width = max(hi - lo, 1e-12)
    band = frac * width
    if value <= lo + band:
        return "lower"
    if value >= hi - band:
        return "upper"
    return "interior"


def summarize_edge_concentration(df: pd.DataFrame, frac: float = EDGE_PROXIMITY_FRACTION) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    n = max(len(df), 1)
    for feature, (lo, hi) in FEATURE_RANGES.items():
        lower = int((df[feature].apply(lambda v: _edge_band(float(v), lo, hi, frac) == "lower")).sum())
        upper = int((df[feature].apply(lambda v: _edge_band(float(v), lo, hi, frac) == "upper")).sum())
        summary[feature] = {
            "lower_count": lower,
            "upper_count": upper,
            "lower_ratio": float(lower / n),
            "upper_ratio": float(upper / n),
        }
    return summary


def feasible_sample(alpha: float, fuel: float, mod: float, boron: float) -> Tuple[float, float, float, float]:
    if fuel < mod:
        # Enforce the design constraint by swapping values inside the feasible region.
        fuel, mod = mod, fuel
    return float(alpha), float(fuel), float(mod), float(boron)


def sobol_feasible_rows(n_unique: int, random_seed: int) -> List[Dict[str, float]]:
    m = int(np.ceil(np.log2(max(1, n_unique))))
    sampler = qmc.Sobol(d=4, scramble=True, seed=random_seed)
    pts = sampler.random_base2(m)
    if len(pts) < n_unique:
        pts = np.vstack([pts, sampler.random(n_unique - len(pts))])
    pts = pts[:n_unique]

    l_bounds = np.array([FEATURE_RANGES[k][0] for k in ["alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm"]], dtype=float)
    u_bounds = np.array([FEATURE_RANGES[k][1] for k in ["alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm"]], dtype=float)
    scaled = qmc.scale(pts, l_bounds=l_bounds, u_bounds=u_bounds)

    rows: List[Dict[str, float]] = []
    for i, (alpha, fuel, mod, boron) in enumerate(scaled, start=1):
        alpha, fuel, mod, boron = feasible_sample(alpha, fuel, mod, boron)
        rows.append(
            {
                "alpha_gd": alpha,
                "fuel_temp": fuel,
                "moderator_temp": mod,
                "boron_ppm": boron,
                "replicate": False,
                "region_tag": "general_design",
                "case_id": f"design_{i:03d}",
            }
        )
    return rows


def _make_targeted_slice_points() -> pd.DataFrame:
    """Create deterministic DOE augmentations for weakly identified dimensions.

    These points add independent variation along moderator_temp and boron_ppm
    while holding alpha_gd and fuel_temp at controlled anchors.
    """
    rows: List[Dict[str, float]] = []

    # Moderator slices: vary moderator_temp with fixed boron.
    for alpha in TARGETED_ALPHA_ANCHORS:
        for fuel in TARGETED_FUEL_ANCHORS:
            for moderator in TARGETED_MODERATOR_SWEEP:
                if fuel < moderator:
                    continue
                rows.append(
                    {
                        "alpha_gd": float(alpha),
                        "fuel_temp": float(fuel),
                        "moderator_temp": float(moderator),
                        "boron_ppm": float(TARGETED_BORON_FIXED),
                        "replicate": False,
                        "region_tag": "targeted_moderator_slice",
                    }
                )

    # Boron slices: vary boron with fixed moderator.
    for alpha in TARGETED_ALPHA_ANCHORS:
        for fuel in TARGETED_FUEL_ANCHORS:
            if fuel < TARGETED_MODERATOR_FIXED:
                continue
            for boron in TARGETED_BORON_SWEEP:
                rows.append(
                    {
                        "alpha_gd": float(alpha),
                        "fuel_temp": float(fuel),
                        "moderator_temp": float(TARGETED_MODERATOR_FIXED),
                        "boron_ppm": float(boron),
                        "replicate": False,
                        "region_tag": "targeted_boron_slice",
                    }
                )

    return pd.DataFrame(rows)


def generate_design(
    n_unique: int = 60,
    n_replicates: int = 12,
    random_seed: int = 42,
    include_targeted_slices: bool = True,
) -> pd.DataFrame:
    rows = sobol_feasible_rows(n_unique=n_unique, random_seed=random_seed)

    df = pd.DataFrame(rows)
    if include_targeted_slices:
        targeted = _make_targeted_slice_points()
        df = pd.concat([df, targeted], ignore_index=True)
        # Remove exact duplicates that can appear where sweeps intersect.
        df = df.drop_duplicates(
            subset=["alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm", "replicate", "region_tag"]
        ).reset_index(drop=True)

    noisy_mask = (
        (df["alpha_gd"] >= 0.75)
        & (df["boron_ppm"] >= 1500.0)
        & (df["fuel_temp"] >= 700.0)
        & (df["moderator_temp"] >= 294.0)
    )
    noisy = df.loc[noisy_mask].copy()
    if len(noisy) == 0:
        noisy = df.sample(n=min(n_replicates, len(df)), random_state=random_seed).copy()

    reps = noisy.head(min(n_replicates, len(noisy))).copy()
    reps["replicate"] = True
    reps["region_tag"] = "replicate_noisy_region"
    reps["case_id"] = [f"replicate_{i:03d}" for i in range(1, len(reps) + 1)]
    df = pd.concat([df, reps], ignore_index=True)
    df["case_id"] = [f"case_{i:03d}" for i in range(1, len(df) + 1)]
    df = df[["case_id", "alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm", "replicate", "region_tag"]]
    return df


def save_design_plots(df: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(df["replicate"], "tab:red", "tab:blue")
    ax.scatter(
        df["alpha_gd"],
        df["fuel_temp"],
        df["boron_ppm"],
        c=colors,
        s=30,
        alpha=0.9,
    )
    ax.set_xlabel("alpha_gd")
    ax.set_ylabel("fuel_temp (K)")
    ax.set_zlabel("boron_ppm")
    ax.set_title("4-feature feasible OpenMC design")
    fig.tight_layout()
    fig.savefig(out_dir / "openmc_4d_design.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["moderator_temp"], df["fuel_temp"], c=np.where(df["replicate"], "tab:red", "tab:blue"), s=35)
    ax.axline((294, 294), slope=1, color="black", linestyle="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("moderator_temp (K)")
    ax.set_ylabel("fuel_temp (K)")
    ax.set_title("Feasible region: fuel_temp >= moderator_temp")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "feasible_temperature_region.png", dpi=220)
    plt.close(fig)


def save_summary(df: pd.DataFrame, out_dir: Path) -> None:
    region_counts = {str(k): int(v) for k, v in df["region_tag"].value_counts().items()}
    summary = {
        "n_total_runs": int(len(df)),
        "n_unique": int((~df["replicate"]).sum()),
        "n_replicates": int(df["replicate"].sum()),
        "region_counts": region_counts,
        "ranges": {
            "alpha_gd": [float(df["alpha_gd"].min()), float(df["alpha_gd"].max())],
            "fuel_temp": [float(df["fuel_temp"].min()), float(df["fuel_temp"].max())],
            "moderator_temp": [float(df["moderator_temp"].min()), float(df["moderator_temp"].max())],
            "boron_ppm": [float(df["boron_ppm"].min()), float(df["boron_ppm"].max())],
        },
        "feasibility_constraint": "fuel_temp >= moderator_temp",
        "replicate_region": "high-alpha + high-boron + high-fuel-region, with additional repeated runs for noise estimation",
        "library_coverage_note": "Confirm library support for 294, 600, 900, and 1200 K before using 1000 K points; 1000 K may require interpolation between 900 and 1200 K.",
        "edge_proximity_fraction": EDGE_PROXIMITY_FRACTION,
        "edge_concentration": summarize_edge_concentration(df, frac=EDGE_PROXIMITY_FRACTION),
        "alpha_gd_definition": "effective absorber-strength multiplier, not a literal Gd2O3 mass fraction unless the chemistry model is explicitly built that way.",
        "fuel_temperature_definition": "single volume-averaged Doppler-effective fuel temperature in K; not a centerline or melting-limit temperature.",
        "moderator_temperature_definition": "single bulk coolant/moderator temperature in K at the design pressure.",
    }
    (out_dir / "openmc_design_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a feasible 4D OpenMC DOE matrix.")
    parser.add_argument("--pilot", action="store_true", help="Generate a small pilot matrix (5 unique points, 0 replicates).")
    parser.add_argument("--full", action="store_true", help="Generate a full matrix (default behavior).")
    parser.add_argument("--n-unique", type=int, default=None, help="Override number of base random unique points.")
    parser.add_argument("--n-replicates", type=int, default=None, help="Override number of replicate points.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for base feasible sampling.")
    parser.add_argument(
        "--no-targeted-slices",
        action="store_true",
        help="Disable targeted moderator/boron slice augmentation.",
    )
    args = parser.parse_args()
    if args.pilot and args.full:
        parser.error("Choose either --pilot or --full, not both.")
    return args


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot_mode = bool(args.pilot)
    n_unique = args.n_unique if args.n_unique is not None else (5 if pilot_mode else 60)
    n_replicates = args.n_replicates if args.n_replicates is not None else (0 if pilot_mode else 12)
    include_targeted = not args.no_targeted_slices and not pilot_mode

    design = generate_design(
        n_unique=n_unique,
        n_replicates=n_replicates,
        random_seed=args.random_seed,
        include_targeted_slices=include_targeted,
    )
    design_path = OUT_DIR / "openmc_4d_design_matrix.csv"
    design.to_csv(design_path, index=False)
    save_design_plots(design, OUT_DIR)
    save_summary(design, OUT_DIR)

    print(f"Wrote OpenMC design matrix to {design_path}")
    print(f"Rows: {len(design)}; unique design points: {int((~design['replicate']).sum())}; replicates: {int(design['replicate'].sum())}")
    print(f"Targeted slices enabled: {include_targeted}")
    region_counts = {str(k): int(v) for k, v in design["region_tag"].value_counts().items()}
    print(f"Region counts: {region_counts}")
    edge = summarize_edge_concentration(design)
    for feature, stats in edge.items():
        print(
            f"Edge concentration {feature}: lower={stats['lower_count']}/{len(design)} ({stats['lower_ratio']:.1%}), "
            f"upper={stats['upper_count']}/{len(design)} ({stats['upper_ratio']:.1%})"
        )
        if stats["lower_ratio"] >= EDGE_CONCENTRATION_WARN_RATIO or stats["upper_ratio"] >= EDGE_CONCENTRATION_WARN_RATIO:
            print(f"[warn] High edge concentration detected for {feature}; consider more unique points or fewer targeted edge anchors")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
