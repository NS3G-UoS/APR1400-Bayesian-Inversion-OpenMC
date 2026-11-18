import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openmc
import pandas as pd
from scipy.stats import qmc


PITCH_CM = 1.285
ASSEMBLY_SIZE = 16
FUEL_R = 0.4095
CLAD_INNER_R = 0.4175
CLAD_OUTER_R = 0.4750
LARGE_TUBE_INNER_R = 1.14495
LARGE_TUBE_OUTER_R = 1.24673
Z_MIN = -0.5
Z_MAX = 0.5
ALPHA_GRID = [round(x, 3) for x in np.linspace(0.25, 1.25, 41)]
PILOT_ALPHA_GRID = [round(x, 3) for x in np.linspace(0.25, 1.25, 5)]
BASE_SEED = 12345
FUEL_STACK_DENSITY = 10.301
UO2_PELLET_DENSITY = 10.44
UO2_TD = 10.96
GD2O3_TD = 7.41
GD_PELLET_FRACTION_TD = 0.9525
BORON_PPM = 1000.0
MODERATOR_TEMP_K = 600.0
FUEL_TEMP_K = 900.0
CLAD_TEMP_K = 600.0
MODERATOR_PRESSURE_MPA = 15.5
ALPHA_NOMINAL = 1.0
ALPHA_TRUE = 1.073

MULTIDIM_FEATURE_RANGES = {
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

C2_MAP = [
    ["L", "L", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "L", "L"],
    ["L", "H", "H", "G", "H", "H", "H", "H", "H", "H", "H", "H", "G", "H", "H", "L"],
    ["H", "H", "H", "L", "L", "H", "H", "H", "H", "H", "H", "L", "L", "H", "H", "H"],
    ["H", "G", "L", "C", "C", "L", "H", "H", "H", "H", "L", "C", "C", "L", "G", "H"],
    ["H", "H", "L", "C", "C", "L", "G", "H", "H", "G", "L", "C", "C", "L", "H", "H"],
    ["H", "H", "H", "L", "L", "H", "H", "H", "H", "H", "H", "L", "L", "H", "H", "H"],
    ["H", "H", "H", "H", "G", "H", "H", "L", "L", "H", "H", "G", "H", "H", "H", "H"],
    ["H", "H", "H", "H", "H", "H", "L", "I", "I", "L", "H", "H", "H", "H", "H", "H"],
    ["H", "H", "H", "H", "H", "H", "L", "I", "I", "L", "H", "H", "H", "H", "H", "H"],
    ["H", "H", "H", "H", "G", "H", "H", "L", "L", "H", "H", "G", "H", "H", "H", "H"],
    ["H", "H", "H", "L", "L", "H", "H", "H", "H", "H", "H", "L", "L", "H", "H", "H"],
    ["H", "H", "L", "C", "C", "L", "G", "H", "H", "G", "L", "C", "C", "L", "H", "H"],
    ["H", "G", "L", "C", "C", "L", "H", "H", "H", "H", "L", "C", "C", "L", "G", "H"],
    ["H", "H", "H", "L", "L", "H", "H", "H", "H", "H", "H", "L", "L", "H", "H", "H"],
    ["L", "H", "H", "G", "H", "H", "H", "H", "H", "H", "H", "H", "G", "H", "H", "L"],
    ["L", "L", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "L", "L"],
]

GD_POS = {
    (1, 3), (1, 12),
    (3, 1), (3, 14),
    (4, 6), (4, 9),
    (6, 4), (6, 11),
    (9, 4), (9, 11),
    (11, 6), (11, 9),
    (12, 1), (12, 14),
    (14, 3), (14, 12),
}


@dataclass
class CaseResult:
    alpha_gd: float
    case_dir: str
    returncode: int
    success: bool
    statepoint: Optional[str]
    keff: Optional[float]
    keff_std: Optional[float]
    flux_mean: Optional[float]
    gd_fission_share: Optional[float]
    gd_fission_share_std: Optional[float]
    warnings: List[str]
    error: Optional[str]


def partition_case_jobs(n_cases: int, workers: int) -> List[List[int]]:
    total = max(0, int(n_cases))
    worker_count = max(1, int(workers))
    if total == 0:
        return []
    worker_count = min(worker_count, total)
    jobs: List[List[int]] = [[] for _ in range(worker_count)]
    for case_number in range(1, total + 1):
        jobs[(case_number - 1) % worker_count].append(case_number)
    return [batch for batch in jobs if batch]


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    project_root = current
    seen: set[Path] = set()

    while current not in seen:
        seen.add(current)
        markers = [
            current / "raw_data",
            current / "processed_data",
            current / "private_truth",
            current / "reports",
            current / "phase2_konly_analysis",
            current / "openmc_apr1400_phase1_runner_final.py",
        ]
        if any(marker.exists() for marker in markers):
            project_root = current
            break
        if current.parent == current:
            break
        current = current.parent

    if not (project_root / "raw_data").exists() and not (project_root / "processed_data").exists():
        project_root = Path(__file__).resolve().parent

    return project_root


def ensure_dirs(run_output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "raw_data": run_output_dir / "raw_data",
        "processed_data": run_output_dir / "processed_data",
        "llm_responses": run_output_dir / "llm_responses",
        "metrics": run_output_dir / "metrics",
        "reports": run_output_dir / "reports",
        "logs": run_output_dir / "logs",
        "plots": run_output_dir / "plots",
        "private_truth": run_output_dir / "private_truth",
        "cases": run_output_dir / "raw_data" / "cases",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_xs_entry_path(xs_xml: Path, entry_path: str) -> Path:
    p = Path(entry_path)
    if not p.is_absolute():
        p = xs_xml.parent / p
    return p


def _materials_list(materials_field: object) -> List[str]:
    if isinstance(materials_field, str):
        return materials_field.split()
    if isinstance(materials_field, (list, tuple, set)):
        return [str(x) for x in materials_field]
    return [str(materials_field)]


def _validate_cross_sections_xml(xs_xml: Path, required_nuclides: Sequence[str]) -> Tuple[bool, str]:
    try:
        lib = openmc.data.DataLibrary.from_xml(str(xs_xml))
    except Exception as exc:
        return False, f"Unable to parse cross_sections.xml at {xs_xml}: {exc}"

    for nuclide in required_nuclides:
        match = None
        for entry in lib.libraries:
            if entry.get("type") != "neutron":
                continue
            materials = _materials_list(entry.get("materials", ""))
            if nuclide in materials:
                match = entry
                break

        if match is None:
            return False, f"Nuclide {nuclide} not declared in cross_sections.xml: {xs_xml}"

        resolved = _resolve_xs_entry_path(xs_xml, str(match.get("path", "")))
        if not resolved.exists():
            return False, f"Nuclide {nuclide} points to missing file: {resolved}"

    return True, "ok"


def _pick_cross_sections_xml(xs_override: Optional[str] = None) -> Path:
    candidates: List[Path] = []

    def add_candidate(p: Optional[str]) -> None:
        if not p:
            return
        path = Path(p).expanduser().resolve()
        if path not in candidates:
            candidates.append(path)

    add_candidate(xs_override)
    add_candidate(os.environ.get("OPENMC_CROSS_SECTIONS"))
    add_candidate(str((Path(__file__).resolve().parent / "cross_sections.xml")))
    add_candidate("/home/dina/ArcLink/physics_models/openmc/data/mcnp_endfb71/cross_sections.xml")
    add_candidate("/home/dina/ArcLink/data/openmc/endfb71/endfb-vii.1-hdf5/cross_sections.xml")

    required = ["U234", "U235", "U238", "O16", "H1"]
    diagnostics: List[str] = []
    for cand in candidates:
        if not cand.exists() or not os.access(cand, os.R_OK):
            diagnostics.append(f"- {cand}: missing or unreadable")
            continue
        ok, message = _validate_cross_sections_xml(cand, required)
        if ok:
            return cand
        diagnostics.append(f"- {cand}: {message}")

    detail = "\n".join(diagnostics) if diagnostics else "- no candidates were provided"
    raise FileNotFoundError(
        "Unable to find a valid OpenMC cross_sections.xml with required nuclides (U234,U235,U238,O16,H1).\n"
        f"Checked:\n{detail}"
    )


def configure_openmc(xs_override: Optional[str] = None, threads: int = 4) -> Dict[str, str]:
    thread_count = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["OPENMC_NUM_THREADS"] = str(thread_count)
    xs = _pick_cross_sections_xml(xs_override)
    openmc.config["cross_sections"] = str(xs)
    os.environ["OPENMC_CROSS_SECTIONS"] = str(xs)
    print(f"[runner] using OPENMC_CROSS_SECTIONS={xs}")
    mode = os.environ.get("OPENMC_CHAIN_MODE", "thermal").lower()
    chain_file = os.environ.get("OPENMC_CHAIN_FILE_THERMAL") if mode == "thermal" else os.environ.get("OPENMC_CHAIN_FILE_FAST")
    if chain_file:
        openmc.config["chain_file"] = chain_file
    return {"cross_sections": str(xs), "chain_mode": mode, "chain_file": chain_file or ""}


def verify_map() -> Dict[str, int]:
    counts = {"H": 0, "L": 0, "G": 0, "C": 0, "I": 0}
    for r in range(ASSEMBLY_SIZE):
        for c in range(ASSEMBLY_SIZE):
            sym = C2_MAP[r][c]
            counts[sym] += 1
    gd_from_map = {(r, c) for r in range(ASSEMBLY_SIZE) for c in range(ASSEMBLY_SIZE) if C2_MAP[r][c] == "G"}
    if gd_from_map != GD_POS:
        missing = sorted(GD_POS - gd_from_map)
        extra = sorted(gd_from_map - GD_POS)
        raise ValueError(f"GD_POS mismatch. missing={missing}, extra={extra}")
    if not (counts["H"] == 168 and counts["L"] == 52 and counts["G"] == 16 and counts["C"] == 16 and counts["I"] == 4):
        raise ValueError(f"Map counts mismatch: {counts}")
    return counts


def _edge_band(value: float, lo: float, hi: float, frac: float) -> str:
    width = max(hi - lo, 1e-12)
    band = frac * width
    if value <= lo + band:
        return "lower"
    if value >= hi - band:
        return "upper"
    return "interior"


def summarize_edge_concentration(points: Sequence[Dict[str, float]], frac: float = EDGE_PROXIMITY_FRACTION) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    n = max(len(points), 1)
    for feature, (lo, hi) in MULTIDIM_FEATURE_RANGES.items():
        lower = 0
        upper = 0
        for p in points:
            band = _edge_band(float(p[feature]), lo, hi, frac)
            if band == "lower":
                lower += 1
            elif band == "upper":
                upper += 1
        summary[feature] = {
            "lower_count": float(lower),
            "upper_count": float(upper),
            "lower_ratio": float(lower / n),
            "upper_ratio": float(upper / n),
        }
    return summary


def _sobol_points_in_unit_cube(n_points: int, dim: int, random_seed: int) -> np.ndarray:
    m = int(math.ceil(math.log2(max(1, n_points))))
    engine = qmc.Sobol(d=dim, scramble=True, seed=random_seed)
    pts = engine.random_base2(m=m)
    if len(pts) < n_points:
        extra = engine.random(n_points - len(pts))
        pts = np.vstack([pts, extra])
    return pts[:n_points, :]


def generate_sobol_design_matrix(n_unique: int = 60, random_seed: int = 42) -> List[Dict[str, float]]:
    pts = _sobol_points_in_unit_cube(n_points=n_unique, dim=4, random_seed=random_seed)
    l_bounds = np.array([
        MULTIDIM_FEATURE_RANGES["alpha_gd"][0],
        MULTIDIM_FEATURE_RANGES["fuel_temp"][0],
        MULTIDIM_FEATURE_RANGES["moderator_temp"][0],
        MULTIDIM_FEATURE_RANGES["boron_ppm"][0],
    ], dtype=float)
    u_bounds = np.array([
        MULTIDIM_FEATURE_RANGES["alpha_gd"][1],
        MULTIDIM_FEATURE_RANGES["fuel_temp"][1],
        MULTIDIM_FEATURE_RANGES["moderator_temp"][1],
        MULTIDIM_FEATURE_RANGES["boron_ppm"][1],
    ], dtype=float)
    scaled = qmc.scale(pts, l_bounds=l_bounds, u_bounds=u_bounds)

    rows: List[Dict[str, float]] = []
    for row in scaled:
        alpha = float(row[0])
        fuel_temp = float(row[1])
        moderator_temp = float(row[2])
        boron_ppm = float(row[3])
        if fuel_temp < moderator_temp:
            fuel_temp, moderator_temp = moderator_temp, fuel_temp
        rows.append(
            {
                "alpha_gd": alpha,
                "fuel_temp": fuel_temp,
                "moderator_temp": moderator_temp,
                "boron_ppm": boron_ppm,
                "replicate": False,
                "region_tag": "general_design",
            }
        )
    return rows


def generate_multidim_design_matrix(n_unique: int = 60, n_replicates: int = 12, random_seed: int = 42) -> List[Dict[str, float]]:
    points = generate_sobol_design_matrix(n_unique=n_unique, random_seed=random_seed)

    if n_unique >= 20:
        for alpha in TARGETED_ALPHA_ANCHORS:
            for fuel in TARGETED_FUEL_ANCHORS:
                for moderator in TARGETED_MODERATOR_SWEEP:
                    if fuel < moderator:
                        continue
                    points.append(
                        {
                            "alpha_gd": float(alpha),
                            "fuel_temp": float(fuel),
                            "moderator_temp": float(moderator),
                            "boron_ppm": float(TARGETED_BORON_FIXED),
                            "replicate": False,
                            "region_tag": "targeted_moderator_slice",
                        }
                    )

        for alpha in TARGETED_ALPHA_ANCHORS:
            for fuel in TARGETED_FUEL_ANCHORS:
                if fuel < TARGETED_MODERATOR_FIXED:
                    continue
                for boron in TARGETED_BORON_SWEEP:
                    points.append(
                        {
                            "alpha_gd": float(alpha),
                            "fuel_temp": float(fuel),
                            "moderator_temp": float(TARGETED_MODERATOR_FIXED),
                            "boron_ppm": float(boron),
                            "replicate": False,
                            "region_tag": "targeted_boron_slice",
                        }
                    )

        dedup: Dict[tuple, Dict[str, float]] = {}
        for p in points:
            key = (
                round(float(p["alpha_gd"]), 8),
                round(float(p["fuel_temp"]), 8),
                round(float(p["moderator_temp"]), 8),
                round(float(p["boron_ppm"]), 8),
                bool(p.get("replicate", False)),
                str(p.get("region_tag", "general_design")),
            )
            dedup[key] = p
        points = list(dedup.values())

    if n_replicates > 0:
        noisy = [p for p in points if p["alpha_gd"] >= 0.75 and p["boron_ppm"] >= 1500.0 and p["fuel_temp"] >= 700.0]
        reps = noisy[: min(n_replicates, len(noisy))]
        for i, rep in enumerate(reps):
            rep_copy = dict(rep)
            rep_copy["replicate"] = True
            rep_copy["region_tag"] = "replicate_noisy_region"
            rep_copy["case_id"] = f"replicate_{i + 1:03d}"
            points.append(rep_copy)
    for idx, point in enumerate(points):
        point.setdefault("case_id", f"case_{idx + 1:03d}")
    return points


def load_design_matrix(design_csv: Optional[str] = None, use_multidim: bool = False, n_unique: int = 60, n_replicates: int = 12) -> List[Dict[str, float]]:
    if design_csv:
        path = Path(design_csv).expanduser().resolve()
        df = pd.read_csv(path)
        required = ["alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Design CSV is missing required columns: {missing}")
        rows: List[Dict[str, float]] = []
        for row in df.to_dict(orient="records"):
            alpha = float(row["alpha_gd"])
            fuel_temp = float(row.get("fuel_temp", FUEL_TEMP_K))
            moderator_temp = float(row.get("moderator_temp", MODERATOR_TEMP_K))
            boron_ppm = float(row.get("boron_ppm", BORON_PPM))
            if fuel_temp < moderator_temp:
                raise ValueError(
                    f"Invalid design row: fuel_temp ({fuel_temp}) must be >= moderator_temp ({moderator_temp})"
                )
            for feature, value in {
                "alpha_gd": alpha,
                "fuel_temp": fuel_temp,
                "moderator_temp": moderator_temp,
                "boron_ppm": boron_ppm,
            }.items():
                lo, hi = MULTIDIM_FEATURE_RANGES[feature]
                if value < lo - 1e-12 or value > hi + 1e-12:
                    raise ValueError(f"Design CSV value out of bounds for {feature}: {value} not in [{lo}, {hi}]")
            rows.append({
                "alpha_gd": alpha,
                "fuel_temp": fuel_temp,
                "moderator_temp": moderator_temp,
                "boron_ppm": boron_ppm,
                "replicate": bool(row.get("replicate", False)),
                "region_tag": str(row.get("region_tag", "custom_design")),
            })
        return rows
    if use_multidim:
        return generate_multidim_design_matrix(n_unique=n_unique, n_replicates=n_replicates)
    return [{"alpha_gd": float(a), "fuel_temp": FUEL_TEMP_K, "moderator_temp": MODERATOR_TEMP_K, "boron_ppm": BORON_PPM, "replicate": False, "region_tag": "legacy_alpha_scan"} for a in ALPHA_GRID]


def make_materials(alpha_gd: float, fuel_temp: float = FUEL_TEMP_K, moderator_temp: float = MODERATOR_TEMP_K, boron_ppm: float = BORON_PPM) -> Dict[str, openmc.Material]:
    def make_uo2(name: str, enrichment: float, density: float) -> openmc.Material:
        m = openmc.Material(name=name)
        m.add_element("U", 1.0, enrichment=enrichment)
        m.add_element("O", 2.0)
        m.set_density("g/cm3", density)
        return m

    fuel_h = make_uo2("fuel_H", 3.64, FUEL_STACK_DENSITY)
    fuel_l = make_uo2("fuel_L", 3.14, FUEL_STACK_DENSITY)

    gd_wt_effective = 0.08 * alpha_gd
    if gd_wt_effective <= 0.0 or gd_wt_effective >= 0.25:
        raise ValueError(f"Invalid effective Gd2O3 weight fraction: {gd_wt_effective}")

    uo2_g = make_uo2("uo2_2wt", 2.0, UO2_TD)
    gd2o3 = openmc.Material(name="gd2o3")
    gd2o3.add_element("Gd", 2.0)
    gd2o3.add_element("O", 3.0)
    gd2o3.set_density("g/cm3", GD2O3_TD)

    fuel_g = openmc.Material.mix_materials([uo2_g, gd2o3], [1.0 - gd_wt_effective, gd_wt_effective], "wo", name="fuel_G")
    fuel_g.set_density("g/cm3", FUEL_STACK_DENSITY)

    clad = openmc.Material(name="clad")
    clad.set_density("g/cm3", 6.55)
    clad.add_element("Zr", 1.0)

    water = openmc.model.borated_water(
        boron_ppm,
        temperature=moderator_temp,
        pressure=MODERATOR_PRESSURE_MPA,
        name="water_borated",
    )
    if not any(name == "c_H_in_H2O" for name, *_ in getattr(water, "_sab", [])):
        water.add_s_alpha_beta("c_H_in_H2O")

    for m in (fuel_h, fuel_l, fuel_g):
        m.temperature = fuel_temp
    clad.temperature = CLAD_TEMP_K
    water.temperature = moderator_temp

    return {"H": fuel_h, "L": fuel_l, "G": fuel_g, "clad": clad, "water": water}


def build_model(
    alpha_gd: float,
    case_number: int,
    fuel_temp: float = FUEL_TEMP_K,
    moderator_temp: float = MODERATOR_TEMP_K,
    boron_ppm: float = BORON_PPM,
    particles: int = 20000,
    batches: int = 120,
    inactive: int = 30,
) -> Tuple[openmc.Model, Dict[str, float]]:
    mats = make_materials(alpha_gd, fuel_temp=fuel_temp, moderator_temp=moderator_temp, boron_ppm=boron_ppm)
    all_mats = openmc.Materials(list(mats.values()))

    fuel_or = openmc.ZCylinder(r=FUEL_R)
    clad_ir = openmc.ZCylinder(r=CLAD_INNER_R)
    clad_or = openmc.ZCylinder(r=CLAD_OUTER_R)
    tube_ir = openmc.ZCylinder(r=LARGE_TUBE_INNER_R)
    tube_or = openmc.ZCylinder(r=LARGE_TUBE_OUTER_R)

    zmin = openmc.ZPlane(z0=Z_MIN, boundary_type="reflective")
    zmax = openmc.ZPlane(z0=Z_MAX, boundary_type="reflective")

    pin_cells = {}
    for label in ["H", "L", "G"]:
        c1 = openmc.Cell(fill=mats[label], region=-fuel_or)
        c2 = openmc.Cell(fill=mats["water"], region=+fuel_or & -clad_ir)
        c3 = openmc.Cell(fill=mats["clad"], region=+clad_ir & -clad_or)
        c4 = openmc.Cell(fill=mats["water"], region=+clad_or)
        pin_cells[label] = openmc.Universe(cells=[c1, c2, c3, c4])

    tube_c1 = openmc.Cell(fill=mats["water"], region=-tube_ir)
    tube_c2 = openmc.Cell(fill=mats["clad"], region=+tube_ir & -tube_or)
    tube_c3 = openmc.Cell(fill=mats["water"], region=+tube_or)
    tube_univ = openmc.Universe(cells=[tube_c1, tube_c2, tube_c3])

    lat = openmc.RectLattice(name="apr1400_c2_16x16")
    lat.pitch = (PITCH_CM, PITCH_CM)
    lat.lower_left = (-0.5 * ASSEMBLY_SIZE * PITCH_CM, -0.5 * ASSEMBLY_SIZE * PITCH_CM)

    universes = []
    for r in range(ASSEMBLY_SIZE):
        row = []
        for c in range(ASSEMBLY_SIZE):
            sym = C2_MAP[r][c]
            if sym in {"H", "L", "G"}:
                row.append(pin_cells[sym])
            else:
                row.append(tube_univ)
        universes.append(row)
    lat.universes = universes

    half = 0.5 * ASSEMBLY_SIZE * PITCH_CM
    xmin = openmc.XPlane(x0=-half, boundary_type="reflective")
    xmax = openmc.XPlane(x0=half, boundary_type="reflective")
    ymin = openmc.YPlane(y0=-half, boundary_type="reflective")
    ymax = openmc.YPlane(y0=half, boundary_type="reflective")

    root_cell = openmc.Cell(fill=lat, region=+xmin & -xmax & +ymin & -ymax & +zmin & -zmax)
    geom = openmc.Geometry([root_cell])

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = int(particles)
    settings.batches = int(batches)
    settings.inactive = int(inactive)
    settings.seed = BASE_SEED + case_number
    settings.temperature = {"method": "interpolation"}
    bounds = [
        -half, -half, Z_MIN,
        half, half, Z_MAX,
    ]
    settings.source = openmc.IndependentSource(space=openmc.stats.Box(bounds[:3], bounds[3:]))

    tally = openmc.Tally(name="flux_tally")
    tally.scores = ["flux"]

    # Gd-specific observable: fission rate in Gd-bearing fuel pins vs total assembly fission
    gd_fiss_tally = openmc.Tally(name="gd_fission_tally")
    gd_fiss_tally.filters = [openmc.MaterialFilter([mats["G"]])]
    gd_fiss_tally.scores = ["fission"]

    tot_fiss_tally = openmc.Tally(name="total_fission_tally")
    tot_fiss_tally.filters = [openmc.MaterialFilter([mats["H"], mats["L"], mats["G"]])]
    tot_fiss_tally.scores = ["fission"]

    tallies = openmc.Tallies([tally, gd_fiss_tally, tot_fiss_tally])

    model = openmc.Model(geom, all_mats, settings, tallies)
    meta = {"half_width_cm": half, "gd_wt_effective": 0.08 * alpha_gd}
    return model, meta


def _library_temperatures(nuclide_or_table: str, data_type: str) -> set:
    import h5py
    lib = openmc.data.DataLibrary.from_xml(openmc.config["cross_sections"])
    for entry in lib.libraries:
        if entry["type"] == data_type and nuclide_or_table in entry["materials"]:
            path = Path(entry["path"])
            if not path.is_absolute():
                path = Path(openmc.config["cross_sections"]).parent / path
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing OpenMC nuclear-data file for {nuclide_or_table}: {path}. "
                    "Your OPENMC_CROSS_SECTIONS points to an incomplete or stale library. "
                    "Install the full ENDF/B-VII.1 or ENDF/B-VIII library or set OPENMC_CROSS_SECTIONS to the correct file."
                )
            with h5py.File(path, "r") as f:
                return {int(round(float(k[:-1]))) for k in f[nuclide_or_table]["kTs"].keys()}
    raise AssertionError(f"{nuclide_or_table} ({data_type}) not found in cross-section library")


def check_library_temperatures(mats: Dict[str, openmc.Material], fuel_temp: float = FUEL_TEMP_K, moderator_temp: float = MODERATOR_TEMP_K) -> Dict[str, str]:
    needed = {"H": fuel_temp, "L": fuel_temp, "G": fuel_temp, "clad": CLAD_TEMP_K, "water": moderator_temp}
    report: Dict[str, str] = {}
    for label, temp in needed.items():
        for nuc in mats[label].get_nuclides():
            have = _library_temperatures(nuc, "neutron")
            assert int(temp) in have, f"{nuc} ({label}) lacks {temp:.0f} K data; has {sorted(have)}"
        report[label] = f"neutron data at {temp:.0f} K present"
    have = _library_temperatures("c_H_in_H2O", "thermal")
    assert int(round(moderator_temp)) in have, f"c_H_in_H2O lacks {moderator_temp:.0f} K; has {sorted(have)}"
    report["thermal"] = f"c_H_in_H2O at {moderator_temp:.0f} K present"
    return report


def check_design_temperature_coverage(design_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    fuel_values = [float(r.get("fuel_temp", FUEL_TEMP_K)) for r in design_rows]
    moderator_values = [float(r.get("moderator_temp", MODERATOR_TEMP_K)) for r in design_rows]
    if len(fuel_values) == 0 or len(moderator_values) == 0:
        raise ValueError("Design contains no temperature values to validate.")

    neutron_temps = sorted(_library_temperatures("U234", "neutron"))
    thermal_temps = sorted(_library_temperatures("c_H_in_H2O", "thermal"))
    n_min, n_max = float(neutron_temps[0]), float(neutron_temps[-1])
    t_min, t_max = float(thermal_temps[0]), float(thermal_temps[-1])

    fmin, fmax = min(fuel_values), max(fuel_values)
    mmin, mmax = min(moderator_values), max(moderator_values)

    if fmin < n_min - 1e-9 or fmax > n_max + 1e-9:
        raise ValueError(
            f"Fuel temperatures [{fmin:.1f}, {fmax:.1f}] exceed neutron library support [{n_min:.1f}, {n_max:.1f}]"
        )
    if mmin < t_min - 1e-9 or mmax > t_max + 1e-9:
        raise ValueError(
            f"Moderator temperatures [{mmin:.1f}, {mmax:.1f}] exceed thermal library support [{t_min:.1f}, {t_max:.1f}]"
        )

    near_margin_k = 5.0
    if abs(fmin - n_min) <= near_margin_k or abs(fmax - n_max) <= near_margin_k:
        print(
            f"[design][warn] fuel temperatures approach neutron library boundaries: design=[{fmin:.1f}, {fmax:.1f}] K, library=[{n_min:.1f}, {n_max:.1f}] K"
        )
    if abs(mmin - t_min) <= near_margin_k or abs(mmax - t_max) <= near_margin_k:
        print(
            f"[design][warn] moderator temperatures approach thermal library boundaries: design=[{mmin:.1f}, {mmax:.1f}] K, library=[{t_min:.1f}, {t_max:.1f}] K"
        )

    return {
        "fuel_temp_min": float(fmin),
        "fuel_temp_max": float(fmax),
        "moderator_temp_min": float(mmin),
        "moderator_temp_max": float(mmax),
        "library_neutron_min": float(n_min),
        "library_neutron_max": float(n_max),
        "library_thermal_min": float(t_min),
        "library_thermal_max": float(t_max),
    }


def material_checks(alpha_gd: float, case_number: int, fuel_temp: float = FUEL_TEMP_K, moderator_temp: float = MODERATOR_TEMP_K, boron_ppm: float = BORON_PPM) -> Dict[str, float]:
    mats = make_materials(alpha_gd, fuel_temp=fuel_temp, moderator_temp=moderator_temp, boron_ppm=boron_ppm)
    out: Dict[str, float] = {"alpha_gd": alpha_gd, "case_number": case_number, "seed": BASE_SEED + case_number}
    for label in ["H", "L", "G"]:
        m = mats[label]
        dens = m.get_nuclide_atom_densities()
        metal = sum(v for k, v in dens.items() if k.startswith(("U", "Gd")))
        oxygen = sum(v for k, v in dens.items() if k.startswith("O"))
        out[f"{label}_density_g_cm3"] = float(m.density)
        out[f"{label}_O_per_metal"] = oxygen / metal
    gd = mats["G"]
    mass = {k: v * openmc.data.atomic_mass(k) for k, v in gd.get_nuclide_atom_densities().items()}
    gd_mass = sum(v for k, v in mass.items() if k.startswith("Gd"))
    m_gd = openmc.data.atomic_weight("Gd")
    m_gd2o3 = 2 * m_gd + 3 * openmc.data.atomic_weight("O")
    out["G_gd2o3_wt_fraction"] = gd_mass * m_gd2o3 / (2 * m_gd) / sum(mass.values())
    out["G_gd2o3_wt_expected"] = 0.08 * alpha_gd

    out["G_density_expected"] = FUEL_STACK_DENSITY

    tag = f"[alpha={alpha_gd:.3f}, case {case_number}]"
    assert math.isclose(out["G_gd2o3_wt_fraction"], out["G_gd2o3_wt_expected"], rel_tol=5e-4), \
        f"{tag} Gd2O3 wt fraction {out['G_gd2o3_wt_fraction']:.6f} != 0.08*alpha {out['G_gd2o3_wt_expected']:.6f}"
    assert math.isclose(out["G_density_g_cm3"], out["G_density_expected"], rel_tol=1e-6), \
        f"{tag} Gd fuel density {out['G_density_g_cm3']:.5f} != formula {out['G_density_expected']:.5f}"
    for label in ["H", "L"]:
        assert math.isclose(out[f"{label}_density_g_cm3"], FUEL_STACK_DENSITY, rel_tol=1e-9), \
            f"{tag} {label} density {out[f'{label}_density_g_cm3']} != {FUEL_STACK_DENSITY}"
        assert abs(out[f"{label}_O_per_metal"] - 2.0) < 1e-3, \
            f"{tag} {label} O/metal = {out[f'{label}_O_per_metal']:.4f}, expected 2"
    assert 1.90 <= out["G_O_per_metal"] <= 2.0001, f"{tag} G O/metal = {out['G_O_per_metal']:.4f} outside [1.90, 2.00]"

    for label in ["H", "L", "G"]:
        assert mats[label].temperature == fuel_temp, f"{tag} {label} temperature {mats[label].temperature}"
    assert mats["clad"].temperature == CLAD_TEMP_K, f"{tag} clad temperature {mats['clad'].temperature}"
    assert mats["water"].temperature == moderator_temp, f"{tag} water temperature {mats['water'].temperature}"
    assert any(n == "c_H_in_H2O" for n, *_ in mats["water"]._sab), f"{tag} water lacks c_H_in_H2O"

    model, _ = build_model(
        alpha_gd,
        case_number,
        fuel_temp=fuel_temp,
        moderator_temp=moderator_temp,
        boron_ppm=boron_ppm,
    )
    assert model.settings.seed == BASE_SEED + case_number, f"{tag} seed {model.settings.seed}"
    assert model.settings.temperature.get("method") == "interpolation", f"{tag} temperature method not 'interpolation'"
    return out


def run_case(
    alpha_gd: float,
    case_number: int,
    dirs: Dict[str, Path],
    fuel_temp: float = FUEL_TEMP_K,
    moderator_temp: float = MODERATOR_TEMP_K,
    boron_ppm: float = BORON_PPM,
    particles: int = 20000,
    batches: int = 120,
    inactive: int = 30,
) -> CaseResult:
    case_name = f"alpha_{alpha_gd:.3f}_fuel_{fuel_temp:.0f}_mod_{moderator_temp:.0f}_boron_{boron_ppm:.0f}".replace(".", "p")
    case_dir = dirs["cases"] / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    if os.environ.get("OPENMC_CROSS_SECTIONS"):
        openmc.config["cross_sections"] = os.environ["OPENMC_CROSS_SECTIONS"]
    if "cross_sections" not in openmc.config or not openmc.config["cross_sections"]:
        openmc.config["cross_sections"] = os.environ.get("OPENMC_CROSS_SECTIONS")

    model, _ = build_model(
        alpha_gd,
        case_number,
        fuel_temp=fuel_temp,
        moderator_temp=moderator_temp,
        boron_ppm=boron_ppm,
        particles=particles,
        batches=batches,
        inactive=inactive,
    )
    model.export_to_xml(directory=case_dir)

    openmc_exec = shutil.which("openmc")
    if openmc_exec is None:
        candidate = Path(sys.executable).resolve().parent / "openmc"
        if candidate.exists() and os.access(candidate, os.X_OK):
            openmc_exec = str(candidate)
    if openmc_exec is None:
        return CaseResult(
            alpha_gd, str(case_dir), 127, False, None, None, None, None, None, None, warnings, "openmc executable not found"
        )

    cmd = [openmc_exec]
    proc = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True, check=False, env=os.environ.copy())

    (dirs["logs"] / f"{case_name}_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (dirs["logs"] / f"{case_name}_stderr.log").write_text(proc.stderr or "", encoding="utf-8")

    statepoints = sorted(case_dir.glob("statepoint.*.h5"))
    statepoint = str(statepoints[-1]) if statepoints else None

    keff = None
    keff_std = None
    flux_mean = None
    gd_fission_share = None
    gd_fission_share_std = None

    if statepoint:
        try:
            with openmc.StatePoint(statepoint) as sp:
                keff = float(sp.keff.nominal_value)
                keff_std = float(sp.keff.std_dev)
                try:
                    t = sp.get_tally(name="flux_tally")
                    flux_mean = float(np.mean(t.mean))
                except Exception as exc:
                    warnings.append(f"Could not extract flux_mean: {exc}")

                try:
                    t_gd = sp.get_tally(name="gd_fission_tally")
                    t_tot = sp.get_tally(name="total_fission_tally")
                    f_gd = float(np.sum(t_gd.mean))
                    s_gd = float(np.sqrt(np.sum(np.square(t_gd.std_dev))))
                    f_tot = float(np.sum(t_tot.mean))
                    s_tot = float(np.sqrt(np.sum(np.square(t_tot.std_dev))))
                    if f_tot > 0.0:
                        gd_fission_share = f_gd / f_tot
                        gd_fission_share_std = gd_fission_share * math.sqrt(
                            (s_gd / max(f_gd, 1e-16)) ** 2 + (s_tot / f_tot) ** 2
                        )
                except Exception as exc:
                    warnings.append(f"Could not extract Gd fission share: {exc}")
        except Exception as exc:
            warnings.append(f"Could not extract keff: {exc}")
    else:
        warnings.append("Statepoint not found")

    stderr_text = (proc.stderr or "").lower()
    if "does not contain cross sections" in stderr_text:
        warnings.append("OpenMC cross-section temperature lookup failed; verify thermal interpolation coverage")
    if "material" in stderr_text and "not found" in stderr_text:
        warnings.append("OpenMC material lookup failure detected; inspect materials.xml and cross_sections.xml")
    if proc.returncode != 0:
        short_err = (proc.stderr or "").strip().splitlines()
        if short_err:
            warnings.append(f"stderr_tail: {short_err[-1][:220]}")

    success = proc.returncode == 0 and statepoint is not None and keff is not None
    err = None if success else f"OpenMC failed or incomplete outputs (rc={proc.returncode})"
    return CaseResult(
        alpha_gd,
        str(case_dir),
        proc.returncode,
        success,
        statepoint,
        keff,
        keff_std,
        flux_mean,
        gd_fission_share,
        gd_fission_share_std,
        warnings,
        err,
    )


def write_private_truth(private_truth_dir: Path, case_results: Sequence[CaseResult]) -> Path:
    manifest = private_truth_dir / "private_truth_manifest.json"
    lookup = private_truth_dir / "blind_alpha_lookup.json"
    payload = {
        "alpha_Gd_true": ALPHA_TRUE,
        "created_at": now_utc_iso(),
        "note": "Do not expose in public dataset",
        "public_dataset_mode": "blind_response_only",
        "n_private_alpha_rows": sum(1 for r in case_results if r.success),
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    alpha_by_row = [float(r.alpha_gd) for r in case_results if r.success]
    lookup.write_text(json.dumps({"alpha_by_row": alpha_by_row}, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="APR1400 OpenMC Phase I runner")
    parser.add_argument("--pilot", action="store_true", help="Run a quick 5-point alpha_Gd pilot sweep over 0.25 to 1.25.")
    parser.add_argument("--full", action="store_true", help="Force the legacy 41-point alpha_Gd sweep (default behavior).")
    parser.add_argument("--multidim", action="store_true", help="Generate and run a 4-feature feasible design matrix using alpha_gd, fuel_temp, moderator_temp, and boron_ppm.")
    parser.add_argument("--design-csv", default=None, help="Optional CSV containing a 4-feature design matrix with alpha_gd,fuel_temp,moderator_temp,boron_ppm columns.")
    parser.add_argument("--cross-sections", default=None, help="Optional explicit path to OpenMC cross_sections.xml")
    parser.add_argument("--workers", type=int, default=None, help="Optional OpenMC worker count (accepted for pipeline compatibility).")
    parser.add_argument("--threads", type=int, default=4, help="OpenMC threads per case.")
    parser.add_argument("--particles", type=int, default=20000, help="OpenMC particles per batch.")
    parser.add_argument("--batches", type=int, default=120, help="OpenMC total batches.")
    parser.add_argument("--inactive", type=int, default=30, help="OpenMC inactive batches.")
    parser.add_argument("--n-unique-full", type=int, default=60, help="Sobol point count for full multidimensional mode.")
    parser.add_argument("--resume", action="store_true", help="Accepted for pipeline compatibility.")
    parser.add_argument(
        "--run-output-dir",
        default=None,
        help="Optional output root for this Phase I execution. When set, all Phase I artifacts are written under this directory.",
    )
    args = parser.parse_args()

    if args.pilot and args.full:
        parser.error("Choose either --pilot or --full, not both.")

    use_multidim = bool(args.multidim or args.design_csv)
    pilot_mode = bool(args.pilot)
    mode_tag = "multidim" if use_multidim else ("pilot" if pilot_mode else "full")
    if use_multidim:
        n_unique = 5 if pilot_mode else int(args.n_unique_full)
        n_replicates = 0 if pilot_mode else 12
        design_rows = load_design_matrix(args.design_csv, use_multidim=True, n_unique=n_unique, n_replicates=n_replicates)
        print(f"[runner] using 4-feature design matrix with {len(design_rows)} cases ({mode_tag} mode)")
        edge_summary = summarize_edge_concentration(design_rows)
        for feature, stats in edge_summary.items():
            lower_ratio = float(stats["lower_ratio"])
            upper_ratio = float(stats["upper_ratio"])
            print(
                f"[design] {feature}: near-lower={int(stats['lower_count'])}/{len(design_rows)} ({lower_ratio:.1%}), "
                f"near-upper={int(stats['upper_count'])}/{len(design_rows)} ({upper_ratio:.1%})"
            )
            if lower_ratio >= EDGE_CONCENTRATION_WARN_RATIO or upper_ratio >= EDGE_CONCENTRATION_WARN_RATIO:
                print(
                    f"[design][warn] edge concentration for {feature} is high; consider increasing n_unique or reducing targeted slices"
                )
    else:
        alpha_grid = PILOT_ALPHA_GRID if pilot_mode else ALPHA_GRID
        design_rows = [{"alpha_gd": float(a), "fuel_temp": FUEL_TEMP_K, "moderator_temp": MODERATOR_TEMP_K, "boron_ppm": BORON_PPM, "replicate": False, "region_tag": "legacy_alpha_scan"} for a in alpha_grid]
        print(f"[runner] alpha grid selected: {alpha_grid} ({len(alpha_grid)} cases)")

    run_output_dir = Path(args.run_output_dir).expanduser().resolve() if args.run_output_dir else Path(__file__).resolve().parent
    run_output_dir.mkdir(parents=True, exist_ok=True)
    if Path.cwd().resolve() != run_output_dir.resolve():
        os.chdir(run_output_dir)

    dirs = ensure_dirs(run_output_dir)
    written: List[str] = []
    particles = int(args.particles)
    batches = int(args.batches)
    inactive = int(args.inactive)
    worker_count = 1 if args.workers is None else max(1, int(args.workers))

    if args.resume:
        print("[runner] note: --resume accepted but not currently used by this runner")

    try:
        project_root = find_project_root()
        _ = project_root
        cfg = configure_openmc(args.cross_sections, threads=args.threads)
    except Exception as exc:
        failure = {
            "status": "failed_missing_dependency",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "engine_execution": {
                "requested_engine": "openmc",
                "used_engine": "openmc",
                "used_real_engine": False,
                "fallback_reason": "OpenMC dependency/config failure"
            }
        }
        fpath = dirs["reports"] / "failure_report.json"
        fpath.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        print(f"- {fpath}")
        return

    counts = verify_map()
    (dirs["reports"] / "map_validation.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    written.append(str((dirs["reports"] / "map_validation.json").resolve()))

    temp_coverage = check_design_temperature_coverage(design_rows)
    print(
        "[design] temperature coverage: "
        f"fuel=[{temp_coverage['fuel_temp_min']:.1f}, {temp_coverage['fuel_temp_max']:.1f}] K within neutron library "
        f"[{temp_coverage['library_neutron_min']:.1f}, {temp_coverage['library_neutron_max']:.1f}] K; "
        f"moderator=[{temp_coverage['moderator_temp_min']:.1f}, {temp_coverage['moderator_temp_max']:.1f}] K within thermal library "
        f"[{temp_coverage['library_thermal_min']:.1f}, {temp_coverage['library_thermal_max']:.1f}] K"
    )

    truth_path = dirs["private_truth"] / "private_truth_manifest.json"
    written.append(str(truth_path.resolve()))

    checks: List[Dict[str, float]] = []
    checks_by_case: Dict[int, Dict[str, float]] = {}
    for case_number, design_row in enumerate(design_rows, start=1):
        alpha = float(design_row["alpha_gd"])
        fuel_temp = float(design_row.get("fuel_temp", FUEL_TEMP_K))
        moderator_temp = float(design_row.get("moderator_temp", MODERATOR_TEMP_K))
        boron_ppm = float(design_row.get("boron_ppm", BORON_PPM))
        chk = material_checks(alpha, case_number, fuel_temp=fuel_temp, moderator_temp=moderator_temp, boron_ppm=boron_ppm)
        checks.append(chk)
        checks_by_case[case_number] = chk
        print(f"[check] case {case_number} alpha={alpha:.3f} fuel={fuel_temp:.0f}K moderator={moderator_temp:.0f}K boron={boron_ppm:.0f}ppm seed={chk['seed']} "
              f"Gd2O3 wt={chk['G_gd2o3_wt_fraction']:.5f} (expect {chk['G_gd2o3_wt_expected']:.5f}) "
              f"rho_G={chk['G_density_g_cm3']:.4f} rho_H={chk['H_density_g_cm3']:.3f} rho_L={chk['L_density_g_cm3']:.3f} "
              f"O/M G={chk['G_O_per_metal']:.4f}")
    seeds = [c["seed"] for c in checks]
    assert len(set(seeds)) == len(seeds), "Seeds are not unique per case"
    lib_report = check_library_temperatures(make_materials(ALPHA_NOMINAL, fuel_temp=FUEL_TEMP_K, moderator_temp=MODERATOR_TEMP_K, boron_ppm=BORON_PPM))
    print(f"[check] unique seeds verified: {len(set(seeds))} distinct values across {len(seeds)} cases")
    print(f"[check] library temperatures: {lib_report}")
    checks_path = dirs["reports"] / "material_checks.json"
    checks_path.write_text(json.dumps({"library": lib_report, "cases": checks}, indent=2), encoding="utf-8")
    written.append(str(checks_path.resolve()))

    case_results: List[CaseResult] = []
    if worker_count > 1:
        print(f"[runner] executing {len(design_rows)} cases over {worker_count} workers")
        job_batches = partition_case_jobs(len(design_rows), worker_count)
        results_by_index: Dict[int, CaseResult] = {}
        from concurrent.futures import ProcessPoolExecutor

        for batch in job_batches:
            with ProcessPoolExecutor(max_workers=min(worker_count, len(batch))) as executor:
                futures = {}
                for case_number in batch:
                    design_row = design_rows[case_number - 1]
                    alpha = float(design_row["alpha_gd"])
                    fuel_temp = float(design_row.get("fuel_temp", FUEL_TEMP_K))
                    moderator_temp = float(design_row.get("moderator_temp", MODERATOR_TEMP_K))
                    boron_ppm = float(design_row.get("boron_ppm", BORON_PPM))
                    futures[executor.submit(
                        run_case,
                        alpha,
                        case_number,
                        dirs,
                        fuel_temp,
                        moderator_temp,
                        boron_ppm,
                        particles,
                        batches,
                        inactive,
                    )] = case_number
                for future in futures:
                    case_number = futures[future]
                    try:
                        results_by_index[case_number] = future.result()
                    except Exception as exc:
                        results_by_index[case_number] = CaseResult(
                            design_rows[case_number - 1]["alpha_gd"],
                            "",
                            1,
                            False,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            ["Exception in run_case"],
                            str(exc),
                        )
        case_results = [results_by_index[i] for i in range(1, len(design_rows) + 1)]
    else:
        for case_number, design_row in enumerate(design_rows, start=1):
            alpha = float(design_row["alpha_gd"])
            fuel_temp = float(design_row.get("fuel_temp", FUEL_TEMP_K))
            moderator_temp = float(design_row.get("moderator_temp", MODERATOR_TEMP_K))
            boron_ppm = float(design_row.get("boron_ppm", BORON_PPM))
            try:
                case_results.append(
                    run_case(
                        alpha,
                        case_number,
                        dirs,
                        fuel_temp=fuel_temp,
                        moderator_temp=moderator_temp,
                        boron_ppm=boron_ppm,
                        particles=particles,
                        batches=batches,
                        inactive=inactive,
                    )
                )
            except Exception as exc:
                case_results.append(
                    CaseResult(alpha, "", 1, False, None, None, None, None, None, None, ["Exception in run_case"], str(exc))
                )

    csv_path = dirs["processed_data"] / "openmc_dataset.csv"
    check_cols = sorted({k for chk in checks for k in chk.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha_gd",
            "fuel_temp",
            "moderator_temp",
            "boron_ppm",
            "case_number",
            "seed",
            "particles",
            "batches",
            "inactive",
            "keff",
            "keff_std",
            "flux_mean",
            "gd_fission_share",
            "gd_fission_share_std",
            "flux_used_in_inverse",
            "success",
            "case_dir",
            "statepoint",
            "returncode",
            *check_cols,
        ])
        for case_number, r in enumerate(case_results, start=1):
            chk = checks_by_case.get(case_number, {})
            design_row = design_rows[case_number - 1]
            writer.writerow([
                r.alpha_gd,
                float(design_row.get("fuel_temp", FUEL_TEMP_K)),
                float(design_row.get("moderator_temp", MODERATOR_TEMP_K)),
                float(design_row.get("boron_ppm", BORON_PPM)),
                case_number,
                BASE_SEED + case_number,
                particles,
                batches,
                inactive,
                r.keff,
                r.keff_std,
                r.flux_mean,
                r.gd_fission_share,
                r.gd_fission_share_std,
                False,
                r.success,
                r.case_dir,
                r.statepoint,
                r.returncode,
                *[chk.get(k) for k in check_cols],
            ])
    written.append(str(csv_path.resolve()))

    dataset_sha256 = _sha256_file(csv_path)

    blind_path = dirs["processed_data"] / "blind_observations.json"
    blind_obs = [
        {
            "keff": r.keff,
            "keff_std": r.keff_std,
            "flux_mean": r.flux_mean,
            "gd_fission_share": r.gd_fission_share,
            "gd_fission_share_std": r.gd_fission_share_std,
        }
        for r in case_results
        if r.success
    ]
    blind_path.write_text(json.dumps(blind_obs, indent=2), encoding="utf-8")
    written.append(str(blind_path.resolve()))

    truth_path = write_private_truth(dirs["private_truth"], case_results)
    written.append(str(truth_path.resolve()))

    registry_path = dirs["metrics"] / "dataset_registry.json"
    registry = {
        "created_at": now_utc_iso(),
        "engine_execution": {
            "requested_engine": "openmc",
            "used_engine": "openmc",
            "used_real_engine": True,
            "fallback_reason": ""
        },
        "config": cfg,
        "provenance": {
            "openmc_python_version": getattr(openmc, "__version__", "unknown"),
            "cross_sections": cfg.get("cross_sections", ""),
            "temperatures_K": {
                "fuel": FUEL_TEMP_K,
                "clad": CLAD_TEMP_K,
                "moderator": MODERATOR_TEMP_K,
            },
            "openmc_settings": {
                "particles": particles,
                "batches": batches,
                "inactive": inactive,
            },
            "dataset_csv_sha256": dataset_sha256,
            "phase1_config_sha256": _sha256_text(
                json.dumps(
                    {
                        "pitch_cm": PITCH_CM,
                        "assembly_size": ASSEMBLY_SIZE,
                        "design_rows": design_rows,
                        "base_seed": BASE_SEED,
                        "fuel_stack_density": FUEL_STACK_DENSITY,
                        "boron_ppm": BORON_PPM,
                        "moderator_temp_K": MODERATOR_TEMP_K,
                        "fuel_temp_K": FUEL_TEMP_K,
                        "clad_temp_K": CLAD_TEMP_K,
                        "moderator_pressure_MPa": MODERATOR_PRESSURE_MPA,
                    },
                    sort_keys=True,
                )
            ),
        },
        "n_cases": len(case_results),
        "n_success": sum(1 for r in case_results if r.success),
        "n_fail": sum(1 for r in case_results if not r.success),
        "design_temperature_coverage": temp_coverage,
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    written.append(str(registry_path.resolve()))

    successful = [r for r in case_results if r.success and r.keff is not None]
    if successful:
        x = [r.alpha_gd for r in successful]
        y = [r.keff for r in successful]
        plt.figure(figsize=(7, 4.5))
        plt.plot(x, y, marker="o")
        plt.xlabel("alpha_Gd")
        plt.ylabel("k-effective")
        plt.title("APR1400 C2: k-effective vs alpha_Gd")
        plt.grid(True, alpha=0.3)
        plot_path = dirs["plots"] / "keff_vs_alpha_gd.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        written.append(str(plot_path.resolve()))

    summary_path = dirs["reports"] / "result_summary.json"
    summary = {
        "status": "completed_with_partial_failures" if any(not r.success for r in case_results) else "completed",
        "created_at": now_utc_iso(),
        "mode": mode_tag,
        "expected_case_count": len(design_rows),
        "n_cases": len(case_results),
        "n_success": sum(1 for r in case_results if r.success),
        "n_fail": sum(1 for r in case_results if not r.success),
        "alpha_range": [float(min(float(r["alpha_gd"]) for r in design_rows)), float(max(float(r["alpha_gd"]) for r in design_rows))],
        "design_type": "multidim_feasible_design" if use_multidim else "legacy_alpha_scan",
        "engine_execution": {
            "requested_engine": "openmc",
            "used_engine": "openmc",
            "used_real_engine": True,
            "fallback_reason": ""
        },
        "failures": [
            {
                "alpha_gd": r.alpha_gd,
                "returncode": r.returncode,
                "error": r.error,
                "warnings": r.warnings,
            }
            for r in case_results if not r.success
        ]
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(str(summary_path.resolve()))

    mode_summary_path = dirs["reports"] / f"phase1_summary_{mode_tag}.json"
    mode_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(str(mode_summary_path.resolve()))

    print("WRITTEN FILES:")
    for p in written:
        print(f"- {p}")


if __name__ == "__main__":
    main()