#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent
PHASE1_SCRIPT = PROJECT_ROOT / "openmc_apr1400_phase1_runner_final.py"
PHASE2_SCRIPT = PROJECT_ROOT / "phase2_konly_analysis" / "src" / "phase2_analysis.py"

PILOT_EXPECTED_CASES = 5
FULL_EXPECTED_CASES = 41
PILOT_DEFAULT_PARTICLES = 200
PILOT_DEFAULT_BATCHES = 12
PILOT_DEFAULT_INACTIVE = 3
FULL_DEFAULT_PARTICLES = 6000
FULL_DEFAULT_BATCHES = 60
FULL_DEFAULT_INACTIVE = 20


@dataclass
class RunLayout:
    run_dir: Path
    phase1_dir: Path
    phase2_dir: Path
    phase2_results_dir: Path
    phase2_plots_dir: Path


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run APR1400 Phase I and Phase II in one mandatory, non-interactive sequence."
    )
    parser.add_argument("--pilot", action="store_true", help="Run pilot mode (5 cases).")
    parser.add_argument("--full", action="store_true", help="Run full mode (default behavior).")
    parser.add_argument("--cross-sections", default=None, help="Optional explicit path to OpenMC cross_sections.xml")
    parser.add_argument("--design-csv", default=None, help="Optional Phase I design CSV to pass through to the multidimensional runner.")
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"), help="Root directory for timestamped runs")
    parser.add_argument("--python", default=sys.executable, help="Python executable to launch Phase I/II")
    parser.add_argument("--workers", type=int, default=None, help="Parallel OpenMC workers for Phase I.")
    parser.add_argument("--threads", type=int, default=1, help="OpenMC threads per case for Phase I.")
    parser.add_argument("--n-unique-full", type=int, default=24, help="Sobol point count in full multidimensional mode.")
    parser.add_argument("--particles", type=int, default=None, help="OpenMC particles per batch for Phase I.")
    parser.add_argument("--batches", type=int, default=None, help="OpenMC total batches for Phase I.")
    parser.add_argument("--inactive", type=int, default=None, help="OpenMC inactive batches for Phase I.")
    parser.add_argument("--cv-folds", type=int, default=10, help="Cross-validation folds for Phase II.")
    parser.add_argument("--models", default="gp,wlin,wquad,wquad_loo", help="Models passed to Phase II.")
    parser.add_argument("--experiments", default="", help="Optional Phase II experiments (e.g., learning,edges,active).")
    parser.add_argument("--resume", action="store_true", help="Forward resume mode to Phase I.")
    args = parser.parse_args(argv)

    if args.pilot and args.full:
        parser.error("Choose either --pilot or --full, not both.")

    return args


def create_run_layout(runs_dir: Path, mode_tag: str) -> RunLayout:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:6]
    run_dir = runs_dir / f"{stamp}_{mode_tag}_{suffix}"

    phase1_dir = run_dir / "phase1"
    phase2_dir = run_dir / "phase2"
    phase2_results_dir = phase2_dir / "results"
    phase2_plots_dir = phase2_dir / "plots"

    phase1_dir.mkdir(parents=True, exist_ok=False)
    phase2_results_dir.mkdir(parents=True, exist_ok=False)
    phase2_plots_dir.mkdir(parents=True, exist_ok=False)

    return RunLayout(
        run_dir=run_dir,
        phase1_dir=phase1_dir,
        phase2_dir=phase2_dir,
        phase2_results_dir=phase2_results_dir,
        phase2_plots_dir=phase2_plots_dir,
    )


def write_failure_report(layout: RunLayout, phase: str, mode_tag: str, message: str, details: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "created_at": now_utc_iso(),
        "status": "failed",
        "mode": mode_tag,
        "phase": phase,
        "message": message,
        "run_dir": str(layout.run_dir),
    }
    if details:
        payload["details"] = details

    out_path = layout.run_dir / "failure_report.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[pipeline] failure report: {out_path}")


def run_cmd(cmd: List[str], label: str) -> None:
    print(f"[pipeline] running {label}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def validate_phase1_dataset(layout: RunLayout, mode_tag: str) -> Dict[str, Any]:
    dataset_csv = layout.phase1_dir / "processed_data" / "openmc_dataset.csv"
    lookup_json = layout.phase1_dir / "private_truth" / "blind_alpha_lookup.json"

    if not dataset_csv.exists():
        raise FileNotFoundError(f"Phase I dataset missing: {dataset_csv}")
    if not lookup_json.exists():
        raise FileNotFoundError(f"Phase I alpha lookup missing: {lookup_json}")

    rows: List[Dict[str, Any]] = []
    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    success_rows = [r for r in rows if _parse_bool(r.get("success"))]
    if len(success_rows) == 0:
        raise ValueError("Phase I produced no successful rows; the 4D design run failed before Phase II could link.")

    fieldnames = set(rows[0].keys()) if rows else set()
    is_multidim = {"alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm"}.issubset(fieldnames)

    if is_multidim:
        min_required = 5 if mode_tag == "pilot" else 10
        if len(success_rows) < min_required:
            raise ValueError(
                f"4D Phase I pilot/full run produced too few successful rows: found {len(success_rows)}, need at least {min_required} for mode={mode_tag}."
            )
    else:
        expected = PILOT_EXPECTED_CASES if mode_tag == "pilot" else FULL_EXPECTED_CASES
        if len(rows) != expected:
            raise ValueError(
                f"Phase I mode/data mismatch: expected {expected} rows for mode={mode_tag}, found {len(rows)} rows"
            )
        if len(success_rows) != expected:
            raise ValueError(
                f"Phase I mode/data mismatch: expected {expected} successful rows for mode={mode_tag}, "
                f"found {len(success_rows)} successful rows"
            )

    alpha_vals = [float(r["alpha_gd"]) for r in success_rows]
    if min(alpha_vals) < 0.25 - 1e-9 or max(alpha_vals) > 1.25 + 1e-9:
        raise ValueError(
            f"Phase I alpha range out of bounds: [{min(alpha_vals):.4f}, {max(alpha_vals):.4f}] not within [0.25, 1.25]"
        )

    return {
        "dataset_csv": dataset_csv,
        "lookup_json": lookup_json,
        "n_rows": len(rows),
        "n_success": len(success_rows),
        "alpha_min": min(alpha_vals),
        "alpha_max": max(alpha_vals),
    }


def write_pipeline_summary(layout: RunLayout, mode_tag: str, phase1_info: Dict[str, Any]) -> Path:
    out = {
        "created_at": now_utc_iso(),
        "status": "completed",
        "mode": mode_tag,
        "run_dir": str(layout.run_dir),
        "phase1": {
            "output_dir": str(layout.phase1_dir),
            "dataset_csv": str(phase1_info["dataset_csv"]),
            "lookup_json": str(phase1_info["lookup_json"]),
            "n_rows": int(phase1_info["n_rows"]),
            "n_success": int(phase1_info["n_success"]),
            "alpha_min": float(phase1_info["alpha_min"]),
            "alpha_max": float(phase1_info["alpha_max"]),
        },
        "phase2": {
            "output_dir": str(layout.phase2_results_dir),
            "plots_dir": str(layout.phase2_plots_dir),
            "summary_json": str(layout.phase2_results_dir / f"phase2_summary_{mode_tag}.json"),
        },
    }

    path = layout.run_dir / f"pipeline_summary_{mode_tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    mode_tag = "pilot" if args.pilot else "full"
    runs_dir = Path(args.runs_dir).expanduser().resolve()

    particles = int(args.particles) if args.particles is not None else (
        PILOT_DEFAULT_PARTICLES if mode_tag == "pilot" else FULL_DEFAULT_PARTICLES
    )
    batches = int(args.batches) if args.batches is not None else (
        PILOT_DEFAULT_BATCHES if mode_tag == "pilot" else FULL_DEFAULT_BATCHES
    )
    inactive = int(args.inactive) if args.inactive is not None else (
        PILOT_DEFAULT_INACTIVE if mode_tag == "pilot" else FULL_DEFAULT_INACTIVE
    )

    try:
        if not PHASE1_SCRIPT.exists():
            raise FileNotFoundError(f"Phase I script not found: {PHASE1_SCRIPT}")
        if not PHASE2_SCRIPT.exists():
            raise FileNotFoundError(f"Phase II script not found: {PHASE2_SCRIPT}")

        layout = create_run_layout(runs_dir=runs_dir, mode_tag=mode_tag)
        print(f"[pipeline] created run folder: {layout.run_dir}")
        print(f"[pipeline] mode: {mode_tag}")

        phase1_cmd = [
            args.python,
            str(PHASE1_SCRIPT),
            "--multidim",
            "--pilot" if mode_tag == "pilot" else "--full",
            "--run-output-dir",
            str(layout.phase1_dir),
        ]
        if args.design_csv:
            phase1_cmd.extend(["--design-csv", str(Path(args.design_csv).expanduser().resolve())])
        if args.resume:
            phase1_cmd.append("--resume")
        if args.cross_sections:
            phase1_cmd.extend(["--cross-sections", str(args.cross_sections)])
        if args.workers is not None:
            phase1_cmd.extend(["--workers", str(int(args.workers))])
        if args.threads is not None:
            phase1_cmd.extend(["--threads", str(int(args.threads))])
        if mode_tag == "full":
            phase1_cmd.extend(["--n-unique-full", str(int(args.n_unique_full))])
        phase1_cmd.extend([
            "--particles",
            str(particles),
            "--batches",
            str(batches),
            "--inactive",
            str(inactive),
        ])

        run_cmd(phase1_cmd, "Phase I")
        phase1_info = validate_phase1_dataset(layout, mode_tag)
        print(
            "[pipeline] Phase I validated: "
            f"rows={phase1_info['n_rows']} success={phase1_info['n_success']} "
            f"alpha=[{phase1_info['alpha_min']:.3f}, {phase1_info['alpha_max']:.3f}]"
        )

        phase2_cmd = [
            args.python,
            str(PHASE2_SCRIPT),
            "--pilot" if mode_tag == "pilot" else "--full",
            "--dataset-csv",
            str(phase1_info["dataset_csv"]),
            "--secret-alpha-lookup",
            str(phase1_info["lookup_json"]),
            "--output-dir",
            str(layout.phase2_results_dir),
            "--plots-dir",
            str(layout.phase2_plots_dir),
            "--cv-folds",
            str(int(args.cv_folds)),
            "--models",
            str(args.models),
        ]
        if args.experiments:
            phase2_cmd.extend(["--experiments", str(args.experiments)])

        run_cmd(phase2_cmd, "Phase II")

        phase2_summary = layout.phase2_results_dir / f"phase2_summary_{mode_tag}.json"
        if not phase2_summary.exists():
            raise FileNotFoundError(f"Expected Phase II summary missing: {phase2_summary}")

        pipeline_summary = write_pipeline_summary(layout, mode_tag, phase1_info)
        print(f"[pipeline] Phase II summary: {phase2_summary}")
        print(f"[pipeline] Pipeline summary: {pipeline_summary}")
        print("[pipeline] completed successfully")
        return 0

    except subprocess.CalledProcessError as exc:
        details = {
            "returncode": exc.returncode,
            "command": exc.cmd,
            "traceback": traceback.format_exc(),
        }
        if "layout" in locals():
            write_failure_report(layout, phase="subprocess", mode_tag=mode_tag, message=str(exc), details=details)
        print(f"[pipeline] ERROR: command failed with return code {exc.returncode}")
        return 1
    except Exception as exc:
        details = {"traceback": traceback.format_exc()}
        if "layout" in locals():
            write_failure_report(layout, phase="validation_or_setup", mode_tag=mode_tag, message=str(exc), details=details)
        print(f"[pipeline] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())