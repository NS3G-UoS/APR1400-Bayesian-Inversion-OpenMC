#!/usr/bin/env python3
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "phase2_konly_analysis/src")
from phase2_analysis import (  # noqa: E402
    alpha_metrics,
    binned_summary_from_long,
    evaluate_splits,
    grouped_cv_folds,
    grouped_splits,
    loo_summary,
    loo_table,
    null_baseline_metrics,
    paired_tests_from_long,
    save_model_comparison_overlay,
    write_goal_report,
)


def main():
    base = Path("runs/20261113_084819_full_0037a1/phase2/results")
    plots = Path("runs/20261113_084819_full_0037a1/phase2/plots")
    base.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("runs/20261113_084819_full_0037a1/phase1/processed_data/openmc_dataset.csv")
    fold_idx = grouped_cv_folds(df.reset_index(drop=True), n_folds=10, group_mode="alpha")
    splits = grouped_splits(df.reset_index(drop=True), fold_idx)

    for scenario_tag, measurement_sd in (("obs50pcm", 0.0005), ("obs0", 0.0)):
        by_model = {}
        summary_rows = []

        for model_key in ("gp", "wlin", "wquad"):
            loo = loo_table(df, model_key, measurement_sd=measurement_sd)
            by_model[model_key] = loo
            loo.to_csv(base / f"loo_inverse_recovery_{model_key}_full_{scenario_tag}.csv", index=False)
            alpha_metrics(loo).to_csv(base / f"loo_alpha_metrics_{model_key}_full_{scenario_tag}.csv", index=False)
            summary_rows.append(loo_summary(loo))

            unique_idx = df.assign(_a=df["alpha_gd"].round(6)).drop_duplicates(subset=["_a"]).index
            unique_df = df.loc[unique_idx].reset_index(drop=True)
            unique_loo = loo_table(unique_df, model_key, measurement_sd=measurement_sd)
            unique_loo.to_csv(base / f"loo_inverse_recovery_{model_key}_full_{scenario_tag}_unique_alpha.csv", index=False)
            pd.DataFrame([loo_summary(unique_loo)]).to_csv(
                base / f"loo_model_summary_{model_key}_full_{scenario_tag}_unique_alpha.csv", index=False
            )

            grouped_eval = evaluate_splits(df.reset_index(drop=True), model_key, splits, measurement_sd=measurement_sd)
            grouped_eval.to_csv(base / f"grouped_alpha_cv_{model_key}_full_{scenario_tag}.csv", index=False)
            pd.DataFrame([loo_summary(grouped_eval)]).to_csv(
                base / f"grouped_alpha_cv_summary_{model_key}_full_{scenario_tag}.csv", index=False
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("model")
        summary_df.to_csv(base / f"loo_model_summary_full_{scenario_tag}.csv", index=False)

        loo_long = pd.concat([by_model[m] for m in ("gp", "wlin", "wquad")], ignore_index=True)
        loo_long.to_csv(base / f"loo_inverse_recovery_models_full_{scenario_tag}.csv", index=False)

        paired_tests_from_long(loo_long).to_csv(base / f"paired_tests_full_{scenario_tag}.csv", index=False)
        pd.DataFrame([null_baseline_metrics(loo_long[loo_long.model == "gp"])]).to_csv(
            base / f"null_baseline_full_{scenario_tag}.csv", index=False
        )
        binned_summary_from_long(loo_long, n_bins=5).to_csv(
            base / f"loo_model_summary_binned_alpha_full_{scenario_tag}.csv", index=False
        )
        save_model_comparison_overlay(summary_df, out_dir=plots, pilot_mode=False, scenario_tag=scenario_tag)

    write_goal_report(pd.read_csv(base / "loo_inverse_recovery_gp_full_obs50pcm.csv"), base / "goal_report_full.json")

    # Legacy aliases
    pd.read_csv(base / "loo_model_summary_full_obs50pcm.csv").to_csv(base / "loo_model_summary_full.csv", index=False)
    pd.read_csv(base / "loo_inverse_recovery_models_full_obs50pcm.csv").to_csv(
        base / "loo_inverse_recovery_models_full.csv", index=False
    )
    pd.read_csv(base / "paired_tests_full_obs50pcm.csv").to_csv(base / "paired_tests_full.csv", index=False)
    pd.read_csv(base / "null_baseline_full_obs50pcm.csv").to_csv(base / "null_baseline_full.csv", index=False)

    print("regenerated phase2 artifacts")


if __name__ == "__main__":
    main()
