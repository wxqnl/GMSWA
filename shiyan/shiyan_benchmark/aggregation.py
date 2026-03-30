from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _load_runs(paths: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_json(Path(path), lines=True) for path in paths]
    if not frames:
        raise ValueError("No input run logs provided.")
    return pd.concat(frames, ignore_index=True)


def _maybe_retention(df: pd.DataFrame) -> pd.DataFrame:
    if "model_name" not in df.columns or "score" not in df.columns:
        return df
    baseline = df[df["model_name"].eq("full_attention")]
    if baseline.empty:
        df["retention"] = None
        return df
    join_cols = [col for col in ["sample_id", "task_name", "task_group", "condition"] if col in df.columns]
    baseline = baseline[join_cols + ["score"]].rename(columns={"score": "baseline_score"})
    merged = df.merge(baseline, on=join_cols, how="left")
    merged["retention"] = merged["score"] / merged["baseline_score"]
    return merged


def aggregate_logs(
    *,
    input_paths: list[str | Path],
    experiment: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    df = _maybe_retention(_load_runs(input_paths))

    table_path = output_root / f"{experiment}.csv"
    df.to_csv(table_path, index=False)

    if experiment == "exp1":
        group_cols = ["task_group", "model_name"]
        summary = (
            df.groupby(group_cols, dropna=False)
            .agg(
                mean_accuracy=("is_correct", "mean"),
                mean_score=("score", "mean"),
                mean_margin=("margin", "mean"),
                mean_retention=("retention", "mean"),
                count=("sample_id", "count"),
            )
            .reset_index()
        )
    elif experiment == "exp2":
        group_cols = ["task_group", "model_name", "condition"]
        summary = (
            df.groupby(group_cols, dropna=False)
            .agg(
                mean_score=("score", "mean"),
                mean_delta_vs_none=("delta_vs_none", "mean"),
                count=("sample_id", "count"),
            )
            .reset_index()
        )
    else:
        group_cols = ["task_group", "model_name", "memory_type", "write_type"]
        summary = (
            df.groupby(group_cols, dropna=False)
            .agg(
                mean_accuracy=("is_correct", "mean"),
                mean_score=("score", "mean"),
                mean_margin=("margin", "mean"),
                count=("sample_id", "count"),
            )
            .reset_index()
        )

    summary_path = output_root / f"{experiment}_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {
        "table": table_path,
        "summary": summary_path,
    }
