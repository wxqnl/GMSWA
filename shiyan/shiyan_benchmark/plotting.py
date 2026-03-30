from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_exp1(summary_csv: str | Path, output_path: str | Path) -> Path:
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = df.pivot(index="task_group", columns="model_name", values="mean_accuracy")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean Accuracy")
    ax.set_xlabel("Task Group")
    ax.set_title("Exp1 Task Type Decomposition")
    ax.legend(title="Model", loc="best")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_exp2(summary_csv: str | Path, output_path: str | Path) -> Path:
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (task_group, model_name), group in df.groupby(["task_group", "model_name"], dropna=False):
        ordered = group.set_index("condition").reindex(["none", "summary_only", "memory_only", "original"]).reset_index()
        label = f"{task_group} / {model_name}"
        ax.plot(ordered["condition"], ordered["mean_score"], marker="o", label=label)
    ax.set_ylabel("Mean Target Log-Prob")
    ax.set_xlabel("Condition")
    ax.set_title("Exp2 Minimal Recovery")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_trace_heatmap(trace_csv: str | Path, output_path: str | Path) -> Path:
    df = pd.read_csv(trace_csv)
    if df.empty:
        raise ValueError("Trace CSV is empty.")
    sample_id = df["sample_id"].dropna().iloc[0]
    pivot = df.pivot_table(index="slot_id", columns="step_id", values="gate_value", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xlabel("Step")
    ax.set_ylabel("Slot")
    ax.set_title(f"Gate Trace Heatmap: {sample_id}")
    fig.colorbar(im, ax=ax, label="Gate Value")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output
