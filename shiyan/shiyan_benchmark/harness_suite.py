from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .modeling import ensure_python_paths

ensure_python_paths()
from lm_eval.evaluator import simple_evaluate


def load_suite_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _save_samples(base_dir: Path, run_name: str, samples: dict[str, list[dict[str, Any]]] | None) -> list[Path]:
    saved_paths: list[Path] = []
    if not samples:
        return saved_paths
    sample_dir = base_dir / "samples" / run_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    for task_name, rows in samples.items():
        path = sample_dir / f"{task_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        saved_paths.append(path)
    return saved_paths


def _flatten_results(
    suite_name: str,
    run_name: str,
    task_group: str | None,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name, metrics in result.get("results", {}).items():
        for metric_name, value in metrics.items():
            if metric_name.endswith("_stderr") or metric_name.endswith(",stderr"):
                continue
            rows.append(
                {
                    "suite_name": suite_name,
                    "run_name": run_name,
                    "task_group": task_group,
                    "task_name": task_name,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "n_samples": result.get("n-samples", {}).get(task_name),
                    "alias": result.get("configs", {}).get(task_name, {}).get("alias"),
                }
            )
    return rows


def run_harness_suite(
    *,
    suite: dict[str, Any],
    lm,
    output_dir: str | Path,
    shared_metadata: dict[str, Any] | None = None,
    batch_size: int | str | None = 1,
    limit: int | float | None = None,
    bootstrap_iters: int = 0,
    log_samples: bool = True,
    continue_on_error: bool = True,
    task_manager=None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    suite_name = suite["suite_name"]
    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    run_outputs: list[dict[str, Any]] = []

    for run in suite["runs"]:
        if run.get("kind", "harness") != "harness":
            continue
        run_name = run["run_name"]
        metadata = dict(shared_metadata or {})
        metadata.update(run.get("metadata", {}))
        try:
            result = simple_evaluate(
                model=lm,
                tasks=run["tasks"],
                batch_size=batch_size,
                limit=limit,
                bootstrap_iters=bootstrap_iters,
                log_samples=log_samples,
                metadata=metadata,
                task_manager=task_manager,
            )
        except Exception as exc:
            error_payload = {
                "suite_name": suite_name,
                "run_name": run_name,
                "task_group": run.get("task_group"),
                "error": str(exc),
            }
            _save_json(suite_dir / f"{run_name}.error.json", error_payload)
            run_outputs.append(
                {
                    "run_name": run_name,
                    "error": str(exc),
                    "error_path": str(suite_dir / f"{run_name}.error.json"),
                }
            )
            if continue_on_error:
                continue
            raise
        if result is None:
            continue
        error_path = suite_dir / f"{run_name}.error.json"
        if error_path.exists():
            error_path.unlink()
        _save_json(suite_dir / f"{run_name}.json", result)
        sample_paths = _save_samples(suite_dir, run_name, result.get("samples"))
        run_outputs.append(
            {
                "run_name": run_name,
                "result_path": str(suite_dir / f"{run_name}.json"),
                "sample_paths": [str(path) for path in sample_paths],
            }
        )
        summary_rows.extend(
            _flatten_results(
                suite_name=suite_name,
                run_name=run_name,
                task_group=run.get("task_group"),
                result=result,
            )
        )

    summary_path = suite_dir / "summary.csv"
    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    manifest_copy = suite_dir / "suite_manifest.json"
    _save_json(manifest_copy, suite)
    return {
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "summary_path": str(summary_path),
        "runs": run_outputs,
    }
