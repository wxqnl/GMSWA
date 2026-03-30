from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .modeling import ensure_python_paths
from .tracing import MemoryTraceRecorder

ensure_python_paths()
from lm_eval.api.instance import Instance


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if limit is not None and line_idx >= limit:
                break
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_trace_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import csv

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "step_id",
        "slot_id",
        "gate_value",
        "slot_norm",
        "read_weight",
        "event_type",
        "event_just_evicted",
        "layer_name",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_choices(sample: dict[str, Any]) -> list[dict[str, str]]:
    if "choices" in sample:
        return sample["choices"]
    pairs = []
    for label in ["A", "B", "C", "D", "E"]:
        key = f"choice_{label.lower()}"
        if key in sample:
            pairs.append({"label": label, "text": sample[key]})
    if not pairs:
        raise ValueError(f"Sample {sample.get('sample_id')} does not contain choices.")
    return pairs


def _continuation_token_count(lm, context: str, continuation: str) -> int:
    if context:
        _, continuation_enc = lm._encode_pair(context, continuation)
    else:
        continuation_enc = lm.tok_encode(continuation, add_special_tokens=False)
    return max(1, len(continuation_enc))


def _batched_score(lm, requests: list[tuple[str, str]]) -> list[dict[str, float | bool | int]]:
    instances = [
        Instance("loglikelihood", {}, (context, continuation), idx)
        for idx, (context, continuation) in enumerate(requests)
    ]
    outputs = lm.loglikelihood(instances)
    scored: list[dict[str, float | bool | int]] = []
    for (context, continuation), (logprob, is_greedy) in zip(requests, outputs, strict=True):
        token_count = _continuation_token_count(lm, context, continuation)
        scored.append(
            {
                "sum_logprob": float(logprob),
                "mean_logprob": float(logprob) / token_count,
                "token_count": token_count,
                "is_greedy": bool(is_greedy),
            }
        )
    return scored


def evaluate_experiment(
    *,
    experiment: str,
    lm,
    samples: list[dict[str, Any]],
    model_name: str,
    window_size: int | None = None,
    condition: str = "default",
    memory_type: str = "none",
    write_type: str = "no_write",
    trace_recorder: MemoryTraceRecorder | None = None,
) -> list[dict[str, Any]]:
    if experiment not in {"exp1", "exp2", "exp3", "exp4"}:
        raise ValueError(f"Unsupported experiment: {experiment}")
    if experiment == "exp2":
        return _evaluate_minimal_recovery(
            experiment=experiment,
            lm=lm,
            samples=samples,
            model_name=model_name,
            window_size=window_size,
            memory_type=memory_type,
            write_type=write_type,
            trace_recorder=trace_recorder,
        )
    return _evaluate_choice_scoring(
        experiment=experiment,
        lm=lm,
        samples=samples,
        model_name=model_name,
        window_size=window_size,
        condition=condition,
        memory_type=memory_type,
        write_type=write_type,
        trace_recorder=trace_recorder,
    )


def _evaluate_choice_scoring(
    *,
    experiment: str,
    lm,
    samples: list[dict[str, Any]],
    model_name: str,
    window_size: int | None,
    condition: str,
    memory_type: str,
    write_type: str,
    trace_recorder: MemoryTraceRecorder | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if trace_recorder is not None:
            trace_recorder.set_sample(sample.get("sample_id"))
        choices = _normalize_choices(sample)
        requests = [(sample["context"], choice["text"]) for choice in choices]
        if trace_recorder is None:
            scores = _batched_score(lm, requests)
        else:
            scores = []
            for request in requests:
                scores.extend(_batched_score(lm, [request]))
        labeled_scores = [
            {
                "label": choice["label"],
                "text": choice["text"],
                **score,
            }
            for choice, score in zip(choices, scores, strict=True)
        ]
        ranked = sorted(labeled_scores, key=lambda item: item["mean_logprob"], reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        prediction = best["label"]
        margin = best["mean_logprob"] - second["mean_logprob"] if second is not None else None
        rows.append(
            {
                "experiment": experiment,
                "sample_id": sample.get("sample_id"),
                "task_name": sample.get("task_name"),
                "task_group": sample.get("task_group"),
                "model_name": model_name,
                "context_length": len(sample.get("context", "")),
                "window_size": window_size,
                "condition": condition,
                "memory_type": memory_type,
                "write_type": write_type,
                "score": best["mean_logprob"],
                "sum_logprob": best["sum_logprob"],
                "metric_name": "mean_logprob_choice",
                "prediction": prediction,
                "target": sample.get("label"),
                "is_correct": prediction == sample.get("label"),
                "margin": margin,
                "choice_scores": json.dumps(labeled_scores, ensure_ascii=False),
                "extra_notes": None,
            }
        )
    return rows


def _evaluate_minimal_recovery(
    *,
    experiment: str,
    lm,
    samples: list[dict[str, Any]],
    model_name: str,
    window_size: int | None,
    memory_type: str,
    write_type: str,
    trace_recorder: MemoryTraceRecorder | None,
) -> list[dict[str, Any]]:
    condition_map = {
        "original": "context_original",
        "summary_only": "context_summary",
        "memory_only": "context_memory",
        "none": "context_none",
    }
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if trace_recorder is not None:
            trace_recorder.set_sample(sample.get("sample_id"))
        requests = [
            (sample[field_name], sample["target_continuation"])
            for field_name in condition_map.values()
        ]
        if trace_recorder is None:
            scores = _batched_score(lm, requests)
        else:
            scores = []
            for request in requests:
                scores.extend(_batched_score(lm, [request]))
        by_condition = {
            condition_name: score
            for condition_name, score in zip(condition_map.keys(), scores, strict=True)
        }
        none_score = by_condition["none"]["mean_logprob"]
        for condition_name, score in by_condition.items():
            rows.append(
                {
                    "experiment": experiment,
                    "sample_id": sample.get("sample_id"),
                    "task_name": sample.get("task_name"),
                    "task_group": sample.get("task_group"),
                    "model_name": model_name,
                    "context_length": len(sample.get(condition_map[condition_name], "")),
                    "window_size": window_size,
                    "condition": condition_name,
                    "memory_type": memory_type,
                    "write_type": write_type,
                    "score": score["mean_logprob"],
                    "sum_logprob": score["sum_logprob"],
                    "metric_name": "mean_logprob_target",
                    "prediction": None,
                    "target": sample.get("target_continuation"),
                    "is_correct": None,
                    "delta_vs_none": score["mean_logprob"] - none_score,
                    "margin": None,
                    "choice_scores": None,
                    "extra_notes": None,
                }
            )
    return rows
