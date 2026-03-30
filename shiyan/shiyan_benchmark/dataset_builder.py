from __future__ import annotations

import json
from pathlib import Path

from .paper_suites import BUILTIN_SUITES


def _repeat_filler(topic: str, turns: int = 24) -> str:
    return " ".join(
        f"Turn {idx}: the discussion continues about {topic} with routine filler."
        for idx in range(1, turns + 1)
    )


def _with_span(context_prefix: str, evidence: str, context_suffix: str) -> tuple[str, dict[str, int | str]]:
    context = f"{context_prefix}{evidence}{context_suffix}"
    start = len(context_prefix)
    end = start + len(evidence)
    return context, {"start_char": start, "end_char": end, "text": evidence}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_demo_datasets(output_dir: str | Path) -> dict[str, Path]:
    output_root = Path(output_dir)
    processed = output_root / "data" / "processed"

    filler_retrieval = _repeat_filler("archived notes", turns=36)
    filler_persistent = _repeat_filler("travel planning", turns=30)

    retrieval_1_ctx, retrieval_1_span = _with_span(
        "Meeting archive begins. ",
        "The access key is YX-2048. ",
        filler_retrieval + " Final prompt: Which access key should the assistant return?",
    )
    retrieval_2_ctx, retrieval_2_span = _with_span(
        "Notebook header. ",
        "The secure room number is 7315. ",
        filler_retrieval + " Final prompt: Which room number is correct?",
    )
    persistent_1_ctx, persistent_1_span = _with_span(
        "Travel assistant system prompt. ",
        "The user has a seafood allergy, a budget under 150 dollars, and avoids air travel. ",
        filler_persistent + " Final prompt: Pick the better travel plan.",
    )
    persistent_2_ctx, persistent_2_span = _with_span(
        "Story state tracker. ",
        "Mina already used the silver key and the north gate is now unlocked. ",
        filler_persistent + " Final prompt: Which continuation is consistent with the story state?",
    )

    exp1_rows = [
        {
            "sample_id": "retrieval_0001",
            "task_name": "needle_retrieval",
            "task_group": "retrieval",
            "context": retrieval_1_ctx,
            "choices": [
                {"label": "A", "text": " The key is YX-2048."},
                {"label": "B", "text": " The key is YX-2408."},
            ],
            "label": "A",
            "evidence_span": retrieval_1_span,
        },
        {
            "sample_id": "retrieval_0002",
            "task_name": "phonebook_retrieval",
            "task_group": "retrieval",
            "context": retrieval_2_ctx,
            "choices": [
                {"label": "A", "text": " The room number is 7135."},
                {"label": "B", "text": " The room number is 7315."},
            ],
            "label": "B",
            "evidence_span": retrieval_2_span,
        },
        {
            "sample_id": "persistent_0001",
            "task_name": "rule_consistency",
            "task_group": "persistent",
            "context": persistent_1_ctx,
            "choices": [
                {"label": "A", "text": " A train itinerary under 150 dollars fits the user's needs."},
                {"label": "B", "text": " A flight with a seafood dinner is ideal for this user."},
            ],
            "label": "A",
            "persistent_factors": [
                "budget under 150",
                "no seafood",
                "avoid air travel",
            ],
            "evidence_span": persistent_1_span,
        },
        {
            "sample_id": "persistent_0002",
            "task_name": "state_tracking",
            "task_group": "persistent",
            "context": persistent_2_ctx,
            "choices": [
                {"label": "A", "text": " Mina opens the north gate without searching for another key."},
                {"label": "B", "text": " Mina still cannot pass because the north gate remains locked."},
            ],
            "label": "A",
            "persistent_factors": [
                "silver key already used",
                "north gate unlocked",
            ],
            "evidence_span": persistent_2_span,
        },
    ]

    exp2_rows = [
        {
            "sample_id": "persistent_0021",
            "task_group": "persistent",
            "task_name": "constraint_continuation",
            "context_original": persistent_1_ctx,
            "context_summary": (
                "Travel assistant system prompt. Constraints: seafood allergy, budget under 150 dollars, avoid air travel. "
                + filler_persistent
                + " Final prompt: Pick the best next sentence."
            ),
            "context_memory": (
                "Travel assistant system prompt. State: seafood_allowed=false; budget=low; air_travel=false. "
                + filler_persistent
                + " Final prompt: Pick the best next sentence."
            ),
            "context_none": (
                "Travel assistant system prompt. "
                + filler_persistent
                + " Final prompt: Pick the best next sentence."
            ),
            "target_continuation": " A train option under 150 dollars would fit the user's needs.",
        },
        {
            "sample_id": "persistent_0022",
            "task_group": "persistent",
            "task_name": "story_state_recovery",
            "context_original": persistent_2_ctx,
            "context_summary": (
                "Story state tracker. Summary: Mina unlocked the north gate using the silver key. "
                + filler_persistent
                + " Final prompt: Continue the story consistently."
            ),
            "context_memory": (
                "Story state tracker. State: north_gate=unlocked; silver_key=consumed. "
                + filler_persistent
                + " Final prompt: Continue the story consistently."
            ),
            "context_none": (
                "Story state tracker. "
                + filler_persistent
                + " Final prompt: Continue the story consistently."
            ),
            "target_continuation": " Mina walks through the north gate without needing another key.",
        },
    ]

    retrieval_only = [row for row in exp1_rows if row["task_group"] == "retrieval"]
    persistent_only = [row for row in exp1_rows if row["task_group"] == "persistent"]

    paths = {
        "task_decomposition_all": processed / "task_decomposition_all.jsonl",
        "task_decomposition_retrieval": processed / "task_decomposition_retrieval.jsonl",
        "task_decomposition_persistent": processed / "task_decomposition_persistent.jsonl",
        "minimal_recovery": processed / "minimal_recovery.jsonl",
    }

    _write_jsonl(paths["task_decomposition_all"], exp1_rows)
    _write_jsonl(paths["task_decomposition_retrieval"], retrieval_only)
    _write_jsonl(paths["task_decomposition_persistent"], persistent_only)
    _write_jsonl(paths["minimal_recovery"], exp2_rows)
    return paths


def build_real_suite_manifests(output_dir: str | Path) -> dict[str, Path]:
    output_root = Path(output_dir)
    split_dir = output_root / "data" / "splits"
    paths: dict[str, Path] = {}
    for suite_name, suite in BUILTIN_SUITES.items():
        path = split_dir / f"{suite_name}.json"
        _write_json(path, suite)
        paths[suite_name] = path
    return paths
