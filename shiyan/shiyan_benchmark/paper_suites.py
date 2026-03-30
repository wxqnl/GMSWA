from __future__ import annotations


BUILTIN_SUITES = {
    "paper_fast_v1": {
        "suite_name": "paper_fast_v1",
        "description": "Real-data fast iteration suite using official lm-eval task definitions for RULER, BABILong, and LongBench v2.",
        "source_notes": [
            "RULER tasks are provided by the local lm-evaluation-harness task registry and reference the NVIDIA RULER benchmark.",
            "BABILong tasks are provided by the local lm-evaluation-harness task registry and use the RMT-team/babilong-1k-samples dataset.",
            "LongBench v2 tasks are provided by the local lm-evaluation-harness task registry and use the longbench2 task definitions already vendored in this repository.",
        ],
        "runs": [
            {
                "run_name": "ruler_fast",
                "kind": "harness",
                "task_group": "retrieval",
                "tasks": [
                    "niah_single_1",
                    "niah_single_2",
                    "niah_multikey_2",
                    "niah_multiquery",
                    "ruler_vt",
                    "ruler_cwe",
                ],
                "metadata": {
                    "max_seq_lengths": [4096, 8192, 16384],
                    "num_samples": 100,
                },
            },
            {
                "run_name": "babilong_fast",
                "kind": "harness",
                "task_group": "persistent",
                "tasks": [
                    "babilong_qa1",
                    "babilong_qa2",
                    "babilong_qa3",
                    "babilong_qa4",
                    "babilong_qa5",
                ],
                "metadata": {
                    "max_seq_lengths": "0k,4k,16k,32k",
                },
            },
            {
                "run_name": "longbench2_fast",
                "kind": "harness",
                "task_group": "persistent",
                "tasks": [
                    "longbench2_agent_history",
                    "longbench2_dialogue_history",
                    "longbench2_user_guide",
                    "longbench2_many_shot",
                    "longbench2_code",
                    "longbench2_table",
                ],
            },
        ],
    },
    "paper_main_v1": {
        "suite_name": "paper_main_v1",
        "description": "Paper-grade main benchmark suite for GM-SWA, centered on real long-context tasks rather than synthetic local placeholders.",
        "source_notes": [
            "RULER: official NVIDIA benchmark task variants as vendored in lm-evaluation-harness.",
            "BABILong: official task families qa1-qa20 via the RMT-team Hugging Face release used by lm-evaluation-harness.",
            "LongBench v2: realistic multiple-choice long-context benchmark categories via the vendored task definitions.",
            "LongBench selected tasks: optional appendix-style real-world long-context tasks supported by the local task registry.",
        ],
        "runs": [
            {
                "run_name": "ruler_core_stable",
                "kind": "harness",
                "task_group": "retrieval",
                "tasks": [
                    "niah_single_1",
                    "niah_single_2",
                    "niah_single_3",
                    "niah_multikey_1",
                    "niah_multikey_2",
                    "niah_multikey_3",
                    "niah_multiquery",
                    "niah_multivalue",
                    "ruler_vt",
                    "ruler_cwe",
                    "ruler_fwe",
                ],
                "metadata": {
                    "max_seq_lengths": [4096, 8192, 16384, 32768, 65536],
                },
            },
            {
                "run_name": "babilong_longctx_core",
                "kind": "harness",
                "task_group": "persistent",
                "tasks": ["babilong_longctx"],
                "metadata": {
                    "max_seq_lengths": "0k,1k,2k,4k,8k,16k,32k,64k,128k",
                },
            },
            {
                "run_name": "longbench2_core",
                "kind": "harness",
                "task_group": "persistent",
                "tasks": [
                    "longbench2_history",
                    "longbench2_incontext",
                    "longbench2_structured",
                    "longbench2_code",
                    "longbench2_single",
                    "longbench2_multi",
                ],
            },
            {
                "run_name": "longbench_appendix_selected",
                "kind": "harness",
                "task_group": "mixed",
                "tasks": [
                    "longbench_passage_retrieval_en",
                    "longbench_hotpotqa",
                    "longbench_2wikimqa",
                    "longbench_qasper",
                    "longbench_gov_report",
                    "longbench_repobench-p",
                ],
            },
        ],
    },
    "mechanism_real_v1": {
        "suite_name": "mechanism_real_v1",
        "description": "Real-data mechanism suite aligned with the paper's retrieval-vs-persistent framing.",
        "source_notes": [
            "Retrieval-oriented tasks are chosen from RULER and LongBench retrieval-style tasks.",
            "Persistent-oriented tasks are chosen from BABILong and LongBench v2 history/in-context reasoning tasks.",
            "This suite is intended for mechanism analysis and ablations, not just headline benchmark reporting.",
        ],
        "runs": [
            {
                "run_name": "retrieval_real",
                "kind": "harness",
                "task_group": "retrieval",
                "tasks": [
                    "niah_single_1",
                    "niah_single_2",
                    "niah_single_3",
                    "niah_multikey_1",
                    "niah_multikey_2",
                    "niah_multikey_3",
                    "longbench_passage_retrieval_en",
                ],
                "metadata": {
                    "max_seq_lengths": [4096, 8192, 16384, 32768],
                },
            },
            {
                "run_name": "persistent_real",
                "kind": "harness",
                "task_group": "persistent",
                "tasks": [
                    "babilong_qa11",
                    "babilong_qa14",
                    "babilong_qa15",
                    "longbench2_agent_history",
                    "longbench2_dialogue_history",
                    "longbench2_user_guide",
                    "longbench2_many_shot",
                    "longbench2_event_order",
                ],
                "metadata": {
                    "max_seq_lengths": "0k",
                },
            },
        ],
    },
}
