#!/usr/bin/bash
# Auto-eval the bighead ablation (8x128 + full-seq memory) the moment training
# finishes: convert DCP->HF, loss-vs-position, NIAH sweep + recall (8 GPUs free
# after training). Compares against GDN (the recall target).
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python; export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
cd "$FLAME"
LOG=$ROOT/eval_results/eval_v5conv.log
RUN=$FLAME/saves/GMSWA-340M-v5conv-10k
CFG=configs/gated_mem_swa_v5conv_340M.json

echo "[$(date)] waiting for bighead training (step-10000 checkpoint)..." | tee -a "$LOG"
until [ -d "$RUN/checkpoint/step-10000" ]; do sleep 120; done
sleep 30
echo "[$(date)] training done -> converting DCP->HF" | tee -a "$LOG"
$PY -m flame.utils.convert_dcp_to_hf --path "$RUN" --step 10000 \
    --config "$CFG" --tokenizer /home/user01/Minko/models/gla-tokenizer 2>&1 | tail -3 | tee -a "$LOG"
# free disk: drop non-final checkpoints
for b in "$RUN"/checkpoint/step-*; do [ "$(basename "$b")" != "step-10000" ] && rm -rf "$b"; done

[ -f "$RUN/model.safetensors" ] || { echo "[$(date)] CONVERT FAILED (no safetensors)" | tee -a "$LOG"; exit 1; }
echo "[$(date)] loss-vs-position" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 $PY "$ROOT/scripts/analysis/ppl_vs_position.py" GMSWA-v5conv "$RUN" 8192 120 \
    > "$ROOT/eval_results/ppl_GMSWA-v5conv.out" 2>&1 || true
echo "[$(date)] NIAH sweep + recall (8 GPUs)" | tee -a "$LOG"
bash eval_suite.sh GMSWA-v5conv=$RUN 2>&1 | tee -a "$LOG"
# standard short bench too (consistency)
SHORT="wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,sciq,copa"
"$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1 --main_process_port 29589 \
  --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py" --model hf --verbosity ERROR \
  --model_args "pretrained=$RUN,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
  --tasks "$SHORT" --batch_size 16 --output_path "$ROOT/eval_results/GMSWA-340M-v5conv-10k/short" 2>&1 \
  | grep -avE "Generating|examples/s|Downloading|Noises" > "$ROOT/eval_results/GMSWA-340M-v5conv-10k/short.log" || true
echo "[$(date)] V5CONV EVAL DONE" | tee -a "$LOG"
