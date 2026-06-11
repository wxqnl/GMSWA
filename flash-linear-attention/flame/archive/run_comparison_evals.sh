#!/usr/bin/bash
# Consolidated GM-SWA vs pure-SWA comparison driver (fixed).
#   1) eval the already-trained+converted GM-SWA checkpoint
#   2) train the pure-SWA baseline on all 8 GPUs (identical config, memory off)
#   3) convert + eval the SWA baseline
# Launch detached:
#   nohup bash run_comparison_evals.sh > .../comparison.console.log 2>&1 &
set -uo pipefail

ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention
FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python
TOKENIZER=/home/user01/Minko/models/gla-tokenizer
STEP=10000
NGPU=${NGPU:-8}
# 8-GPU data-parallel eval: accelerate launches one model replica per GPU and
# lm_eval shards the request set across ranks (~NGPU x faster than 1 GPU).
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes "$NGPU" --num_machines 1
     --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")
SHORT_TASKS=${SHORT_TASKS:-"wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,sciq,copa"}
LONG_TASKS=${LONG_TASKS:-"niah_single_1,niah_single_2"}
export PYTHONPATH="$FLA:${PYTHONPATH:-}"

GMSWA_RUN=$FLAME/saves/GMSWA-340M-v2-10k;  GMSWA_EVAL=$ROOT/eval_results/GMSWA-340M-v2-10k
SWA_RUN=$FLAME/saves/SWA-340M-v2-10k;      SWA_EVAL=$ROOT/eval_results/SWA-340M-v2-10k
GMSWA_CONFIG=$FLAME/configs/gated_mem_swa_340M.json
SWA_CONFIG=$FLAME/configs/swa_baseline_340M.json
mkdir -p "$GMSWA_EVAL" "$SWA_EVAL" "$SWA_RUN"

have_hf_weights () {  # $1 = dir
  local d="$1"
  [[ -f "$d/model.safetensors" || -f "$d/model.safetensors.index.json" || -f "$d/pytorch_model.bin" ]] \
    || ls "$d"/model-*.safetensors >/dev/null 2>&1
}

convert () {  # $1 RUN  $2 CONFIG  $3 EVAL_DIR
  echo "[$(date)] convert $1 step-$STEP -> HF"
  ( cd "$FLAME" && "$PY" -m flame.utils.convert_dcp_to_hf \
      --path "$1" --step "$STEP" --config "$2" --tokenizer "$TOKENIZER" ) 2>&1 | tee "$3/convert.log"
}

run_eval () {  # $1 RUN  $2 EVAL_DIR  $3 label   (8-GPU data-parallel)
  echo "[$(date)] ===== EVAL ($3) short-context on ${NGPU} GPUs ====="
  # max_length=2048 prevents wikitext rolling-ppl from issuing 32768-token forwards
  # (config max_position_embeddings) that blow up the lm_head GEMM. Fixed batch avoids
  # the auto-batch probe picking a pathologically large matmul.
  "${ACC[@]}" --model hf \
    --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
    --tasks "$SHORT_TASKS" --batch_size 16 \
    --output_path "$2/short" 2>&1 | tee "$2/short.log"
  echo "[$(date)] ===== EVAL ($3) long-context on ${NGPU} GPUs (non-fatal) ====="
  "${ACC[@]}" --model hf \
    --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
    --tasks "$LONG_TASKS" --batch_size 1 \
    --metadata '{"max_seq_lengths":[2048,4096,8192]}' \
    --output_path "$2/long" 2>&1 | tee "$2/long.log" || echo "[$(date)] ($3) long probe failed (non-fatal)."
}

# ---------- 1) GM-SWA eval (checkpoint already trained + converted) ----------
if ! have_hf_weights "$GMSWA_RUN"; then convert "$GMSWA_RUN" "$GMSWA_CONFIG" "$GMSWA_EVAL"; fi
if have_hf_weights "$GMSWA_RUN"; then
  run_eval "$GMSWA_RUN" "$GMSWA_EVAL" "GM-SWA"
else
  echo "[$(date)] WARNING: GM-SWA HF weights missing; skipping GM-SWA eval."
fi

# ---------- 2) SWA baseline training (ALL 8 GPUs; do NOT leak CUDA_VISIBLE_DEVICES) ----------
while [[ -n "$(pgrep -f 'flame.train')" ]]; do echo "[$(date)] waiting for GPUs..."; sleep 30; done
echo "[$(date)] starting SWA baseline training on 8 GPUs ..."
env -u CUDA_VISIBLE_DEVICES bash "$FLAME/run_swa_340M_10k.sh" 2>&1 | tee "$SWA_RUN/train.console.log"
echo "[$(date)] SWA training returned."

# ---------- 3) SWA convert + 4) SWA eval ----------
if [[ ! -d "$SWA_RUN/checkpoint/step-$STEP" ]]; then
  echo "[$(date)] ERROR: $SWA_RUN/checkpoint/step-$STEP missing — SWA training did not finish. Aborting."
  exit 1
fi
sleep 30
convert "$SWA_RUN" "$SWA_CONFIG" "$SWA_EVAL"
if have_hf_weights "$SWA_RUN"; then
  run_eval "$SWA_RUN" "$SWA_EVAL" "SWA"
else
  echo "[$(date)] ERROR: SWA HF weights missing after convert."; exit 1
fi

echo "[$(date)] ALL DONE."
echo "  GM-SWA results: $GMSWA_EVAL/short"
echo "  SWA   results: $SWA_EVAL/short"
