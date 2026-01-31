#!/usr/bin/env bash
set -euo pipefail

# ====== Config (edit as needed) ======
BASE_ROOT="${BASE_ROOT:-/home/dataset/data/h30081741/VLM4D-video}"
MODEL_PATH="${MODEL_PATH:-/abs/path/to/models/dsr_baseline}"
REAL_JSON="${REAL_JSON:-scripts/real_mc.json}"
SYN_JSON="${SYN_JSON:-scripts/synthetic_mc.json}"

OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_REAL="${OUT_REAL:-${OUT_ROOT}/real_mc_cot}"
OUT_SYN="${OUT_SYN:-${OUT_ROOT}/synthetic_mc_cot}"
OUT_NAME="${OUT_NAME:-dsr_vlm4d.json}"
PROMPT_MODE="${PROMPT_MODE:-cot}"

# Optional: limit samples for quick test (0 = full)
LIMIT_REAL="${LIMIT_REAL:-0}"
LIMIT_SYN="${LIMIT_SYN:-0}"

# GPU count (override with NUM_GPUS=)
if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
  else
    NUM_GPUS=1
  fi
fi

# GPU IDs to use (override with GPU_IDS="0,1,2,3")
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="0,1,2,3"
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"

VLM4D_DIR="${VLM4D_DIR:-VLM4D}"

# ====== Helpers ======
get_total() {
  python - "$1" <<'PY'
import json, sys
p=sys.argv[1]
with open(p,"r") as f:
    data=json.load(f)
print(len(data))
PY
}

run_dataset() {
  local name="$1"
  local json_path="$2"
  local out_dir="$3"
  local limit="$4"

  if [[ ! -f "$json_path" ]]; then
    echo "[ERROR] Missing JSON: $json_path"
    exit 1
  fi

  local total
  total="$(get_total "$json_path")"
  if [[ "$limit" -gt 0 && "$limit" -lt "$total" ]]; then
    total="$limit"
  fi

  mkdir -p "$out_dir"
  local chunk=$(( (total + NUM_GPUS - 1) / NUM_GPUS ))
  echo "[INFO] $name: total=$total, num_gpus=$NUM_GPUS, chunk=$chunk"

  pids=()
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    local start=$(( gpu * chunk ))
    if [[ "$start" -ge "$total" ]]; then
      break
    fi
    local sublimit=$chunk
    if [[ $(( start + sublimit )) -gt "$total" ]]; then
      sublimit=$(( total - start ))
    fi
    local part_name="${OUT_NAME%.json}_part${gpu}.json"
    local gpu_id="${GPU_ARRAY[$gpu]}"
    echo "[INFO] $name: GPU${gpu_id} start=$start limit=$sublimit -> $part_name"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      python scripts/run_vlm4d_dsr.py \
        --input_json "$json_path" \
        --base_root "$BASE_ROOT" \
        --model_path "$MODEL_PATH" \
        --output_dir "$out_dir" \
        --output_name "$part_name" \
        --prompt_mode "$PROMPT_MODE" \
        --device_map cuda:0 \
        --start "$start" \
        --limit "$sublimit" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  # Merge parts
  python - <<PY
import json, glob
parts=sorted(glob.glob("${out_dir}/${OUT_NAME%.json}_part*.json"))
out=[]
for p in parts:
    out += json.load(open(p))
json.dump(out, open("${out_dir}/${OUT_NAME}","w"), ensure_ascii=False, indent=2)
print("merged:", len(out), "->", "${out_dir}/${OUT_NAME}")
PY
}

# ====== Sanity checks ======
if [[ ! -d "$BASE_ROOT" ]]; then
  echo "[ERROR] BASE_ROOT not found: $BASE_ROOT"
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH not found: $MODEL_PATH"
  exit 1
fi

# ====== Run inference ======
run_dataset "real" "$REAL_JSON" "$OUT_REAL" "$LIMIT_REAL"
run_dataset "synthetic" "$SYN_JSON" "$OUT_SYN" "$LIMIT_SYN"

# ====== Evaluation ======
if [[ ! -d "$VLM4D_DIR" ]]; then
  echo "[ERROR] VLM4D dir not found: $VLM4D_DIR"
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[WARN] OPENAI_API_KEY not set. Skipping acc_evaluation/acc_final_statistics."
  exit 0
fi

pushd "$VLM4D_DIR" >/dev/null
python acc_evaluation.py --output_dir "../${OUT_REAL}"
python acc_evaluation.py --output_dir "../${OUT_SYN}"
python acc_final_statistics.py
popd >/dev/null

echo "[OK] Done."
