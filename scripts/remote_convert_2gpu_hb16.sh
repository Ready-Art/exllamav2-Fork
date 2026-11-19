#!/bin/bash
# Convert google/gemma-4-31B-it -> exl2 6.0bpw, hb16, on exactly TWO GPUs.
#
# -d 0,1          : devices 0 and 1 only. Device 0 is primary and owns every Hessian
#                    inversion; independent tensors are distributed across both.
# -crs on          : shard calibration rows across both GPUs, one module replica per
#                    device. Without this the measurement pass is single-device and the
#                    second GPU sits idle for the first half of the run.
# -hb 16           : unquantized fp16 head (was 8).
# -b 6.0           : decoder target bitrate, unchanged.
# -l 2048 -r 100   : calibration rows, unchanged.
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.5
export MAX_JOBS=32
export TORCH_CUDA_ARCH_LIST="8.6"
export HF_HOME="$HOME/exl2work/hf"
export HF_HUB_CACHE="$HOME/exl2work/hf/hub"

VENV="$HOME/exl2work/venv"
SRC="$HOME/exl2work/models/gemma-4-31B-it"
OUT="$HOME/exl2work/models/gemma-4-31b-exl2-6.0bpw-hb16-2gpu"

mkdir -p "$HF_HOME"
cd "$HOME/exl2work/exllamav2-fork" || exit 1

echo "=== preflight ==="
git log --oneline -1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
"$VENV/bin/python" - <<'PY'
import torch
n = torch.cuda.device_count()
print("visible devices:", n)
for i in range(n):
    print(f"  cuda:{i}", torch.cuda.get_device_name(i),
          round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1), "GiB")
assert n >= 2, "need at least 2 visible devices"
PY

ls -la "$SRC"

echo "=== convert (2 GPUs, hb16, crs on) ==="
"$VENV/bin/python" convert.py \
  -i "$SRC" \
  -o "$OUT" \
  -l 2048 \
  -r 100 \
  -b 6.0 \
  -hb 16 \
  -d 0,1 \
  -crs on

echo "EXIT=$?"
echo "=== output ==="
ls -la "$OUT"
du -sh "$OUT"
echo "=== CONVERT DONE ==="
