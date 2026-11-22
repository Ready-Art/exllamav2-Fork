#!/bin/bash
# Stage-3 (row-parallel decode_repl, EXL2_TP_ROW_REPL) validation: parity + speed, off vs on.
# Fresh subprocess per config. FUSE_AGNR held OFF to isolate ROW_REPL.
cd ~/exl2work
PY=venv/bin/python
T=exllamav2-Fork/mtp/fuse_agnr_test.py
export NCCL_P2P_LEVEL=PXB EXL2_MTP_DRAFT_GRAPH=1 PYTORCH_CUDA_ALLOC_CONF=backend:native
export FLAG_NAME=EXL2_TP_ROW_REPL EXL2_TP_FUSE_AGNR=0
echo "=== s3 parity off ==="; $PY $T 2048 2 150 0 0 /tmp/r0.json
echo "=== s3 parity on ===";  $PY $T 2048 2 150 1 0 /tmp/r1.json
echo "=== s3 speed off ===";  $PY $T 79067 2 400 0 1 /tmp/u0.json
echo "=== s3 speed on ===";   $PY $T 79067 2 400 1 1 /tmp/u1.json
echo "=== COMPARE ==="
$PY - <<'EOF'
import json
r0=json.load(open('/tmp/r0.json')); r1=json.load(open('/tmp/r1.json'))
u0=json.load(open('/tmp/u0.json')); u1=json.load(open('/tmp/u1.json'))
a,b=r0['tokens'],r1['tokens']
nd=sum(1 for x,y in zip(a,b) if x!=y)
print("PARITY off-vs-on:", "EXACT MATCH" if a==b else f"{nd}/{len(a)} differ")
print("  off tail", a[-8:]); print("  on  tail", b[-8:])
print(f"SPEED off: {u0['gen_tps']} t/s verify {u0['warm_ms']['t_verify']} round {sum(u0['warm_ms'].values()):.2f} acc {u0['warm_acc']}")
print(f"SPEED on:  {u1['gen_tps']} t/s verify {u1['warm_ms']['t_verify']} round {sum(u1['warm_ms'].values()):.2f} acc {u1['warm_acc']}")
print(f"STAGE3 speedup on/off: {u1['gen_tps']/u0['gen_tps']:.3f}x")
print("on warm_ms:", u1['warm_ms'])
EOF
echo DONE > ~/exl2work/fuse_s3.done