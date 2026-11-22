#!/bin/bash
# Drive the fused-AGNR validation: one fresh subprocess per config (second pg init
# in-process crashes). Parity = short greedy off/on; speed = long sampled off/on.
cd ~/exl2work
PY=venv/bin/python
T=exllamav2-Fork/mtp/fuse_agnr_test.py
export NCCL_P2P_LEVEL=PXB EXL2_MTP_DRAFT_GRAPH=1 PYTORCH_CUDA_ALLOC_CONF=backend:native
echo "=== parity off ==="; $PY $T 2048 2 150 0 0 /tmp/p0.json
echo "=== parity on ===";  $PY $T 2048 2 150 1 0 /tmp/p1.json
echo "=== speed off ===";  $PY $T 79067 2 400 0 1 /tmp/s0.json
echo "=== speed on ===";   $PY $T 79067 2 400 1 1 /tmp/s1.json
echo "=== COMPARE ==="
$PY - <<'EOF'
import json
p0=json.load(open('/tmp/p0.json')); p1=json.load(open('/tmp/p1.json'))
s0=json.load(open('/tmp/s0.json')); s1=json.load(open('/tmp/s1.json'))
t0,t1=p0['tokens'],p1['tokens']
ndiff=sum(1 for a,b in zip(t0,t1) if a!=b)
print(f"PARITY: {'EXACT MATCH' if t0==t1 else f'{ndiff}/{len(t0)} tokens differ'}")
print(f"  off tail {t0[-10:]}")
print(f"  on  tail {t1[-10:]}")
print(f"SPEED off: {s0['gen_tps']} t/s  verify {s0['warm_ms']['t_verify']} ms  round {sum(s0['warm_ms'].values()):.2f} ms  acc {s0['warm_acc']}")
print(f"SPEED on:  {s1['gen_tps']} t/s  verify {s1['warm_ms']['t_verify']} ms  round {sum(s1['warm_ms'].values()):.2f} ms  acc {s1['warm_acc']}")
print(f"SPEEDUP fused/off: {s1['gen_tps']/s0['gen_tps']:.3f}x")
EOF
echo DONE > ~/exl2work/fuse_test.done