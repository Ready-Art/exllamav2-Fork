#!/bin/bash
# Stage-2 (warp-parallel P2P rendezvous) validation. The protocol change is unconditional,
# so fusion-OFF now uses the new protocol too. Compare against stage-1 references:
#   /tmp/p0.json (stage-1 greedy ctx2048)  -> parity
#   /tmp/s0.json (stage-1 sampled 79k off) -> stage-2 latency effect
cd ~/exl2work
PY=venv/bin/python
T=exllamav2-Fork/mtp/fuse_agnr_test.py
export NCCL_P2P_LEVEL=PXB EXL2_MTP_DRAFT_GRAPH=1 PYTORCH_CUDA_ALLOC_CONF=backend:native
echo "=== s2 greedy off ==="; $PY $T 2048 2 150 0 0 /tmp/q0.json
echo "=== s2 speed off ===";  $PY $T 79067 2 400 0 1 /tmp/t0.json
echo "=== s2 speed on ===";   $PY $T 79067 2 400 1 1 /tmp/t1.json
echo "=== COMPARE ==="
$PY - <<'EOF'
import json, os
q0=json.load(open('/tmp/q0.json'))
t0=json.load(open('/tmp/t0.json')); t1=json.load(open('/tmp/t1.json'))
if os.path.exists('/tmp/p0.json'):
    p0=json.load(open('/tmp/p0.json'))
    a,b=p0['tokens'],q0['tokens']
    nd=sum(1 for x,y in zip(a,b) if x!=y)
    print("PARITY vs stage1:", "EXACT MATCH" if a==b else f"{nd}/{len(a)} differ")
if os.path.exists('/tmp/s0.json'):
    s0=json.load(open('/tmp/s0.json'))
    print(f"stage1 off: {s0['gen_tps']} t/s verify {s0['warm_ms']['t_verify']} round {sum(s0['warm_ms'].values()):.2f}")
print(f"stage2 off: {t0['gen_tps']} t/s verify {t0['warm_ms']['t_verify']} round {sum(t0['warm_ms'].values()):.2f} acc {t0['warm_acc']}")
print(f"stage2 on:  {t1['gen_tps']} t/s verify {t1['warm_ms']['t_verify']} round {sum(t1['warm_ms'].values()):.2f} acc {t1['warm_acc']}")
if os.path.exists('/tmp/s0.json'):
    print(f"STAGE2 effect (off vs stage1 off): {t0['gen_tps']/s0['gen_tps']:.3f}x")
EOF
echo DONE > ~/exl2work/fuse_s2.done