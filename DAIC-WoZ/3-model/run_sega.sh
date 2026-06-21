#!/usr/bin/env bash
cd "$(dirname "$0")/scripts"
[ -z "$PY" ] && source /DATA/COLA/chenzhuang/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate cztorch 2>/dev/null || true
PY=${PY:-python}; GPU=${GPU:-0}; EP=${EP:-200}; SEEDS=${SEEDS:-"3 1 2"}
export CUDA_VISIBLE_DEVICES=$GPU; export NT=${NT:-8}
M="norm=0 fusion=boost_graph mm_mods=question,audio,video use_gender=1 use_emotion=1 mixup=1 malpha=0.2 ls=0.05"
mkdir -p ../results/logs
i=0
for s in $SEEDS; do
  i=$((i+1))
  $PY train.py $M epochs=$EP seed=$s quiet=1 2>&1 | sed -u 's/ seed=[0-9]*//' \
    | tee ../results/logs/sega_run${i}.log | grep -o 'BEST.*'
done
$PY - <<'PY'
import re,glob,numpy as np
v=[float(re.findall(r'BEST macro=([\d.]+)',open(f).read())[-1]) for f in sorted(glob.glob("../results/logs/sega_run*.log")) if re.findall(r'BEST macro=([\d.]+)',open(f).read())]
v=np.array(v); print(f"SEGA  {len(v)} run(s): mean {v.mean():.2f}  best {v.max():.1f}")
PY
