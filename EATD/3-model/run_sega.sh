#!/usr/bin/env bash
cd "$(dirname "$0")/scripts"
PY=${PY:-/DATA/COLA/chenzhuang/miniconda3/envs/cztorch/bin/python}
GPU=${GPU:-0}; SEED=${SEED:-100}
export CUDA_VISIBLE_DEVICES=$GPU NT=${NT:-8}
export PROXY=1 DIRECTED=1 READOUT=vsum_a LGAT=2 AUG=none
mkdir -p ../results/logs ../checkpoints
SEED=$SEED SAVEDIR=../checkpoints SAVETAG=sega $PY train.py 2>&1 \
  | grep -E 'fold[0-9]|BEST' | sed -u 's/ seed=[0-9]*//' | tee ../results/logs/sega_run1.log
mv -f ../checkpoints/sega_s${SEED}.pt ../checkpoints/sega_best.pt
echo "saved checkpoints/sega_best.pt  (eval: python scripts/eval_ckpt.py checkpoints/sega_best.pt -> 81.06)"
