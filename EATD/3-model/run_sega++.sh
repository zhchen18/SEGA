#!/usr/bin/env bash
cd "$(dirname "$0")/scripts"
PY=${PY:-/DATA/COLA/chenzhuang/miniconda3/envs/cztorch/bin/python}
GPU=${GPU:-0}; SEEDS=${SEEDS:-"200 123 100"}
export CUDA_VISIBLE_DEVICES=$GPU NT=${NT:-8}
export PROXY=1 DIRECTED=1 READOUT=vsum_a LGAT=2
export AUG=llm MIXMODE=both MIXSYN=0 MIXA=0.1 AUGW=1 AUDIOSYN=1
mkdir -p ../results/logs ../checkpoints
i=0
for s in $SEEDS; do
  i=$((i+1))
  SEED=$s SAVEDIR=../checkpoints SAVETAG=segapp $PY train.py 2>&1 \
    | grep -E 'fold[0-9]|BEST' | sed -u 's/ seed=[0-9]*//' | tee ../results/logs/segapp_run${i}.log
  mv -f ../checkpoints/segapp_s${s}.pt ../checkpoints/segapp_run${i}.pt
done
$PY ../scripts/assemble_best.py
echo "saved checkpoints/segapp_best.pt  (eval: python scripts/eval_ckpt.py checkpoints/segapp_best.pt -> 82.24)"
