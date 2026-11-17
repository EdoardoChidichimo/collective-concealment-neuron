#!/bin/bash
set -e
cd "$(dirname "$0")"
PHASE=$1; shift; GPUS=(${@:-0}); NG=${#GPUS[@]}
PY=${PY:-.venv/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ABL=${ABL:-$QWEN-ablated}
export CN_MODEL=$QWEN CN_JUDGE=$JUDGE
G="--temptation 4 --games audit,trial"
G2="--temptation 4 --games food,maint,news"
C=conditions
$PY conditions.py

run_all() {
  local dir=$1 pids=(); shift
  mkdir -p results/$dir
  for k in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} $PY run_games.py --shard $k/$NG --out results/$dir/all.s$k.jsonl "$@" &
    pids+=($!)
  done
  wait "${pids[@]}"
}

case $PHASE in
calib)
  for B in 512 1024; do
    M="--model $QWEN $G --max-tokens $B --n-agents 5 --episodes 20 --seed0 19000"
    run_all calib-$B $M --conditions $C/calib-team.json
    run_all calib-$B $M --conditions $C/calib-free.json
    $PY budget.py $B && break
  done ;;
main)
  B=$($PY -c "import json; print(json.load(open('results/budget.json'))['budget'])")
  M="--model $QWEN --max-tokens $B --seed0 20000"
  run_all focal $M $G --n-agents 5 --episodes 200 --conditions $C/focal-team.json
  run_all focal $M $G --n-agents 1 --episodes 200 --conditions $C/focal-solo.json
  run_all free $M $G --n-agents 5 --episodes 200 --conditions $C/free.json
  run_all dose $M $G --n-agents 5 --episodes 100 --conditions $C/dose-team.json
  run_all dose $M $G --n-agents 1 --episodes 100 --conditions $C/dose-solo.json
  run_all rand $M $G --n-agents 5 --episodes 10 --conditions $C/rand-team.json
  run_all rand $M $G --n-agents 1 --episodes 10 --conditions $C/rand-solo.json
  run_all free-rand $M $G --n-agents 5 --episodes 10 --conditions $C/free-rand.json
  run_all gen $M $G2 --n-agents 5 --episodes 200 --conditions $C/focal-team.json
  run_all gen $M $G2 --n-agents 1 --episodes 200 --conditions $C/focal-solo.json
  run_all gen $M $G2 --n-agents 5 --episodes 200 --conditions $C/free.json
  [ -d "$ABL" ] || $PY ablate.py "$QWEN" "$ABL" 16 8168
  export CN_MODEL=$ABL
  MA="--model $ABL --max-tokens $B --seed0 20000"
  run_all abl $MA $G --n-agents 5 --episodes 200 --conditions $C/abl-team.json
  run_all abl $MA $G --n-agents 1 --episodes 200 --conditions $C/abl-solo.json
  run_all abl $MA $G --n-agents 5 --episodes 200 --conditions $C/abl-free.json ;;
post)
  mkdir -p results/quality
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY act_range.py --episodes results/focal/*.jsonl results/free/*.jsonl --out results/act_range
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY judge.py quality --focal-only \
    --episodes results/{focal,free,dose,rand,free-rand,abl,gen}/*.jsonl --out results/quality/judge.jsonl
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY judge.py nll --episodes results/{focal,free,gen}/*.jsonl --out results/quality/nll.jsonl ;;
analyze)
  $PY analyze.py
  $PY quality.py ;;
esac
