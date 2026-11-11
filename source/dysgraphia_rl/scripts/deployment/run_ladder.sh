#!/usr/bin/env bash
# Runs the pre-declared candidate ladder (docs/DQN_DEPLOYMENT_CANDIDATE_PROTOCOL.md):
# a rung runs only if no seed of the previous rung qualified. Run from backend/.
set -u
CK=../dysgraphia_rl/checkpoints/deployment_candidates
qualified() { python3 -c "
import json,glob,sys
fs=sorted(glob.glob('$CK/$1/seed*.json'))
sys.exit(0 if len(fs)==5 and any(json.load(open(f))['qualifies'] for f in fs) else 1)"; }
done5() { [ "$(ls $CK/$1/seed*.json 2>/dev/null | wc -l)" -ge 5 ]; }
until done5 C1; do sleep 15; done
for R in C2 C3; do
  prev=$([ $R = C2 ] && echo C1 || echo C2)
  if qualified $prev; then echo "rung $prev produced a qualifying candidate; $R not run"; exit 0; fi
  echo "running $R"
  OMP_NUM_THREADS=1 python3 ../dysgraphia_rl/scripts/deployment/train_candidates.py $R 0 2 4 &
  OMP_NUM_THREADS=1 python3 ../dysgraphia_rl/scripts/deployment/train_candidates.py $R 1 3 &
  wait
done
qualified C3 && echo "C3 produced a qualifying candidate" || echo "no candidate qualified on any rung"
