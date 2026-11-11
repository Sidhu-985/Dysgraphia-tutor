# Experimental training on a broad synthetic learner population

This is an **experimental candidate-generation path**, not a deployment fix by itself.

## Why this exists

The existing environment samples from five hand-authored profiles. Running thousands of episodes against those same five profiles increases episode count but not learner-population diversity. The optional `synthetic_population` sampler creates a reproducible Latin-hypercube population of 1,000+ parameter combinations using the documented minimum/maximum values of the existing five profiles.

The population varies:
- baseline skill
- learning rate
- fatigue accumulation
- frustration sensitivity
- spacing weakness
- stroke instability
- recovery rate

These ranges are still hand-authored. The generated profiles are **not real children, not fitted clinical data, and not evidence of efficacy**. The purpose is robustness testing against a wider range of simulator assumptions.

## Run on Windows PowerShell

Activate the same Python virtual environment used by the backend, then:

```powershell
cd "C:\path\to\DYSgraphia_TUTOR_ALPHABET_DQN_UPGRADE\source\backend"
python -m pip install -r requirements.txt
python ..\dysgraphia_rl\scripts\deployment\train_synthetic_population_candidate.py --episodes 5000 --profiles 1000 --seed 42
```

For a larger experiment:

```powershell
python ..\dysgraphia_rl\scripts\deployment\train_synthetic_population_candidate.py --episodes 10000 --profiles 2000 --seed 43
```

The script writes to:

`source/dysgraphia_rl/checkpoints/deployment_candidates/SYNTHETIC_POPULATION/`

It refuses to overwrite an existing candidate. It does not touch the historical `double_dqn.pt` or promote a candidate to live deployment.

## Qualification

A candidate is marked for manual review only if all of these pass:
1. Existing behavioural gate, unchanged.
2. Held-out low-fatigue populations (five seeds), each below the existing break-rate limit.
3. 50 jittered probes for G1–G3, with zero forbidden actions.
4. Offline evaluation on a separately generated held-out synthetic profile population (at least 500 profiles), using separate evaluation seeds, positive difficulty-matched skill gain, and executed-break fraction below 25%.

A nonzero exit code means the candidate did not qualify. Inspect the JSON report beside the checkpoint for exact failures.

## Important evaluation caveat

The offline evaluator now accepts explicit environment overrides and uses a separately seeded held-out synthetic profile population rather than silently reverting to the five-profile catalog. This first experiment is a robustness screen, not proof that the policy generalizes to real handwriting or children. If the candidate fails, do not relax the gate or copy it into the deployment folder. Investigate the failed action/state pairs and design a separately documented experiment.
