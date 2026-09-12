# Reproducibility guide

## Execution modes

- **Smoke check:** demo workload timings, quick sweep grids, and two or three seeds. Use this only to verify installation and output generation.
- **Reported-profile reproduction:** checked-in payload and edge-work measurements with the final seed block 7001–7020.
- **Local benchmark:** fresh liboqs measurements followed by experiments using the resulting CSV. This evaluates a new machine rather than exactly reproducing the manuscript timing input.

## Full reported-profile run

```bash
python -m pip install -e .
pqcho run \
  --profiles data/pqc_profiles_reported.csv \
  --seeds 20 \
  --seed-start 7001 \
  --outdir results/final_7001_7020
```

The command produces:

- `raw/`: seed-level and selected job-level records;
- `tables/`: CSV summaries, paired tests, and LaTeX tables;
- `figures/`: PDF and 300-dpi PNG figures;
- `run_metadata.json`: configuration, environment, profiles, seed status, and scheduler description.

## Seed hygiene

The V3 mechanism was tuned on 6001–6020, checked on 6201–6220, and frozen before the final run on 7001–7020. Any future tuning should use a new development block and reserve another untouched block for final evaluation.

## Cryptographic benchmark

The optional benchmark requires `liboqs-python` and a usable liboqs build:

```bash
python -m pip install -e ".[benchmark]"
pqcho benchmark --out data/pqc_profiles_local.csv --repeats 2000 --warmup 100
```

Record the CPU model, operating system, Python version, liboqs version, compiler/build flags, power mode, and whether the system was otherwise idle. The pipeline records general platform information in `run_metadata.json`, but detailed CPU metadata should also be included with published benchmark results.

## Parameter sensitivity

```bash
python scripts/run_sensitivity.py
```

This preserved standalone analysis writes its outputs to `PQC_HO_PARAMETER_SENSITIVITY`. It intentionally uses the reported workload values embedded in the script.

## Scientific scope

The simulator evaluates scheduling effects under a controlled radio/MEC abstraction. It is not a packet-accurate 3GPP NR-V2X stack, a cryptographic security proof, or a production implementation.

