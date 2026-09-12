# PQC-HO

Reproducibility package for **“PQC-HO: Workload-Aware Radio–Edge Scheduling for Low-Latency 6G Vehicular Handover.”**

PQC-HO is a 1-ms discrete-time simulator for deadline-constrained vehicular handover. It models radio transmission followed by MEC authentication processing, while keeping each vehicle's externally assigned post-quantum security tier fixed. The scheduler uses the communication and computation workload created by that tier; it never selects or downgrades cryptography.

## What is included

```text
PQC-HO/
├── src/pqcho/                 # Reusable simulator, schedulers, CLI, plots, statistics
├── scripts/                   # Standalone parameter-sensitivity analysis
├── data/                      # Reported ML-KEM/ML-DSA workload measurements
├── notebooks/                # Clean quickstart plus archived source notebooks
├── tests/                    # Fast deterministic unit tests
├── docs/                     # Reproduction and result notes
├── results/                  # Generated outputs (ignored except .gitkeep)
├── pyproject.toml            # Installable Python package
├── environment.yml           # Conda environment
└── Makefile                  # Common commands
```

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Run a small pipeline check using explicit demo timings:

```bash
pqcho run --demo --quick --seeds 2 --seed-start 9001 --outdir results/smoke
```

Demo outputs are only software checks and must not be reported as paper evidence.

## Reproduce the reported experiment configuration

The repository includes the measured workload values reported in the manuscript:

```bash
pqcho run \
  --profiles data/pqc_profiles_reported.csv \
  --seeds 20 \
  --seed-start 7001 \
  --outdir results/final_7001_7020
```

This runs the full baseline comparison and all parameter sweeps. It can take substantially longer than the smoke test. Generated tables, figures, raw seed-level data, and run metadata are written below the selected output directory.

To benchmark ML-KEM and ML-DSA on a new machine, install `liboqs-python` and run:

```bash
pqcho benchmark --out data/pqc_profiles_local.csv --repeats 2000 --warmup 100
```

Then pass the new CSV to `pqcho run`. Cryptographic timing is machine-dependent, so keep the generated CSV and `run_metadata.json` with any reported results.

## Main model defaults

| Setting | Default |
|---|---:|
| Vehicles | 300 |
| Arrival window | 300 ms |
| Burst fraction/window | 75% / 60 ms |
| Mobility deadline | Uniform 70–140 ms |
| Radio bandwidth | 20 MHz |
| SNR | 12 ± 4 dB |
| Packet loss | 1% |
| MEC reference cores | 4 |
| Authentication CPU share | 10% |
| Urgency-window width, `W` | 22 ms |
| Tier-reservation fraction, `rho` | 0.85 |
| Final seed block | 7001–7020 |

The model is a reproducible systems-level abstraction, not a full 3GPP NR-V2X implementation.

## Methods and outputs

The evaluation includes FIFO, Round Robin, EDF, Proportional Fair, Least Laxity, SRPT, an architecture-matched workload-unaware scheduler, and PQC-HO. The pipeline reports latency, deadline violation, tardiness, hard-failure, per-tier reliability, fairness, scheduler runtime, paired Wilcoxon tests, and Holm-adjusted p-values.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete workflow and [docs/RESULTS.md](docs/RESULTS.md) for the expected headline results.

## Citation

Use the metadata in [CITATION.cff](CITATION.cff). Add the paper DOI and final publication venue after acceptance.

## License

No reuse license is granted yet; see [LICENSE](LICENSE). Before making the repository public, the authors should deliberately choose an open-source license if reuse is intended.

