# PQC-HO

Official reproducibility package for:

> **PQC-HO: Workload-Aware Radio–Edge Scheduling for Low-Latency 6G Vehicular Handover**  
> Poushali Sengupta and Mayank Raikwar

PQC-HO is a 1-ms discrete-time simulator for deadline-constrained vehicular handover. It models radio transmission followed by multi-access edge computing (MEC) authentication processing. Each vehicle's post-quantum cryptography (PQC) security tier is assigned externally and remains fixed: the scheduler uses the communication and computation workload created by that tier but never selects or weakens the cryptographic protection.

## Repository contents

This final GitHub export keeps the principal files at the repository root:

| File | Purpose |
|---|---|
| `experiments.py` | Frozen V3 simulator, schedulers, experiment suite, figures, statistical tests, and command-line interface |
| `run_sensitivity.py` | Standalone sensitivity analysis for urgency-window width `W` and tier-reservation fraction `rho` |
| `pqc_profiles_reported.csv` | Reported ML-KEM/ML-DSA payload sizes and measured edge-processing workloads |
| `pqcho_quickstart.ipynb` | Compact demonstration notebook from the structured release |
| `icc_updated_revised_original.ipynb` | Final supplied development notebook, preserved for provenance |
| `icc_original.ipynb` | Earlier supplied notebook, preserved for provenance |
| `REPRODUCIBILITY.md` | Detailed reproduction workflow and seed-hygiene guidance |
| `RESULTS.md` | Expected headline results and interpretation notes |
| `CITATION.cff` | Citation metadata |
| `PQC-HO-v1.0.0.zip` | Complete structured, installable release with source, tests, CI, documentation, and notebooks |

For normal use directly from this repository, follow the commands below. For editable package installation, automated tests, or the clean notebook workflow, extract `PQC-HO-v1.0.0.zip` and follow the README inside that structured release.

## Resulted Figures

 The Result figures are also provided in high-resolution PNG format here.

The PNG versions were generated at **600 dpi** from the corresponding publication-quality PDF figures. The underlying plots, numerical values, labels, and visual layout were not modified; only the output format was changed.

### Final PNG Figures

The following files are used in the manuscript:

| Figure file | Description |
|---|---|
| `fig_main_deadline_violation_final.png` | Main comparison of deadline-violation performance across scheduling methods |
| `fig_ablation_violation_final.png` | Ablation study showing the effect of individual PQC-HO components |
| `fig_density_violation_final.png` | Vehicle-density sensitivity analysis |
| `fig_radio_mec_gain_heatmap_final.png` | Relative PQC-HO deadline-violation reduction across joint radio and MEC capacity conditions |



## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- Matplotlib
- SciPy
- Optional: `liboqs-python` for fresh cryptographic benchmarking

Create an isolated environment and install the simulation dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick software check

Run the shortened experiment suite with explicit demo timings:

```bash
python experiments.py run \
  --demo \
  --quick \
  --seeds 2 \
  --seed-start 9001 \
  --outdir results/smoke
```

Demo timings are placeholders for software validation. **Do not report demo outputs as paper evidence.**

## Reproduce the reported experiment configuration

Run the frozen V3 evaluation using the workload measurements reported in the manuscript and final seeds 7001–7020:

```bash
python experiments.py run \
  --profiles pqc_profiles_reported.csv \
  --seeds 20 \
  --seed-start 7001 \
  --outdir results/final_7001_7020
```

This command runs the main scheduler comparison and the density, packet-loss, radio-capacity, MEC-capacity, joint-bottleneck, workload-mixture, stochastic-loss, and ablation experiments. The complete evaluation takes considerably longer than the smoke check.

Generated outputs include:

- `raw/`: seed-level and selected job-level records;
- `tables/`: CSV summaries, manuscript tables, paired tests, and Holm-adjusted p-values;
- `figures/`: publication-ready PDF and 300-dpi PNG figures;
- `run_metadata.json`: profiles, configuration, platform information, seed status, and scheduler description.

## Benchmark PQC workloads locally

To measure ML-KEM and ML-DSA on another machine, install `liboqs-python` and run:

```bash
python -m pip install liboqs-python
python experiments.py benchmark \
  --out pqc_profiles_local.csv \
  --repeats 2000 \
  --warmup 100
```

Then rerun the experiments with `--profiles pqc_profiles_local.csv`. Cryptographic timings are machine-dependent, so retain the generated CSV and record the CPU model, operating system, Python version, liboqs version, compiler/build configuration, and power mode with any reported results.

## Parameter sensitivity

Run the preserved standalone sensitivity analysis with:

```bash
python run_sensitivity.py
```

It evaluates the urgency-window width `W` and tier-reservation fraction `rho` and writes results to `PQC_HO_PARAMETER_SENSITIVITY`.

## Main model configuration

| Setting | Default |
|---|---:|
| Vehicles | 300 |
| Simulation resolution | 1 ms |
| Arrival window | 300 ms |
| Burst fraction/window | 75% / 60 ms |
| PQC tier mixture | 40% P1 / 40% P2 / 20% P3 |
| Mobility deadline | Uniform 70–140 ms |
| Radio bandwidth | 20 MHz |
| SNR | 12 ± 4 dB |
| Packet loss | 1% |
| MEC reference cores | 4 |
| Authentication CPU share | 10% |
| Urgency-window width, `W` | 22 ms |
| Tier-reservation fraction, `rho` | 0.85 |
| Final seed block | 7001–7020 |

The evaluation includes FIFO, Round Robin, EDF, Proportional Fair, Least Laxity, SRPT, an architecture-matched workload-unaware scheduler, and PQC-HO.

## Expected headline result

Under the nominal frozen configuration, PQC-HO reduces the mean deadline-violation rate from **18.90%** for the architecture-matched workload-unaware scheduler to **17.90%**. The benefit becomes larger when heterogeneous PQC workloads compete under constrained MEC capacity. See `RESULTS.md` for the complete interpretation and the distinction between the nominal result and the separate parameter-sensitivity estimate.

## Scientific scope

This is a reproducible systems-level radio/MEC scheduling abstraction. It is not a packet-accurate 3GPP NR-V2X stack, a new authentication protocol, a cryptographic security proof, or production security software. The scheduler does not change the security tier assigned to any vehicle.

## Citation

Citation metadata is provided in `CITATION.cff`. Add the final conference details and DOI after publication.

## License

The current `LICENSE` grants no reuse rights. The authors should deliberately select an open-source license before public release if external reuse and modification are intended.
