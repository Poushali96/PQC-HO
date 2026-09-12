# ============================================================
# STANDALONE PQC-HO PARAMETER SENSITIVITY
# W = urgency-window width
# rho = tier-reservation fraction
#
# Completely independent cell.
# Does NOT require functions from previous notebook cells.
# ============================================================

import math
import copy
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. PQC PROFILES
#    Same P1/P2/P3 values used in the manuscript
# ============================================================

@dataclass(frozen=True)
class PQCProfile:
    profile: str
    kem: str
    signature: str
    kem_public_key_bytes: int
    kem_ciphertext_bytes: int
    sig_public_key_bytes: int
    signature_bytes: int
    edge_crypto_ms: float
    protocol_overhead_bytes: int = 600

    @property
    def payload_bytes(self):
        return int(
            self.kem_public_key_bytes
            + self.kem_ciphertext_bytes
            + self.sig_public_key_bytes
            + self.signature_bytes
            + self.protocol_overhead_bytes
        )


# Measured profiles used in the current study
profiles = {
    "P1": PQCProfile(
        profile="P1",
        kem="ML-KEM-512",
        signature="ML-DSA-44",
        kem_public_key_bytes=800,
        kem_ciphertext_bytes=768,
        sig_public_key_bytes=1312,
        signature_bytes=2420,
        edge_crypto_ms=0.0896,
    ),

    "P2": PQCProfile(
        profile="P2",
        kem="ML-KEM-768",
        signature="ML-DSA-65",
        kem_public_key_bytes=1184,
        kem_ciphertext_bytes=1088,
        sig_public_key_bytes=1952,
        signature_bytes=3309,
        edge_crypto_ms=0.1007,
    ),

    "P3": PQCProfile(
        profile="P3",
        kem="ML-KEM-1024",
        signature="ML-DSA-87",
        kem_public_key_bytes=1568,
        kem_ciphertext_bytes=1568,
        sig_public_key_bytes=2592,
        signature_bytes=4627,
        edge_crypto_ms=0.1337,
    ),
}

PROFILE_ORDER = ["P1", "P2", "P3"]


print("PQC profiles:")
for p, x in profiles.items():
    print(
        p,
        f"payload={x.payload_bytes} bytes,",
        f"edge={x.edge_crypto_ms:.4f} ms"
    )


# ============================================================
# 2. SIMULATION CONFIGURATION
# ============================================================

@dataclass
class SimConfig:

    # Time
    dt_ms: float = 1.0
    fixed_signaling_ms: float = 4.0
    max_simulation_ms: float = 6000.0

    # Vehicles / arrivals
    n_vehicles: int = 300
    arrival_window_ms: float = 300.0
    burst_window_ms: float = 60.0
    storm_fraction: float = 0.75
    arrival_mode: str = "storm"

    # PQC mixture
    profile_p1: float = 0.40
    profile_p2: float = 0.40
    profile_p3: float = 0.20

    # Deadlines
    deadline_min_ms: float = 70.0
    deadline_max_ms: float = 140.0
    hard_failure_ms: float = 200.0

    # Parameters tested here
    urgency_window_ms: float = 22.0
    tier_fairness_fraction: float = 0.85

    # Radio
    total_bandwidth_mhz: float = 20.0
    mean_snr_db: float = 12.0
    snr_std_db: float = 4.0
    snr_random_walk_std_db: float = 0.12
    min_snr_db: float = -5.0
    max_snr_db: float = 30.0

    packet_loss: float = 0.01
    mtu_bytes: int = 1500
    per_packet_overhead_bytes: int = 64

    # MEC
    mec_reference_cores: float = 4.0
    crypto_cpu_share: float = 0.10

    # Small workload variation
    payload_noise_sigma: float = 0.02
    compute_noise_sigma: float = 0.05

    # Admission
    admission_capacity_fraction: float = 1.0
    scheduler_epsilon_ms: float = 1e-6


# ============================================================
# 3. JOB MODEL
# ============================================================

@dataclass
class Job:
    job_id: int
    profile: str

    arrival_ms: float
    deadline_ms: float
    hard_deadline_ms: float

    payload_bytes: float

    total_wire_bits: float
    remaining_wire_bits: float

    compute_work_ms: float
    remaining_compute_ms: float

    snr_db: float
    packet_loss: float

    phase: str = "radio"

    tx_done_ms: Optional[float] = None
    complete_ms: Optional[float] = None

    served_bits: float = 0.0
    served_compute_ms: float = 0.0

    @property
    def absolute_deadline_ms(self):
        return self.arrival_ms + self.deadline_ms

    @property
    def absolute_hard_deadline_ms(self):
        return self.arrival_ms + self.hard_deadline_ms

    def clone(self):
        return copy.deepcopy(self)


# ============================================================
# 4. BASIC HELPERS
# ============================================================

def spectral_efficiency(snr_db):
    snr_linear = 10.0 ** (snr_db / 10.0)
    return math.log2(1.0 + snr_linear)


def wire_bits(payload_bytes, cfg):
    n_packets = max(
        1,
        math.ceil(payload_bytes / cfg.mtu_bytes)
    )

    total_bytes = (
        payload_bytes
        + n_packets * cfg.per_packet_overhead_bytes
    )

    return float(total_bytes * 8.0)


def profile_probabilities(cfg):

    p = np.array([
        cfg.profile_p1,
        cfg.profile_p2,
        cfg.profile_p3
    ])

    return p / p.sum()


# ============================================================
# 5. GENERATE NOMINAL HANDOVER-STORM JOBS
# ============================================================

def generate_jobs(cfg, profiles, seed):

    rng = np.random.default_rng(seed)

    n = cfg.n_vehicles

    # ---------------- Arrival process ----------------

    n_storm = int(round(n * cfg.storm_fraction))

    center = cfg.arrival_window_ms / 2

    storm = rng.uniform(
        center - cfg.burst_window_ms / 2,
        center + cfg.burst_window_ms / 2,
        size=n_storm,
    )

    normal = rng.uniform(
        0,
        cfg.arrival_window_ms,
        size=n - n_storm,
    )

    arrivals = np.concatenate([storm, normal])

    rng.shuffle(arrivals)

    # ---------------- Security profiles ----------------

    tier_ids = rng.choice(
        PROFILE_ORDER,
        size=n,
        p=profile_probabilities(cfg),
    )

    # ---------------- Deadlines ----------------

    deadlines = rng.uniform(
        cfg.deadline_min_ms,
        cfg.deadline_max_ms,
        size=n,
    )

    # ---------------- Initial channels ----------------

    snrs = np.clip(
        rng.normal(
            cfg.mean_snr_db,
            cfg.snr_std_db,
            size=n,
        ),
        cfg.min_snr_db,
        cfg.max_snr_db,
    )

    jobs = []

    for i in range(n):

        prof = profiles[tier_ids[i]]

        payload_factor = rng.lognormal(
            0.0,
            cfg.payload_noise_sigma
        )

        compute_factor = rng.lognormal(
            0.0,
            cfg.compute_noise_sigma
        )

        payload = (
            prof.payload_bytes
            * payload_factor
        )

        compute = (
            prof.edge_crypto_ms
            * compute_factor
        )

        bits = wire_bits(payload, cfg)

        jobs.append(
            Job(
                job_id=i,
                profile=prof.profile,

                arrival_ms=float(arrivals[i]),
                deadline_ms=float(deadlines[i]),
                hard_deadline_ms=cfg.hard_failure_ms,

                payload_bytes=float(payload),

                total_wire_bits=float(bits),
                remaining_wire_bits=float(bits),

                compute_work_ms=float(compute),
                remaining_compute_ms=float(compute),

                snr_db=float(snrs[i]),
                packet_loss=cfg.packet_loss,
            )
        )

    jobs.sort(
        key=lambda x: (
            x.arrival_ms,
            x.job_id
        )
    )

    return jobs


# ============================================================
# 6. SERVICE-TIME ESTIMATION
# ============================================================

def radio_service_ms(job, cfg):

    rate = (
        cfg.total_bandwidth_mhz
        * 1e6
        * spectral_efficiency(job.snr_db)
        * (1.0 - job.packet_loss)
    )

    return (
        1000.0
        * job.remaining_wire_bits
        / max(rate, 1e-12)
    )


def compute_service_ms(job, cfg):

    capacity = (
        cfg.mec_reference_cores
        * cfg.crypto_cpu_share
    )

    return (
        job.remaining_compute_ms
        / max(capacity, 1e-12)
    )


def remaining_service_ms(
    job,
    stage,
    cfg,
):

    comp = compute_service_ms(
        job,
        cfg
    )

    if stage == "compute":
        return comp

    radio = radio_service_ms(
        job,
        cfg
    )

    return radio + comp


def stage_service_ms(
    job,
    stage,
    cfg,
):

    if stage == "radio":
        return radio_service_ms(job, cfg)

    return compute_service_ms(job, cfg)


# ============================================================
# 7. OBJECTIVE SLACK
# ============================================================

def objective_slack_ms(
    job,
    t_ms,
    service_ms,
    cfg,
):

    soft = (
        job.absolute_deadline_ms
        - t_ms
    )

    if soft > 0:
        return max(
            soft,
            cfg.scheduler_epsilon_ms
        )

    hard = (
        job.absolute_hard_deadline_ms
        - t_ms
    )

    if hard > 0:
        return max(
            hard,
            cfg.scheduler_epsilon_ms
        )

    return max(
        service_ms,
        cfg.dt_ms,
        cfg.scheduler_epsilon_ms
    )


# ============================================================
# 8. REQUIRED STAGE FRACTIONS
# ============================================================

def required_stage_fractions(
    jobs,
    stage,
    t_ms,
    cfg,
):

    values = []

    for j in jobs:

        total_service = remaining_service_ms(
            j,
            stage,
            cfg,
        )

        stage_service = stage_service_ms(
            j,
            stage,
            cfg,
        )

        slack = objective_slack_ms(
            j,
            t_ms,
            total_service,
            cfg,
        )

        values.append(
            stage_service / slack
        )

    return values


# ============================================================
# 9. CAPACITY-BASED ADMISSION
# ============================================================

def capacity_admission_count(
    ordered_required,
    cfg,
):

    if len(ordered_required) == 0:
        return 0

    cap = cfg.admission_capacity_fraction

    total = 0.0
    worst = 0.0

    best_k = 1

    for k, d in enumerate(
        ordered_required,
        start=1
    ):

        d = max(
            float(d),
            cfg.scheduler_epsilon_ms
        )

        total += d
        worst = max(worst, d)

        feasible = (
            total <= cap + 1e-12
            and
            k * worst <= cap + 1e-12
        )

        if k == 1 or feasible:
            best_k = k
        else:
            break

    return max(
        1,
        min(
            len(ordered_required),
            best_k
        )
    )


# ============================================================
# 10. PARTIAL TIER NON-STARVATION
# ============================================================

def partial_tier_fair_select(
    jobs,
    ranked_indices,
    k,
    t_ms,
    cfg,
):

    if k <= 0:
        return []

    groups = {}

    for idx in ranked_indices:

        tier = jobs[idx].profile

        groups.setdefault(
            tier,
            []
        ).append(idx)

    tiers = [
        x
        for x in PROFILE_ORDER
        if x in groups
    ]

    L = len(tiers)

    if L == 0:
        return []

    if k >= len(jobs):
        return list(ranked_indices)

    # ------------------------------------------------
    # If fewer service slots than active tiers:
    # deterministically rotate which tiers are served.
    # ------------------------------------------------

    if k < L:

        epoch = int(
            round(
                t_ms
                / max(cfg.dt_ms, 1e-12)
            )
        )

        start = epoch % L

        chosen = []

        for r in range(k):

            tier = tiers[
                (start + r) % L
            ]

            chosen.append(
                groups[tier][0]
            )

        chosen_set = set(chosen)

        return [
            idx
            for idx in ranked_indices
            if idx in chosen_set
        ][:k]

    # ------------------------------------------------
    # One guaranteed opportunity per active tier
    # ------------------------------------------------

    quotas = {
        tier: 1
        for tier in tiers
    }

    remaining = k - L

    # ------------------------------------------------
    # rho controls how much of remaining capacity
    # receives proportional tier reservation
    # ------------------------------------------------

    if (
        remaining > 0
        and
        cfg.tier_fairness_fraction > 0
    ):

        counts = {
            tier: len(groups[tier])
            for tier in tiers
        }

        total = float(
            sum(counts.values())
        )

        rho = np.clip(
            cfg.tier_fairness_fraction,
            0.0,
            1.0
        )

        desired_reserved = {
            tier:
                rho
                * remaining
                * counts[tier]
                / total
            for tier in tiers
        }

        for tier in tiers:

            add = min(
                counts[tier] - 1,
                int(
                    math.floor(
                        desired_reserved[tier]
                    )
                )
            )

            quotas[tier] += max(
                0,
                add
            )

    chosen = []

    for tier in tiers:

        chosen.extend(
            groups[tier][
                :quotas[tier]
            ]
        )

    chosen_set = set(chosen)

    # Remaining unreserved slots:
    # use global workload/urgency ranking

    for idx in ranked_indices:

        if len(chosen) >= k:
            break

        if idx not in chosen_set:

            chosen.append(idx)

            chosen_set.add(idx)

    return [
        idx
        for idx in ranked_indices
        if idx in chosen_set
    ][:k]


# ============================================================
# 11. PQC-HO SCHEDULER
# ============================================================

def select_jobs(
    jobs,
    stage,
    t_ms,
    cfg,
):

    n = len(jobs)

    if n == 0:
        return []

    W = max(
        cfg.urgency_window_ms,
        cfg.dt_ms,
        cfg.scheduler_epsilon_ms,
    )

    scores = []

    for i, j in enumerate(jobs):

        service = remaining_service_ms(
            j,
            stage,
            cfg,
        )

        soft_slack = (
            j.absolute_deadline_ms
            - t_ms
        )

        hard_slack = (
            j.absolute_hard_deadline_ms
            - t_ms
        )

        # ---------------------------------------------
        # Before soft deadline
        # ---------------------------------------------

        if soft_slack > 0:

            urgency_bin = int(
                math.floor(
                    soft_slack / W
                )
            )

            key = (
                0,
                urgency_bin,
                service,
                j.absolute_deadline_ms,
                j.job_id,
            )

        # ---------------------------------------------
        # Soft deadline missed, hard threshold remains
        # ---------------------------------------------

        elif hard_slack > 0:

            urgency_bin = int(
                math.floor(
                    hard_slack / W
                )
            )

            key = (
                1,
                urgency_bin,
                service,
                j.absolute_hard_deadline_ms,
                j.job_id,
            )

        # ---------------------------------------------
        # Hard threshold already passed
        # ---------------------------------------------

        else:

            key = (
                2,
                service,
                j.arrival_ms,
                j.job_id,
            )

        scores.append(
            (i, key)
        )

    ranked_indices = [
        x[0]
        for x in sorted(
            scores,
            key=lambda x: x[1]
        )
    ]

    required = required_stage_fractions(
        jobs,
        stage,
        t_ms,
        cfg,
    )

    k = capacity_admission_count(
        [
            required[i]
            for i in ranked_indices
        ],
        cfg,
    )

    selected = partial_tier_fair_select(
        jobs,
        ranked_indices,
        k,
        t_ms,
        cfg,
    )

    # Verify actual fair-selected set is feasible

    cap = cfg.admission_capacity_fraction

    while (
        len(selected) > 1
        and
        (
            sum(
                required[i]
                for i in selected
            ) > cap + 1e-12

            or

            len(selected)
            * max(
                required[i]
                for i in selected
            ) > cap + 1e-12
        )
    ):

        k -= 1

        selected = partial_tier_fair_select(
            jobs,
            ranked_indices,
            k,
            t_ms,
            cfg,
        )

    return selected


# ============================================================
# 12. RESOURCE ALLOCATION
# ============================================================

def allocation_fractions(
    jobs,
    selected,
    stage,
    t_ms,
    cfg,
):

    out = np.zeros(
        len(jobs),
        dtype=float
    )

    if len(selected) == 0:
        return out

    required = required_stage_fractions(
        jobs,
        stage,
        t_ms,
        cfg,
    )

    demand = np.array([
        max(
            required[i],
            cfg.scheduler_epsilon_ms
        )
        for i in selected
    ])

    total_demand = demand.sum()

    if (
        not np.isfinite(total_demand)
        or total_demand <= 0
    ):

        out[selected] = (
            1.0 / len(selected)
        )

        return out

    if total_demand <= 1.0:

        residual = (
            1.0 - total_demand
        )

        shares = (
            demand
            + residual / len(selected)
        )

        shares /= shares.sum()

    else:

        shares = (
            demand / total_demand
        )

    out[selected] = shares

    return out


# ============================================================
# 13. SIMULATION
# ============================================================

def simulate(
    template_jobs,
    cfg,
    seed,
):

    rng = np.random.default_rng(
        seed + 880301
    )

    jobs = [
        j.clone()
        for j in template_jobs
    ]

    t_ms = 0.0
    next_idx = 0

    active = []

    last_arrival = max(
        j.arrival_ms
        for j in jobs
    )

    stop_at = min(
        cfg.max_simulation_ms,
        last_arrival
        + max(
            1500.0,
            12.0 * cfg.hard_failure_ms
        )
    )

    while t_ms <= stop_at:

        # ---------------------------------------------
        # New arrivals
        # ---------------------------------------------

        while (
            next_idx < len(jobs)
            and
            jobs[next_idx].arrival_ms
            <= t_ms + 1e-9
        ):

            active.append(
                jobs[next_idx]
            )

            next_idx += 1

        # ---------------------------------------------
        # Channel evolution
        # ---------------------------------------------

        for j in active:

            if j.phase == "radio":

                j.snr_db = float(
                    np.clip(
                        j.snr_db
                        + rng.normal(
                            0.0,
                            cfg.snr_random_walk_std_db
                        ),

                        cfg.min_snr_db,
                        cfg.max_snr_db,
                    )
                )

        radio_jobs = [
            j
            for j in active
            if j.phase == "radio"
        ]

        compute_jobs = [
            j
            for j in active
            if j.phase == "compute"
        ]

        # ---------------------------------------------
        # Scheduling
        # ---------------------------------------------

        radio_selected = select_jobs(
            radio_jobs,
            "radio",
            t_ms,
            cfg,
        )

        compute_selected = select_jobs(
            compute_jobs,
            "compute",
            t_ms,
            cfg,
        )

        radio_alloc = allocation_fractions(
            radio_jobs,
            radio_selected,
            "radio",
            t_ms,
            cfg,
        )

        compute_alloc = allocation_fractions(
            compute_jobs,
            compute_selected,
            "compute",
            t_ms,
            cfg,
        )

        # ---------------------------------------------
        # RADIO SERVICE
        # ---------------------------------------------

        total_bw_hz = (
            cfg.total_bandwidth_mhz
            * 1e6
        )

        for j, fraction in zip(
            radio_jobs,
            radio_alloc
        ):

            if fraction <= 0:
                continue

            bandwidth = (
                total_bw_hz
                * fraction
            )

            rate = (
                bandwidth
                * spectral_efficiency(
                    j.snr_db
                )
            )

            attempted_bits = (
                rate
                * cfg.dt_ms
                / 1000.0
            )

            served_bits = (
                attempted_bits
                * (1.0 - j.packet_loss)
            )

            actual = min(
                j.remaining_wire_bits,
                served_bits
            )

            j.remaining_wire_bits -= actual
            j.served_bits += actual

            if (
                j.remaining_wire_bits
                <= 1e-6
            ):

                j.remaining_wire_bits = 0.0

                j.phase = "compute"

                j.tx_done_ms = (
                    t_ms
                    + cfg.dt_ms
                )

        # ---------------------------------------------
        # MEC SERVICE
        # ---------------------------------------------

        compute_capacity = (
            cfg.mec_reference_cores
            * cfg.crypto_cpu_share
        )

        for j, fraction in zip(
            compute_jobs,
            compute_alloc
        ):

            if fraction <= 0:
                continue

            service = (
                compute_capacity
                * fraction
                * cfg.dt_ms
            )

            actual = min(
                j.remaining_compute_ms,
                service
            )

            j.remaining_compute_ms -= actual
            j.served_compute_ms += actual

            if (
                j.remaining_compute_ms
                <= 1e-9
            ):

                j.remaining_compute_ms = 0

                j.phase = "complete"

                j.complete_ms = (
                    t_ms
                    + cfg.dt_ms
                    + cfg.fixed_signaling_ms
                )

        # Remove finished requests

        active = [
            j
            for j in active
            if j.phase != "complete"
        ]

        if (
            next_idx >= len(jobs)
            and
            len(active) == 0
        ):
            break

        t_ms += cfg.dt_ms


    # ========================================================
    # METRICS
    # ========================================================

    records = []

    for j in jobs:

        if j.complete_ms is None:

            latency = np.nan

            violation = 1

            effective_latency = max(
                0.0,
                stop_at - j.arrival_ms
            )

        else:

            latency = (
                j.complete_ms
                - j.arrival_ms
            )

            violation = int(
                latency
                > j.deadline_ms
            )

            effective_latency = latency

        tardiness = max(
            0.0,
            effective_latency
            - j.deadline_ms
        )

        records.append({
            "job_id": j.job_id,
            "profile": j.profile,
            "latency_ms": latency,
            "deadline_violation": violation,
            "tardiness_ms": tardiness,
        })

    df = pd.DataFrame(records)

    return {
        "deadline_violation_rate":
            df["deadline_violation"].mean(),

        "mean_tardiness_ms":
            df["tardiness_ms"].mean(),

        "mean_latency_ms":
            df["latency_ms"].mean(),

        "p95_latency_ms":
            df["latency_ms"].quantile(0.95),
    }


# ============================================================
# 14. RUN ONE PARAMETER SETTING
# ============================================================

def run_setting(
    cfg,
    seeds
):

    rows = []

    for seed in seeds:

        jobs = generate_jobs(
            cfg,
            profiles,
            seed
        )

        metrics = simulate(
            jobs,
            cfg,
            seed
        )

        rows.append({
            "seed": seed,
            **metrics
        })

    return pd.DataFrame(rows)


# ============================================================
# 15. 95% CONFIDENCE INTERVAL
# ============================================================

def mean_ci(x):

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    mean = np.mean(x)

    if len(x) <= 1:

        return mean, np.nan

    ci = (
        stats.t.ppf(
            0.975,
            len(x) - 1
        )
        * stats.sem(x)
    )

    return mean, ci


# ============================================================
# 16. PARAMETER VALUES
# ============================================================

SEEDS = list(
    range(
        7001,
        7021
    )
)

W_VALUES = [
    10,
    15,
    22,
    30,
    40
]

RHO_VALUES = [
    0.60,
    0.70,
    0.85,
    0.90,
    1.00
]

BASE_CFG = SimConfig()

OUTDIR = Path(
    "PQC_HO_PARAMETER_SENSITIVITY"
)

OUTDIR.mkdir(
    exist_ok=True
)


# ============================================================
# 17. W SENSITIVITY
#     rho fixed at 0.85
# ============================================================

all_raw = []
summary_rows = []

print(
    "\n========================================"
)
print(
    "W SENSITIVITY"
)
print(
    "========================================"
)

for W in W_VALUES:

    print(
        f"Running W = {W} ms..."
    )

    cfg = replace(
        BASE_CFG,
        urgency_window_ms=float(W),
        tier_fairness_fraction=0.85,
    )

    result = run_setting(
        cfg,
        SEEDS
    )

    result["parameter"] = "W"
    result["value"] = W

    all_raw.append(result)

    v_mean, v_ci = mean_ci(
        100
        * result[
            "deadline_violation_rate"
        ]
    )

    t_mean, t_ci = mean_ci(
        result[
            "mean_tardiness_ms"
        ]
    )

    summary_rows.append({
        "Parameter": "W (ms)",
        "Value": W,

        "Viol. (%)": v_mean,
        "Viol. CI95": v_ci,

        "Tard. (ms)": t_mean,
        "Tard. CI95": t_ci,
    })


# ============================================================
# 18. rho SENSITIVITY
#     W fixed at 22 ms
# ============================================================

print(
    "\n========================================"
)
print(
    "RHO SENSITIVITY"
)
print(
    "========================================"
)

for rho in RHO_VALUES:

    print(
        f"Running rho = {rho:.2f}..."
    )

    cfg = replace(
        BASE_CFG,
        urgency_window_ms=22.0,
        tier_fairness_fraction=float(rho),
    )

    result = run_setting(
        cfg,
        SEEDS
    )

    result["parameter"] = "rho"
    result["value"] = rho

    all_raw.append(result)

    v_mean, v_ci = mean_ci(
        100
        * result[
            "deadline_violation_rate"
        ]
    )

    t_mean, t_ci = mean_ci(
        result[
            "mean_tardiness_ms"
        ]
    )

    summary_rows.append({
        "Parameter": "rho",
        "Value": rho,

        "Viol. (%)": v_mean,
        "Viol. CI95": v_ci,

        "Tard. (ms)": t_mean,
        "Tard. CI95": t_ci,
    })


# ============================================================
# 19. SAVE RESULTS
# ============================================================

raw_df = pd.concat(
    all_raw,
    ignore_index=True
)

summary_df = pd.DataFrame(
    summary_rows
)

raw_df.to_csv(
    OUTDIR
    / "parameter_sensitivity_raw.csv",
    index=False
)

summary_df.to_csv(
    OUTDIR
    / "parameter_sensitivity_summary.csv",
    index=False
)


# ============================================================
# 20. PRINT MANUSCRIPT RESULTS
# ============================================================

print(
    "\n\n========================================"
)
print(
    "FINAL PARAMETER SENSITIVITY TABLE"
)
print(
    "========================================"
)

print(
    summary_df[
        [
            "Parameter",
            "Value",
            "Viol. (%)",
            "Tard. (ms)"
        ]
    ].to_string(
        index=False,
        formatters={
            "Viol. (%)":
                lambda x: f"{x:.2f}",

            "Tard. (ms)":
                lambda x: f"{x:.2f}",
        }
    )
)


# ============================================================
# 21. GENERATE LATEX TABLE AUTOMATICALLY
# ============================================================

latex = r"""
\begin{table}[t]
\centering
\caption{Sensitivity to urgency-window width $W$ and tier-reservation
fraction $\rho$.}
\label{tab:param_sensitivity}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{c c c c}
\hline
Parameter & Value & Viol. (\%) & Tard. (ms) \\
\hline
"""

w_part = summary_df[
    summary_df["Parameter"]
    == "W (ms)"
]

for i, (_, row) in enumerate(
    w_part.iterrows()
):

    pname = (
        r"$W$ (ms)"
        if i == 0
        else ""
    )

    latex += (
        f"{pname} & "
        f"{row['Value']:g} & "
        f"{row['Viol. (%)']:.2f} & "
        f"{row['Tard. (ms)']:.2f} "
        "\\\\\n"
    )


latex += "\\hline\n"


rho_part = summary_df[
    summary_df["Parameter"]
    == "rho"
]

for i, (_, row) in enumerate(
    rho_part.iterrows()
):

    pname = (
        r"$\rho$"
        if i == 0
        else ""
    )

    latex += (
        f"{pname} & "
        f"{row['Value']:.2f} & "
        f"{row['Viol. (%)']:.2f} & "
        f"{row['Tard. (ms)']:.2f} "
        "\\\\\n"
    )


latex += r"""\hline
\end{tabular}
\end{table}
"""


print(
    "\n========================================"
)
print(
    "LATEX TABLE"
)
print(
    "========================================\n"
)

print(latex)


with open(
    OUTDIR
    / "parameter_sensitivity_table.tex",
    "w"
) as f:

    f.write(latex)


# ============================================================
# 22. PLOT W
# ============================================================

wplot = summary_df[
    summary_df["Parameter"]
    == "W (ms)"
].copy()

plt.figure(
    figsize=(6, 4)
)

plt.errorbar(
    wplot["Value"],
    wplot["Viol. (%)"],
    yerr=wplot["Viol. CI95"],
    marker="o",
    capsize=4,
)

plt.axvline(
    22,
    linestyle="--",
    alpha=0.6,
)

plt.xlabel(
    r"Urgency-window width $W$ (ms)"
)

plt.ylabel(
    "Deadline violations (%)"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTDIR
    / "sensitivity_W.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 23. PLOT rho
# ============================================================

rplot = summary_df[
    summary_df["Parameter"]
    == "rho"
].copy()

plt.figure(
    figsize=(6, 4)
)

plt.errorbar(
    rplot["Value"],
    rplot["Viol. (%)"],
    yerr=rplot["Viol. CI95"],
    marker="o",
    capsize=4,
)

plt.axvline(
    0.85,
    linestyle="--",
    alpha=0.6,
)

plt.xlabel(
    r"Tier-reservation fraction $\rho$"
)

plt.ylabel(
    "Deadline violations (%)"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTDIR
    / "sensitivity_rho.pdf",
    bbox_inches="tight"
)

plt.show()


print(
    "\nFinished."
)

print(
    f"Results saved in: {OUTDIR.resolve()}"
)