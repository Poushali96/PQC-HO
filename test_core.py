import math
import unittest
from pathlib import Path

from pqcho.experiments import (
    PROFILE_ORDER,
    SimConfig,
    capacity_admission_count,
    demo_profiles,
    load_profiles,
    spectral_efficiency,
    wire_bits,
)


class CoreTests(unittest.TestCase):
    def test_spectral_efficiency_at_zero_db(self):
        self.assertTrue(math.isclose(spectral_efficiency(0.0), 1.0))

    def test_wire_bits_includes_per_packet_overhead(self):
        cfg = SimConfig(mtu_bytes=1500, per_packet_overhead_bytes=64)
        self.assertEqual(wire_bits(1501, cfg), (1501 + 2 * 64) * 8)

    def test_demo_profiles_are_complete_and_ordered(self):
        profiles = demo_profiles()
        self.assertEqual(list(profiles), PROFILE_ORDER)
        self.assertLess(profiles["P1"].payload_bytes, profiles["P2"].payload_bytes)
        self.assertLess(profiles["P2"].payload_bytes, profiles["P3"].payload_bytes)

    def test_capacity_admission_stops_at_infeasible_prefix(self):
        cfg = SimConfig(admission_capacity_fraction=1.0)
        self.assertEqual(capacity_admission_count([0.20, 0.20, 0.40], cfg), 2)
        self.assertEqual(capacity_admission_count([0.20, 0.20, 0.40], cfg, disable_cap=True), 3)

    def test_reported_profiles_load(self):
        path = Path(__file__).resolve().parents[1] / "data" / "pqc_profiles_reported.csv"
        profiles = load_profiles(str(path))
        self.assertEqual(list(profiles), PROFILE_ORDER)
        self.assertAlmostEqual(profiles["P1"].edge_crypto_ms, 0.089634)


if __name__ == "__main__":
    unittest.main()
