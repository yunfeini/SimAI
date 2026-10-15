"""Check duration-weighted plotting independently of matplotlib and real logs."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scheduler_plot_style", ROOT / "scripts" / "scheduler_plot_style.py")
STYLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STYLE)


class TimeBinCurveTest(unittest.TestCase):
    def test_irregular_events_are_weighted_by_duration(self):
        x, mean, low, high = STYLE.time_bin_curve(
            [0, 1, 9, 10], [100, 0, 100, 0], bins=2)
        self.assertEqual(x, [2.5, 7.5])
        self.assertEqual(mean, [20, 20])
        self.assertEqual(low, [0, 0])
        self.assertEqual(high, [100, 100])

    def test_last_state_wins_at_same_timestamp(self):
        _, mean, low, high = STYLE.time_bin_curve(
            [0, 1, 1, 3, 4], [0, 90, 10, 0, 99], bins=1)
        self.assertEqual(mean, [5])
        self.assertEqual(low, [0])
        self.assertEqual(high, [10])

    def test_average_stays_bounded_and_preserves_integral(self):
        times = [0, 1, 2, 2.5, 6, 9, 10]
        values = [0, 100, 0, 50, 0, 90, 0]
        integral = sum(values[i] * (times[i + 1] - times[i])
                       for i in range(len(times) - 1))
        for bins in (1, 2, 3, 4):
            x, mean, low, high = STYLE.time_bin_curve(times, values, bins)
            self.assertEqual(len(x), bins)
            self.assertAlmostEqual(sum(mean) * 10 / bins, integral)
            self.assertTrue(all(a <= b <= c for a, b, c in zip(low, mean, high)))

    def test_raw_mode_keeps_values_without_mutating_input(self):
        times, values = [0, 1, 2, 10], [0, 100, 0, 40]
        x, mean, low, high = STYLE.time_bin_curve(times, values, bins=0)
        self.assertEqual(x, times)
        self.assertEqual(mean, values)
        self.assertEqual(low, values)
        self.assertEqual(high, values)
        mean[0] = 50
        self.assertEqual(values[0], 0)

    def test_empty_and_single_timestamp(self):
        self.assertEqual(STYLE.time_bin_curve([], [], 60), ([], [], [], []))
        self.assertEqual(STYLE.time_bin_curve([1, 1], [0, 100], 1),
                         ([1], [100], [100], [100]))

    def test_invalid_input(self):
        for times, values, bins in (([0], [], 1), ([0], [1], -1),
                                    ([0], [float("nan")], 1)):
            with self.assertRaises(ValueError):
                STYLE.time_bin_curve(times, values, bins)


if __name__ == "__main__":
    unittest.main()
