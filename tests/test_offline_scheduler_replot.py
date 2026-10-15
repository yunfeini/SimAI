"""Verify offline dispatch without rendering plots or launching simulations."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("offline_replot", SCRIPTS / "plot_scheduler_comparison.py")
REPLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLOT)


class OfflineReplotTest(unittest.TestCase):
    def test_only_plotting_is_dispatched_and_original_summary_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="simai-replot-test-") as directory:
            root = Path(directory)
            for policy in ("fcfs", "fgd", "crux"):
                (root / policy).mkdir()
                (root / policy / "SimAI_scheduler.csv").write_text("existing log\n")
            summary = root / "comparison_summary.json"
            summary.write_text("original simulation summary\n")
            with mock.patch.object(sys, "argv", ["replot", "--run-dir", str(root), "--curve-bins", "40"]), \
                    mock.patch.object(REPLOT, "plot_policy", return_value="ok") as policy_plot, \
                    mock.patch.object(REPLOT, "plot_comparison", return_value="ok") as comparison_plot, \
                    mock.patch.object(REPLOT, "result_metrics", return_value={"e2e_mean_ns": 7}), \
                    mock.patch("subprocess.run", side_effect=AssertionError("no subprocess expected")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(REPLOT.main(), 0)
            self.assertEqual(policy_plot.call_count, 3)
            self.assertTrue(all(call.args[2] == 0 for call in policy_plot.call_args_list))
            self.assertEqual(comparison_plot.call_args.args[2:], (0, 40))
            self.assertEqual(summary.read_text(), "original simulation summary\n")
            metadata = json.loads((root / "replot_summary.json").read_text())
            self.assertEqual(metadata["mode"], "offline_replot")
            self.assertEqual(metadata["curve_bins"], 40)
            self.assertEqual(metadata["max_latency_jobs"], 15)

    def test_missing_log_fails_before_plotting(self):
        with tempfile.TemporaryDirectory(prefix="simai-replot-test-") as directory:
            with mock.patch.object(sys, "argv", ["replot", "--run-dir", directory, "--policies", "fcfs"]), \
                    mock.patch.object(REPLOT, "plot_policy") as plot, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                REPLOT.main()
            self.assertEqual(error.exception.code, 2)
            plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
