"""Exercise workload splitting with synthetic data, without launching SimAI."""

import importlib.util
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "trace_adapter", ROOT / "scripts" / "generate_cluster_trace_jobs.py")
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TraceWorkloadSplittingTest(unittest.TestCase):
    def test_stalled_jobs_have_bounded_network_flows(self):
        # The observed failure used two channels on two GPUs, resulting in
        # 7559994/12959994-byte flows with an 8994-byte final payload.
        for total in (30239975, 51839975):
            self.assertEqual(((total + 3) // 4) % 9000, 8994)
            parts = list(ADAPTER.split_communication(total, 2))
            self.assertGreater(len(parts), 1)
            self.assertEqual(sum(parts), total)
            for channels in (1, 2):
                for part in parts:
                    payload = part // 2 // channels
                    self.assertGreater(payload, 60)
                    self.assertLessEqual(payload, 8192)
                    self.assertLess(payload + 512, 9000)

    def test_bounds_and_total_for_different_gpu_counts(self):
        for gpus in (1, 2, 3, 4, 7, 8, 16, 56, 128):
            for cap in (8192, 65536, 1048576, 4194304):
                limit = ADAPTER.allreduce_limit(gpus, cap)
                for total in (1, 4095, 4096, limit - 1, limit, limit + 1,
                              2 * limit + 1, 260633500):
                    parts = list(ADAPTER.split_communication(total, gpus, cap))
                    self.assertEqual(sum(parts), total)
                    self.assertTrue(all(0 < part <= limit for part in parts))
                    self.assertLessEqual(max(parts) - min(parts), 1)
                    if len(parts) > 1:
                        self.assertGreaterEqual(min(parts), 4096)

    def test_limit_plus_one_does_not_leave_one_byte_tail(self):
        limit = ADAPTER.allreduce_limit(2, 1048576)
        self.assertEqual(list(ADAPTER.split_communication(limit + 1, 2)),
                         [8193, 8192])

    def test_workload_layer_count_ids_and_bytes(self):
        with tempfile.TemporaryDirectory(prefix="simai-split-test-") as directory:
            path = Path(directory) / "workload.txt"
            count = ADAPTER.write_workload(path, 2, 30239975)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertIn("all_gpus: 2", lines[0])
            self.assertEqual(int(lines[1]), count)
            self.assertEqual(len(lines) - 2, count)
            amounts = []
            for index, line in enumerate(lines[2:]):
                tokens = line.split()
                self.assertEqual(tokens[0], "trace_job_part_{}".format(index))
                self.assertEqual(tokens[1:4], ["-1", "1", "ALLREDUCE"])
                self.assertEqual(tokens[5:], ["1", "NONE", "0", "1", "NONE", "0", "1"])
                amounts.append(int(tokens[4]))
            self.assertEqual(sum(amounts), 30239975)

    def test_small_job_retains_single_collective(self):
        with tempfile.TemporaryDirectory(prefix="simai-split-test-") as directory:
            path = Path(directory) / "small.txt"
            self.assertEqual(ADAPTER.write_workload(path, 8, 950), 1)
            self.assertEqual(path.read_text().splitlines()[2],
                             "trace_job -1 1 ALLREDUCE 950 1 NONE 0 1 NONE 0 1")

    def test_manifest_modes_keep_independent_split_workloads(self):
        for compact in (False, True):
            with self.subTest(compact=compact), tempfile.TemporaryDirectory(
                    prefix="simai-split-test-") as directory:
                trace_dir = Path(directory)
                with (trace_dir / "job_info_df.csv").open("w", newline="") as output:
                    writer = csv.writer(output)
                    writer.writerow(["job_name", "gpu_request", "worker_num",
                                     "submit_time", "duration"])
                    writer.writerow([1, 1, 1, 0, 1209599])
                    writer.writerow([2, 1, 1, 200, 2073599])
                (trace_dir / "node_info_df.csv").write_text("node_id\nnode0\n")
                output_dir = trace_dir / "out"
                argv = ["generate_cluster_trace_jobs.py", "--trace-dir", str(trace_dir),
                        "--output-dir", str(output_dir)]
                if compact:
                    argv.append("--compact-csv")
                with mock.patch.object(ADAPTER.sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(ADAPTER.main(), 0)
                with (output_dir / "multi_jobs.csv").open(newline="") as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), 2)
                self.assertNotEqual(rows[0]["workload"], rows[1]["workload"])
                for row, expected in zip(rows, (604800, 1036800)):
                    self.assertEqual(row["gpus"], "2")
                    lines = Path(row["workload"]).read_text().splitlines()
                    self.assertGreater(int(lines[1]), 1)
                    self.assertEqual(sum(int(line.split()[4]) for line in lines[2:]), expected)
                    self.assertEqual("communication_bytes" in row, compact)
                    if compact:
                        self.assertEqual(int(row["communication_bytes"]), expected)
                summary = json.loads((output_dir / "trace_adapter_summary.json").read_text())
                self.assertEqual(summary["jobs_with_split_allreduces"], 2)
                self.assertEqual(summary["total_workload_communication_bytes"], 1641600)

    def test_invalid_limits_and_totals(self):
        for gpus, cap in ((0, 1048576), (-1, 1048576), (2, 0), (2, 4096)):
            with self.assertRaises(ValueError):
                list(ADAPTER.split_communication(100000, gpus, cap))
        for total in (0, -1):
            with self.assertRaises(ValueError):
                list(ADAPTER.split_communication(total, 2))


if __name__ == "__main__":
    unittest.main()
