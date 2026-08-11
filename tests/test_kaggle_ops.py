import json
import tempfile
import unittest
from pathlib import Path

from src.kaggle_ops import (
    RagRunConfig,
    atomic_write_json,
    bind_input_pdf,
    canonical_hash,
    create_run_layout,
    mark_phase_complete,
    seal_run,
)


class KaggleOpsTests(unittest.TestCase):
    def test_canonical_hash_ignores_mapping_order(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))

    def test_resume_checks_config_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_run_layout(directory, RagRunConfig(), resume_run_id="known-run")
            second = create_run_layout(directory, RagRunConfig(), resume_run_id="known-run")
            self.assertEqual(first["config_hash"], second["config_hash"])

    def test_pdf_fingerprint_is_sealed_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            run = create_run_layout(directory, RagRunConfig(), resume_run_id="known-run")
            bind_input_pdf(
                run["run_dir"], {"sha256": "a", "bytes": 10, "page_count": 426}
            )
            with self.assertRaisesRegex(RuntimeError, "PDF fingerprint differs"):
                bind_input_pdf(
                    run["run_dir"],
                    {"sha256": "b", "bytes": 10, "page_count": 426},
                )

    def test_phase_and_seal_files_are_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            run = create_run_layout(directory, RagRunConfig(), resume_run_id="known-run")
            mark_phase_complete(run["run_dir"], "retrieval", {"mrr": 0.5})
            metrics_path = Path(run["results"]) / "metrics.json"
            atomic_write_json(
                metrics_path,
                {
                    "sealed_test": True,
                    "post_test_tuning_allowed": False,
                    "mrr": 0.5,
                },
            )
            lock = seal_run(run["run_dir"], metrics_path)
            parsed = json.loads(
                (Path(run["run_dir"]) / "COMPLETE.lock.json").read_text(encoding="utf-8")
            )
            self.assertEqual(parsed["metrics_sha256"], lock["metrics_sha256"])


if __name__ == "__main__":
    unittest.main()
