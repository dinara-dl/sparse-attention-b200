"""Pure CPU aggregation tests; no synthetic results are written into evidence."""
import json
import unittest
from summarize import aggregate, LABELS
from protocol import SEEDS, LENGTHS


class FakePath:
    def __init__(self, name, data):
        self.name, self.stem, self.data = name, name.removesuffix(".json"), data

    def read_text(self):
        return json.dumps(self.data)

    def __lt__(self, other):
        return self.name < other.name


def fixture():
    files = []
    for skv in LENGTHS:
        records = []
        for selector in LABELS:
            for seed, a, b in zip(SEEDS, (1, 9, 9), (10, 2, 10)):
                samples = {"selector": [1]*3, "ours_attention": [2]*3, "ours_pipeline": [5]*3,
                           "fa4_a_pipeline": [a]*3, "fa4_b_pipeline": [b]*3}
                records.append({"selector": selector, "seed": seed,
                                "timing": {"samples_ms": samples, "median_ms": {n: v[0] for n, v in samples.items()}}})
        files.append(FakePath(f"test_performance_{skv}.json", {
            "passed": True, "mode": "performance", "skv": skv, "records": records,
            "provenance": {"source_sha256": {"kernel": "fake"}}}))
    return files


class SummaryTests(unittest.TestCase):
    def test_fixed_config_not_oracle(self):
        data = aggregate(fixture())
        self.assertEqual(data["case_count"], 27)
        for row in data["rows"]:
            self.assertEqual(row["best_fa4"], "fa4_a_pipeline")
            self.assertEqual(row["median_us"]["fa4_a_pipeline"], 9000)
            self.assertEqual(row["pipeline_speedup"], 1.8)

    def test_ledger_excluded(self):
        files = fixture() + [FakePath("test_performance_call.json", {"passed": True, "mode": "performance"})]
        self.assertEqual(aggregate(files)["case_count"], 27)

    def test_duplicate_rejected(self):
        files = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate(files + files[:1])

    def test_mixed_sources_rejected(self):
        files = fixture()
        files[1].data["provenance"]["source_sha256"]["kernel"] = "different"
        with self.assertRaisesRegex(ValueError, "mixed source"):
            aggregate(files)

    def test_incomplete_rejected(self):
        with self.assertRaises(KeyError):
            aggregate(fixture()[:1])


if __name__ == "__main__":
    unittest.main()
