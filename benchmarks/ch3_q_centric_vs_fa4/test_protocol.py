import ast
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fa4_same_mask"))
from metadata import build_plan, make_lists
from protocol import LENGTHS, ROUNDS, SEEDS, measurement_order, performance_cases


class ProtocolTests(unittest.TestCase):
    def test_case_coverage(self):
        self.assertEqual(performance_cases(2048), [("all", seed) for seed in SEEDS])
        for size in LENGTHS[1:]:
            self.assertEqual(set(performance_cases(size)),
                             {(pattern, seed) for pattern in ("independent", "paired") for seed in SEEDS})

    def test_rotation(self):
        names = list(range(9))
        starts = []
        for seed in SEEDS:
            for rnd in range(ROUNDS):
                order = measurement_order(names, seed, rnd)
                self.assertEqual(sorted(order), names)
                starts.append(order[0])
        self.assertEqual(sorted(starts), names)

    def test_exact_masks_all_cases(self):
        for skv in LENGTHS:
            for pattern, seed in performance_cases(skv):
                lists = make_lists(2, 1, 32, skv // 128, pattern, seed + 1000)
                for stage in (1, 2):
                    plan = build_plan(lists, skv // 128, stage)
                    self.assertEqual(plan.stats()["useful_q128_kv128_pairs_per_kv_head"], 2 * 32 * 16)
                    if pattern in ("paired", "all"):
                        self.assertEqual(plan.stats()["partial_block_count"], 0)

    def test_timeout_and_syntax(self):
        for name in ("protocol.py", "current_benchmark.py", "run_modal.py"):
            ast.parse((HERE / name).read_text())
        runner = (HERE / "run_modal.py").read_text()
        self.assertIn("timeout=8 * 60", runner)


if __name__ == "__main__":
    unittest.main()
