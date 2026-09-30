import ast
from pathlib import Path
import unittest
from protocol import SEEDS, METHODS, order, rotated


class ProtocolTests(unittest.TestCase):
    def test_rotations(self):
        self.assertEqual([rotated(METHODS, i)[0] for i in range(3)], list(METHODS))

    def test_order_is_permutation(self):
        for si in range(len(SEEDS)):
            for mi in range(len(METHODS)):
                for rnd in range(3):
                    self.assertEqual(sorted(order(range(7), si, mi, rnd)), list(range(7)))

    def test_timeout(self):
        tree = ast.parse(Path(__file__).with_name("run_modal.py").read_text())
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "function"]
        self.assertEqual(len(functions), 1)
        value = next(k.value for k in functions[0].keywords if k.arg == "timeout")
        self.assertEqual(ast.unparse(value), "8 * 60")

    def test_current_loader(self):
        text = Path(__file__).with_name("current_benchmark.py").read_text()
        self.assertIn('stage_functions("p03")[1]', text)
        self.assertNotIn("load_ours(", text)


if __name__ == "__main__":
    unittest.main()
