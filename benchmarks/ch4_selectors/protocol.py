"""CPU-only definitions for the current p03 selector benchmark."""
SEEDS = (17, 29, 41)
LENGTHS = (4096, 8192, 16384)
METHODS = ("r1", "r4", "fp4")
WARMUP_MS, MEASURE_MS, ROUNDS, GRAPH_CALLS = 250, 1000, 3, 8


def rotated(items, offset):
    items = list(items)
    offset %= len(items)
    return items[offset:] + items[:offset]


def order(items, seed_index, method_index, round_index):
    result = rotated(items, seed_index + method_index + round_index)
    return result[::-1] if round_index % 2 else result
