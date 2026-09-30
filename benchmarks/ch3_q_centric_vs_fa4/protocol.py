"""CPU-only, fixed protocol for the current p03 vs pinned FA4 comparison."""

SEEDS = (17, 29, 41)
LENGTHS = (2048, 4096, 8192, 16384)
ROUNDS = 3
WARMUP_MS = 250
MEASURE_MS = 1000
CALLS_PER_GRAPH = 8
PROTOCOL = "p03_fa4_graph8_w250_m1000_r3_seeds17_29_41_v1"


def performance_cases(skv):
    if skv not in LENGTHS:
        raise ValueError("KV length must be 2048, 4096, 8192, or 16384")
    cases = []
    for index, seed in enumerate(SEEDS):
        patterns = ["all"] if skv == 2048 else ["independent", "paired"]
        if index % 2:
            patterns.reverse()
        cases.extend((pattern, seed) for pattern in patterns)
    return cases


def measurement_order(names, seed, round_index):
    # Nine backends, three rounds, three seeds: each backend starts once.
    shift = (SEEDS.index(seed) * ROUNDS + round_index) % len(names)
    return list(names[shift:]) + list(names[:shift])
