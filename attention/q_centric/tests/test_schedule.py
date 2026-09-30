"""CPU address/ring model, NOT a substitute for CUDA correctness or race tests."""
import random
import unittest


def check_ring(count, shared, seed):
    rng = random.Random(seed)
    tokens_per_slot = 2 if shared else 4

    def token(slot, kind, q):
        return tokens_per_slot * slot + (kind if shared else 2 * kind + q)

    operations = [(token(0, 0, q), q) for q in range(2)]
    for slot in range(count):
        for q in range(2):
            operations.append((token(slot, 1, q), q))
            if slot + 1 < count:
                operations.append((token(slot + 1, 0, q), q))
    total = tokens_per_slot * count
    required = {t: (2 if shared else 1) for t in range(total)}
    buffers = [None] * 3
    consumed = [0] * total
    producer = consumer = 0
    pending = []
    ready_phase = [0] * 3
    observed_generation = [-1] * 3
    ticks = 0
    while producer < total or consumer < len(operations) or pending:
        possible = []
        if producer < total and (producer < 3 or consumed[producer - 3] == required[producer - 3]):
            possible.append("load")
        if consumer < len(operations):
            t, q = operations[consumer]
            if buffers[t % 3] == t:
                possible.append("issue")
        if pending:
            possible.append("complete")
        assert possible, ("deadlock", count, shared, producer, consumer, buffers)
        action = rng.choice(possible)
        if action == "load":
            st = producer % 3
            old = buffers[st]
            assert old is None or consumed[old] == required[old], "overwrite before last consumer"
            buffers[st] = producer
            ready_phase[st] ^= 1
            producer += 1
        elif action == "issue":
            t, q = operations[consumer]
            st, generation = t % 3, t // 3
            if observed_generation[st] != generation:
                assert generation == observed_generation[st] + 1
                assert ready_phase[st] == ((generation + 1) & 1)
                observed_generation[st] = generation
            else:
                assert shared and q == 1
            pending.append((t, q))
            consumer += 1
        else:
            t, q = pending.pop(0)
            assert buffers[t % 3] == t, "asynchronous MMA reads overwritten buffer"
            consumed[t] += 1
        ticks += 1
        assert ticks < 20 * total
    assert all(consumed[t] == required[t] for t in range(total))


class ScheduleTests(unittest.TestCase):
    def test_ring(self):
        for shared in (False, True):
            for count in (1, 2, 3, 4, 15, 16, 17, 32, 128):
                for seed in range(100):
                    check_ring(count, shared, seed)

    def test_mapping_and_persistence(self):
        for group in (1, 2, 4, 8, 16):
            for qtiles in (2, 4, 32):
                batch, hkv = 2, 3
                hq = group * hkv
                for shared in (False, True):
                    if shared and group == 1:
                        continue
                    work = batch * hq * qtiles // 2
                    for grid in (1, 3, min(148, work)):
                        visits = []
                        for cta in range(grid):
                            for bid in range(cta, work, grid):
                                if shared:
                                    head0, tile0 = 2 * (bid // qtiles), bid % qtiles
                                    pairs = [(head0 + q, tile0) for q in range(2)]
                                else:
                                    head, pair = divmod(bid, qtiles // 2)
                                    pairs = [(head, pair * 2 + q) for q in range(2)]
                                for head, tile in pairs:
                                    b, h = divmod(head, hq)
                                    kv_instance = head // group
                                    self.assertEqual(kv_instance, b * hkv + h // group)
                                    visits.append((b, h, tile))
                                if shared:
                                    self.assertEqual(pairs[0][0] // group, pairs[1][0] // group)
                        self.assertEqual(len(visits), len(set(visits)))
                        self.assertEqual(len(visits), batch * hq * qtiles)


if __name__ == "__main__":
    unittest.main()
