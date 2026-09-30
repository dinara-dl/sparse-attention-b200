"""Training-free BF16 block summaries and an exact GPU FA4 Q2 adapter."""
import triton as tr
import triton.language as tl


@tr.jit
def pool_blocks(X, Desc, H: tl.constexpr, T: tl.constexpr, R: tl.constexpr,
                SB: tl.constexpr, SH: tl.constexpr, SS: tl.constexpr):
    p = tl.program_id(0)
    r, tile = p % R, (p // R) % T
    head, batch = (p // (R * T)) % H, p // (R * T * H)
    token = tile * 128 + r * (128 // R) + tl.arange(0, 128 // R)
    d = tl.arange(0, 128)
    values = tl.load(X + batch * SB + head * SH + token[:, None] * SS + d[None, :]).to(tl.float32)
    mean = tl.sum(values, axis=0) / (128 // R)
    tl.store(Desc + p * 128 + d, mean)


@tr.jit
def descriptor_dot(Q, K, Raw, M: tl.constexpr, N: tl.constexpr):
    batch = tl.program_id(2)
    m = tl.program_id(0) * 32 + tl.arange(0, 32)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    d = tl.arange(0, 128)
    q = tl.load(Q + (batch * M + m[:, None]) * 128 + d[None, :], m[:, None] < M, 0)
    k = tl.load(K + (batch * N + n[None, :]) * 128 + d[:, None], n[None, :] < N, 0)
    score = tl.dot(q, k).to(tl.float32) * (128.0 ** -0.5)
    tl.store(Raw + (batch * M + m[:, None]) * N + n[None, :], score,
             (m[:, None] < M) & (n[None, :] < N))


@tr.jit
def reduce_topk(Raw, Scores, IDs, QT: tl.constexpr, NT: tl.constexpr, R: tl.constexpr):
    p = tl.program_id(0)
    batch, qt = p // QT, p % QT
    hr = tl.arange(0, 16 * R)
    n = tl.arange(0, NT * R)
    m = (hr // R * QT + qt) * R + hr % R
    values = tl.load(Raw + (batch * (16 * QT * R) + m[:, None]) * (NT * R) + n[None, :])
    values = tl.reshape(values, (16 * R, NT, R))
    score = tl.max(tl.max(values, axis=0), axis=1)
    block_ids = tl.arange(0, NT)
    tl.store(Scores + p * NT + block_ids, score)
    # Stable descending score order: lower block index wins a tie.
    for rank in range(16):
        best = tl.max(score, axis=0)
        chosen = tl.min(tl.where(score == best, block_ids, 2147483647), axis=0)
        tl.store(IDs + p * 16 + rank, chosen)
        score = tl.where(block_ids == chosen, -float("inf"), score)


@tr.jit
def q2_metadata(IDs, FC, FI, PC, PI, Lookup, QT: tl.constexpr, NT: tl.constexpr):
    p = tl.program_id(0)
    batch, pair = p // (QT // 2), p % (QT // 2)
    slots = tl.arange(0, 16)
    a = tl.load(IDs + (batch * QT + pair * 2) * 16 + slots)
    b = tl.load(IDs + (batch * QT + pair * 2 + 1) * 16 + slots)
    blocks = tl.arange(0, NT)
    allow_a = tl.sum((blocks[:, None] == a[None, :]).to(tl.int32), axis=1) > 0
    allow_b = tl.sum((blocks[:, None] == b[None, :]).to(tl.int32), axis=1) > 0
    full, partial = allow_a & allow_b, allow_a ^ allow_b
    fpos = tl.cumsum(full.to(tl.int32), axis=0) - 1
    ppos = tl.cumsum(partial.to(tl.int32), axis=0) - 1
    tl.store(FI + p * 32 + fpos, blocks, full)
    tl.store(PI + p * 32 + ppos, blocks, partial)
    tl.store(FC + p, tl.sum(full.to(tl.int32), axis=0))
    tl.store(PC + p, tl.sum(partial.to(tl.int32), axis=0))
    tl.store(Lookup + (batch * QT + pair * 2) * NT + blocks, allow_a.to(tl.uint8))
    tl.store(Lookup + (batch * QT + pair * 2 + 1) * NT + blocks, allow_b.to(tl.uint8))
