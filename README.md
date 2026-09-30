# sparse-attention-b200

Many frontier models use some form of sparse attention to handle long contexts efficiently. In my blog, I show how to build a block-sparse attention kernel on NVIDIA B200: I start from a dense kernel, then cover types of sparse attention, indexers, GQA, and two ways to schedule the work: Q-centric and KV-centric.

Blog post: https://dinara-dl.github.io/posts/sparse-attention/

## What's here

- `attention/q_centric/` – block-sparse kernel and its GQA version
- `attention/kv_centric/` – KV-centric kernel
- `indexers/` – R1, R4 and FP4 block selectors
- `benchmarks/` – scripts and raw results for the tables in the post
- `blog/` – the post and its diagrams

## Notes

- Forward only, non-causal, BF16, head dim 128, B200 only.
- Data is synthetic, so there are no model quality results.
- Scripts use paths from my working folder, so they don't run from this repo as is.

## Credits

Dense baseline: [Iaroslav Elistratov's B200 attention kernel series](https://iaroslavelistratov.github.io/b200-attention/). FP4 scorer: MiniMax MSA (MIT license). KV-centric dataflow follows the MiniMax MSA paper.
