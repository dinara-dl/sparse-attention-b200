# Выбор референса: MSA как основа, FSA как дополнительный источник

Проверено 2026-09-06 до реализации. Это выбор dataflow, не копирование библиотеки
и не сравнение скорости. Математический oracle остаётся независимым PyTorch.

| Вопрос | MSA | FSA v3 | Решение v0 |
|---|---|---|---|
| Выбор | Per-query и GQA-group | Per-query sparse selection | Готовые per-token TopK16 |
| Владение работой | KV-block / chunks связанных Q | Query-head / KV-block | KV-block / fixed chunk |
| Упаковка | Query positions и GQA-heads | В описании обход heads по отдельности | M128 из query × GQA heads |
| Межблочный softmax | Локальный O и LSE, затем Combine | Отдельный предварительный проход statistics | Локальный O и LSE |
| Сложный runtime | Scheduling, paging и другие функции | Свои mappings / reduction | Только CPU CSR и fixed chunks |

MSA §4.2 непосредственно описывает нужный двухфазный forward. FSA §3.1–3.2
полезна для разбора непоследовательных query и распределённых outputs, но её
отдельный statistics pass для этой первой версии не нужен. Это не утверждение,
что FSA вообще хуже: performance здесь не исследуется.

Не перенимаем утверждение, будто локальные статистики сами по себе запрещены:
локальные softmax корректно объединяются при сохранении веса каждого раздела.
Для натурального логарифма и нормализованного partial `O_j`:

```
L_j = logsumexp(scores выбранного блока j)
L   = logsumexp_j(L_j)
O   = sum_j exp(L_j - L) * O_j
```

Просто сложить или усреднить локальные outputs неверно. Внутренний BF16 P для
Tensor Core PV вносит обычную численную погрешность; partial O/LSE в v0 FP32.
До GPU-запуска фиксируем допуск final O `atol=0.02, rtol=0.02`, LSE
`atol=0.002, rtol=0.0002`; не расширяем его после неудачного результата.

## Проверенные первичные источники

1. [MSA v2, §4.2](https://arxiv.org/html/2606.13392v2#S4.SS2): reverse relation,
   chunking, query concatenation, normalized partial O/LSE и Combine.
2. [FSA v3, §3.1–3.2](https://arxiv.org/html/2508.18224v3#S3.SS2): KV outer,
   non-contiguous Q, output mappings и выбранный авторами statistics prepass.
3. Локальный MSA checkout HEAD `80434d7f67877c6570ca19cac444b84bc9855dac`,
   `python/fmha_sm100/cute/interface.py:600` и `prepare_k2q_csr.py`:
   public input CSR, BF16 D128, G1/G2/G4/G8/G16, final O после Combine.
   Checkout только читался; чистота всего дерева не утверждается и его код
   не включается в нашу сборку.
4. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html):
   tcgen05 MMA, TMEM load/store и ordering между потоками/proxies.

## Что заимствовано из собственного проекта

`codex_teaching/kernels/01_selected_blocks.cu`, root `common.cuh` и
`smem_layout_swizzle128.cuh` — источник уже применявшихся SM100 primitives и
128×128 layouts. Их происхождение — dense teaching progression
[Iaroslav Elistratov](https://github.com/IaroslavElistratov/b200-attention).

Новый вклад этой ступени: per-token CPU metadata, GQA-aware gathered row mapping,
KV-owned CTA, partial-slot addressing, самостоятельный Combine и проверки.
Не заявляются новые MMA primitives, изобретение KV-centric или рекорд скорости.
Существующие kernels и headers не меняем и не делаем runtime-зависимостью v0.
