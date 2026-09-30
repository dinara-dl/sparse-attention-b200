# KV-centric round7 — профиль и атрибуция времени, 20 сентября 2026

Раунд ничего не ускоряет: он отвечает, **куда уходит время `rawmax8`**, двумя
независимыми способами — абляцией (убираем компонент, смотрим, сколько
времени исчезло) и Nsight Compute. Оба сошлись на одном: main-kernel
latency-bound при occupancy 12.5%, и самый дорогой участок на критическом
пути — epilogue записи FP32 partials, а не MMA, не MUFU и не полоса HBM.
Combine — чисто HBM-bound.

## 1. Абляционная атрибуция

Форма `large_g16` (B8/Hq16/Hkv1/4096/4096, seed 5501) и `q8k`
(B1/Hq16/Sq=Skv=8192, seed 5505). Все варианты — сборки rawmax8 с одним
компонентом, убранным на этапе компиляции (`if constexpr`); барьеры и
commit'ы убранных компонентов остаются. `diag0` (та же сборка, ничего не
убрано) побитно равен rawmax8 по partial/final O/LSE, rawmax8 побитно равен
overlap8 — обе проверки прошли на обеих формах. Разницы **не аддитивны**:
`rawmax8 − no_X` — маргинальная стоимость X на критическом пути.

| Вариант (main, ms) | large_g16 | Δ | q8k | Δ |
|---|---:|---:|---:|---:|
| overlap8 (round5) | 2.221633 | +3.1% | 0.577191 | +4.2% |
| **rawmax8** (база) | **2.155552** | — | **0.554007** | — |
| diag0 (контроль сборки) | 2.163607 | +0.4% | 0.557027 | +0.5% |
| no_store — без записи partial O/LSE | 1.459451 | **−32.3%** | 0.356452 | **−35.7%** |
| no_row — без softmax row-path | 1.845278 | −14.4% | 0.508375 | −8.2% |
| no_exp — без MUFU.EX2 | 1.931496 | −10.4% | 0.521626 | −5.8% |
| no_gather — без cp.async Q | 1.967612 | −8.7% | 0.498363 | −10.0% |
| no_pv — без PV MMA | 2.021045 | −6.2% | 0.533434 | −3.7% |
| no_qk — без QK MMA | 2.075074 | −3.7% | 0.534471 | −3.5% |
| mma_gather_only — без row и store | 1.016658 | −52.8% | 0.242624 | −56.2% |
| gather_store_only — без row, QK, PV | 1.666946 | −22.7% | 0.465483 | −16.0% |

Разброс p10–p90 у всех ключей ≤0.5%. Полный оператор rawmax8 в этом
вызове: 2.831766 ms (round6: 2.837), Combine 0.627630, metadata 0.031000.

Чтение: путь «gather + evacuation + store» без единой арифметики занимает
**77% времени main** (84% на q8k); убрать одну только запись partials —
минус треть. Весь row-path — 14%, exp — 10%, оба MMA вместе — под 10%.
Это не «HBM забита»: см. профиль ниже.

SASS подтверждает, что убрано именно то, что заявлено: `no_exp` — 0
MUFU.EX2 (база 128), `no_store` — 0 STG (база 5), `no_gather` — 0 LDGSTS
(база 5), `no_qk`/`no_pv` — 8/16 tcgen05.mma вместо 24. Регистры: база 163,
`no_exp` 180, `no_row`/`gather_store_only` 55–63; occupancy это не меняет
(1 CTA/SM из-за 160 KiB SMEM).

## 2. Nsight Compute (`--set full`, `--clock-control none`)

Форма B2/Hq16/Hkv1/4096/4096 (1 GiB partials; поведение CTA от B не зависит,
но Q здесь целиком помещается в L2, на B8 — нет). Один запуск main и Combine
после двух прогревочных.

| Метрика | main (rawmax8) | Combine |
|---|---:|---:|
| Duration | 562.8 µs | 162.6 µs |
| DRAM throughput, % пика | **27.1** (2.08 TB/s) | **89.5** (6.86 TB/s) |
| DRAM bytes read / write | 0.137 / 1.033 GB | 1.082 / 0.034 GB |
| Compute (SM) throughput / issue slots busy | 39.5% | 16.9% |
| Tensor pipe active | 11.7% | — |
| FMA pipe / XU (MUFU) pipe / LSU pipe | 12.7 / 11.8 / 12.2% | — |
| ALU pipe | 33.7% | — |
| Achieved occupancy | **12.4%** (7.96 warps/SM; лимит — SMEM и регистры: 1 block) | 88.9% |
| Eligible warps per scheduler / issued per cycle | 0.49 / 0.41 | 0.22 / 0.18 |
| Warp cycles per issued instruction | 4.88 | 81.7 |
| L2 hit rate | 22.6% | 0.4% |
| Registers/thread | 163 | 32 |

Стойлы main на одну выданную инструкцию (`smsp__average_warps_issue_stalled_*`):
**barrier 1.13**, **wait 0.99** (зависимости фиксированной латентности),
long_scoreboard 0.66, short_scoreboard 0.36, not_selected 0.20,
no_instruction 0.15, branch_resolving 0.15, math_pipe_throttle 0.12,
**lg_throttle 0.00**, mio_throttle 0.01. Combine: long_scoreboard 75.8 — ждёт
HBM.

Вывод для main: ни один pipe не загружен выше 40%, HBM — на четверть, очередь
LSU никогда не заполнена (`lg_throttle=0`). Время уходит на ожидание у CTA
барьеров и на цепочки зависимостей при двух warps на scheduler. Дорог не
объём записи partials, а **инструкционный путь epilogue** на критическом
пути: 16 × (LDS.128 + STG.128 + 64-битная адресация + LDS query/rank) на
поток, два `__syncthreads`, и всё это при 8 warps на SM, которым нечем это
перекрыть. Вывод для Combine: ускорить его можно только меньшим числом байт.

## 3. Что это меняет в плане

- **Epilogue через bulk-copy (exact)** — новый первый кандидат round8:
  `cp.async.bulk shared→global` по одной инструкции на строку вместо
  цикла из 16 LDS+STG; байты и значения те же → побитный gate против rawmax8
  сохраняется. Нацелен ровно на измеренные 0.70 ms.
- **BF16 partials (precision)** — второй кандидат round8, поверх первого:
  главный ожидаемый эффект в Combine (HBM-bound → ~½ байт), в main —
  вдвое меньше STS/bulk-байт. Требует отдельного precision-gate, не exact.
- **Warp-specialization** — по-прежнему следующий архитектурный раунд, но
  теперь ясно, что она прячет row-path (−0.31 ms маргинально) и барьеры, а
  не MMA; до неё разумно сначала снять epilogue.
- Log2-домен / полиномиальный exp2 — exp даёт 10%, но exp-проход не
  MUFU-bound (XU 11.8%); низкий приоритет.
- 2-CTA MMA, persistent — не нужны (tensor 11.7%, tail ~0.2%).

## 4. Исправление харнеса

`timing7.py`: diagnostic-ключи (`shared/combine`, `control/metadata`)
измеряются после ротации вариантов, settle 60 ms. В этом вызове ни у одного
ключа p10–p90 не превысил 0.5%; артефакт «`*/operator` после combine»
(round5/6) не воспроизводится. Round5/6 не перемерялись; их медианы
устойчивы к артефакту (он затрагивал 2 sample из 5).

## 5. Modal

| Вызов | Назначение | Статус | remote_seconds |
|---|---|---|---:|
| `fc-01M2Z3Z3MTZAKE2JJ36X723694` | компиляция, абляция на двух формах, ncu | passed; ncu упал: «Failed to lock GPU clock frequencies» | 63.104 |
| `fc-01M2Z4P2GG5NH7YEGH3ZCBNXS2` | ncu с `--clock-control none` (только профиль) | passed | 18.125 |

Второй вызов — предусмотренный корректирующий: причина сбоя конкретная
(контейнер не даёт фиксировать частоты), исправление — один флаг, абляция
не повторялась. Сумма 81.2 s remote. `timeout=8 * 60` в обоих.

Артефакты: [run 001](results/run_001_profile.json) (абляция, SASS-гистограммы,
resource usage), [run 002](results/run_002_ncu_only.json) (ncu details и
raw CSV, ~2400 метрик на kernel). Код: [diag7_kernel.cu](diag7_kernel.cu),
[benchmark7.py](benchmark7.py), [timing7.py](timing7.py).

## 6. Ограничения

- Абляция измеряет маргинальную стоимость при удалении, не «долю» — суммы
  превышают 100%, это ожидаемо.
- Диагностические сборки меняют регистры (55–180) и, в `no_row`/`no_store`,
  дают компилятору другой граф; `diag0` показывает, что сам шаблон стоит
  +0.4%.
- ncu снят на B2, не B8: DRAM read занижен (Q в L2). Соотношение стойл и
  утилизация pipes на B8 могут отличаться количественно, не качественно.
- Один seed на форму; это диагностика, не подтверждающий замер.
