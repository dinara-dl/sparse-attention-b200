# Анализ round11 KV-centric — 20 сентября 2026

## Исходная точка

Родители — round9 `split8` (exact, 2.278 ms полного оператора на
B8/Hq16/Hkv1/4096/TopK16) и `split8_bf16` (1.681 ms). Профиль round10
показал, что gather Q (2048 `cp.async` с 64-битной адресацией на chunk) и
per-row epilogue (128 bulk-копий по 256 B) стоят столько же инструкций,
сколько весь softmax. У MSA (forward 0.88 ms на той же маске) это дёшево
не из-за pipeline, а из-за layout: Q `[tokens, Hq, D]` и partial O
`[rank, tokens, Hq, D]` держат G голов одного токена контигуозно — chunk из
128/G токенов читается и пишется блоками по G×256 B.

## Изменение контракта

- Q: `[B, Sq, Hq, D]` (bshd; родной формат MSA и inference-движков) вместо
  `[B, Hq, Sq, D]`.
- Partial O: `[B, Sq, 16, Hq, D]` вместо `[B, Hq, Sq, 16, D]`; LSE остаётся
  `[B, Hq, Sq, 16]`, final O — `[B, Hq, Sq, D]`.
- Combine читает по новому адресу, математика та же → final O/LSE побитно
  те же.

## Кандидаты

На базе split9 (две независимые compute-группы), template `<CHUNKS, BF16, TMA>`:

- **Gather через TMA**: 2D tensor map над Q как `[B·Sq·Hq строк, 128]`,
  box `{64, G}` со swizzle 128B. Issuer группы выдаёт 2 TMA на токен
  (128/G токенов → 16 TMA на chunk при G16) и `expect_tx` на суммарные
  байты реальных токенов; `q_ready[g]` — mbarrier с `complete_tx`.
  Ноль адресной арифметики у 128 потоков, данные ложатся swizzled — как
  раньше их писал cp.async вручную.
- **Epilogue через TMA store**: стейджинг в 128B-swizzled layout (панели по
  128 B, `chunk ^ (row&7)` — без конфликтов банков, что проверено
  структурно), `fence.proxy.async` → `bar.sync` группы → issuer выдаёт по
  2 (BF16) или 4 (FP32, двумя проходами) box'а на токен: 16 TMA store на
  chunk вместо 128 bulk-копий. Завершение — `bulk_group` issuer'а,
  `wait_group.read` перед перезаписью стейджинга.
- **G<8**: box в G строк не выровнен на 1 KiB swizzle-атом → TMA не
  используется; остаётся round9-путь (cp.async gather, bulk по строкам) с
  новым partial-индексом. Suite с G1/G2/G4 проходит через него.

Варианты: `tma8` (FP32, exact-родитель split8), `tma8_bf16`, `tma16_bf16`
(родитель split8_bf16). Побитный gate через view `[B,Hq,Sq,16,D]`.

## Ожидание и риски

Инструкции gather/epilogue на chunk падают в разы; compute-группы
освобождаются от адресной работы. Ожидание: main bf16 к ~0.7–0.9 ms.
Риски: (1) неверные координаты/выравнивание TMA → sanitizers или
неверные значения (gate поймает); (2) TMA store из swizzled стейджинга
требует точного соответствия swizzle-формулы — проверено структурно
для банков, на GPU — побитным сравнением; (3) expect_tx с пропуском
padding-токенов — если байты не сойдутся, hang до `timeout=8 * 60`.

## Бюджет

Локально: 8 structural tests. **Вызов 1** (candidates): три варианта,
exact-gates, sanitizers, три tuning shapes. **Вызов 2** (confirmation):
seeds 59xx, 11 форм. Запасной третий — корректирующий. `timeout=8 * 60`.

## Результат (дописано после трёх вызовов)

- **BF16 — подтверждён**: tma8_bf16 −7.8%, tma16_bf16 −10.7% оператора
  (main −10…−15%) на трёх новых seeds; на всех G16-формах −2…−11%.
- **FP32 — нейтрально** (±1.5%): FP32-epilogue упирается в 4 GiB записи,
  TMA меняет инструкции, не байты; попытка убрать ожидание между
  проходами (run 2) ничего не дала.
- Combine token-major на 1–3% медленнее; G1/G4 (fallback) ±1–3%.
- Побитно равны родителям во всех вызовах; sanitizers чистые.

Детали: [RESULTS_RU.md](RESULTS_RU.md).
