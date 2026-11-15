# MASHeart verification record

This distribution was checked on a CPU-only Python 3.12 host. The optional T4 pipeline is implemented and logically tested, but no GPU training, VRAM benchmark or Qwen/MAS performance measurement was performed.

| Check | Actual outcome |
|---|---|
| Automated software tests | 35 passed; 0 failures, 0 errors, 0 skipped |
| Source data | Five preserved files verified against byte counts and SHA-256 |
| Feature boundary | Gold labels and source identities excluded from inference payloads |
| Research split | Train 95/8 groups, validation 30/3 groups, test 134/10 groups; source overlap checks passed |
| CPU workflow | Trained, frozen before loading test records, evaluated on all 134 rows |
| Reference model | Saved/reloaded and all test predictions reproduced |
| Figures | Eight measured PNG/SVG pairs with source CSVs and integrity hashes |
| Notebook execution | Both notebooks ran every code cell in separate clean CPU kernels with zero cell errors |
| Standalone notebooks | Embedded source/data checksums match this project's files; no parent repository is imported |
| ZIP portability | Extracted to a new folder, parent PYTHONPATH removed, six CLI commands passed |
| Repeated CPU experiment | All134 prediction rows matched the distributed reference |
| Figure regeneration | Re-rendered successfully while preserving completed CSV/model/report hashes |
| GPU status | Not run; no synthetic fixture is reported as a model result |

## What the tests cover

The suite exercises source verification, path boundaries, source-group splitting, prevention of label leakage, training-only vocabulary, failed-output denominators, paired bootstrap, complete artifact integrity, and saved model prediction equality. GPU-independent cases cover assistant-only loss masking, padding and EOS, strict packet parsing, unique source quotes, the revision gate, matched-mask selective scoring, actual call accounting, frozen runtime changes, interrupted JSONL handling and compatible/incompatible resume behavior.

## Inspect the evidence

- [Full software and notebook report](verification.json)
- [Independent ZIP extraction and CLI report](portability.json)
- [Measured CPU research report](../results/cpu_reference/reports/CPU_RESEARCH_REPORT.md)
- [Complete test measurements](../results/cpu_reference/tables/test_metrics.csv)
- [GPU hardware-execution status](../results/gpu/STATUS.json)

Portability was checked in a new extraction location using the existing validated Python environment, not by claiming a fresh dependency installation. The final ZIP includes these reports and is rechecked for CRC and SHA-256 consistency. Documentation/report additions after the extraction check do not change the verified executable/data payload.
