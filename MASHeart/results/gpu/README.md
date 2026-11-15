# GPU execution status

The T4 training and MAS inference pipeline is implemented in `masheart/gpu.py` and configured in `configs/t4.json`. It has not been executed on a CUDA GPU on the packaging host. `STATUS.json` records that limitation; there are no fabricated GPU scores or empty charts presented as results.

On Colab T4, follow the main notebook or run from the project root:

```bash
python -m pip install -r requirements-t4.txt
python -m masheart run --mode t4 --config configs/t4.json --output results/my_t4_run
```

Use a new output directory. Keep the resulting adapter, protocol, validation/test tables, raw traces, cost records, and report together. A CPU logical test is not a T4 training or inference benchmark.
