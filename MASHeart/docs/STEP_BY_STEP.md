# Step-by-step research workflow

This walkthrough starts from the standalone **MASHeart** folder. It does not require the parent HeartMAS repository. The commands use the Python environment you activate; run them from the project root containing `pyproject.toml`, `configs`, `data`, `docs` and `scripts`.

The outcome is a traceable experiment: audit → group split → training → validation → frozen choices → final test → error analysis → saved artifacts → a new-input demonstration. Read [the protocol](RESEARCH_PROTOCOL.md) before changing the method.

## 1. Create an isolated environment

Use a supported Python version from the project metadata. From the folder containing MASHeart:

```text
cd MASHeart
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the CPU requirements. An editable package installation is convenient but optional when running `python -m masheart` from this project root:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable installation points Python to this project's source. A virtual environment keeps its dependencies separate from other work. The default CPU workflow does not download a language model. Optional development checks use `requirements-dev.txt`.

## 2. Audit data before training

```text
python -m masheart audit
```

Inspect the audit output. This command checks the manifest, source byte counts and SHA-256 fingerprints **without parsing JSONL labels**. Schema, known labels, IDs and source separation are checked when the relevant tables are loaded by setup/training/evaluation. The declared development group split is 95 train/30 validation; the official test contains 134 statements.

Read [DATA_AND_METHODS.md](DATA_AND_METHODS.md) alongside the output. `question` and `text` are the only model inputs; `label` is the training/evaluation target. `group_id`, PMID and IDs support the audit and must not become predictors. Do not replace the grouped split with a random row split to improve a score.

Prepare inspectable training/validation files in a new output folder:

```text
python -m masheart setup-data --output runs/data_setup
```

This command verifies staged raw sources, derives the grouped training/validation partition from development and writes `train.jsonl`, `validation.jsonl` and `split_manifest.json`. It does not export final-test labels into the prepared training files. The full experiment independently verifies its own source files and split; this setup step makes that preparation visible for teaching.

## 3. Run the complete CPU experiment

```text
python -m masheart run --mode cpu --output results/my_cpu_run
```

This is a real supervised experiment, not a simulated MAS. The run fits the majority control and one word/character TF-IDF + Logistic Regression Pipeline on 95 training examples. It transforms validation and test using training-only vectorizers, records validation metrics, freezes its choices and scores all 134 test examples.

Use a new output directory for a different experiment; preserve the reference run. Inspect the output's tables, figures, model and run metadata. In the distributed reference, the key table is [results/cpu_reference/tables/test_metrics.csv](../results/cpu_reference/tables/test_metrics.csv), with figures in `results/cpu_reference/figures/` and supporting CSV data in tables.

The CPU model has two feature blocks but one classifier. Inspecting its word/character coefficients helps explain learned associations, not medical causes. No confidence gate or multi-agent roles run in CPU mode.

The recorded CPU run contains `models/tfidf_logreg.joblib`, `models/majority.joblib`, `run_summary.json`, a complete manifest, `frozen_protocol.json` and `reports/CPU_RESEARCH_REPORT.md`. Its tables include `test_metrics.csv`, `test_predictions.csv`, `validation_metrics.csv`, `split_summary.csv`, `test_group_scores.csv`, `test_errors.csv`, `test_per_class_metrics.csv`, `test_confusion_counts.csv` and `bootstrap_differences.csv`, alongside plot-source CSVs. Eight scientific figures are exported as both PNG and SVG.

## 4. Read validation before the final-test interpretation

Ask whether the learned model improves on the majority control, how NO recall differs from YS recall, and whether validation topics differ from training topics. Accuracy alone can conceal a neglected minority class.

Training is learning parameters; validation is model/checkpoint selection; final test is the fixed final comparison. If you change features after reading final-test errors, record a new exploratory development cycle. Do not keep calling the reused test untouched.

## 5. Prepare the T4 experiment

The QLoRA path needs a compatible CUDA environment, normally a Colab T4. Install the project's optional GPU requirements in that environment; this file also includes the CPU requirements:

```text
python -m pip install -r requirements-t4.txt
```

Keep the CUDA-compatible PyTorch supplied by Colab unless the environment setup explicitly requires another build. Read [configs/t4.json](../configs/t4.json): pinned Qwen revision, NF4/FP16, LoRA rank 8 on q/v, 3 epochs, learning rate `1e-4`, micro-batch 1, accumulation 8 and 512-token training limit. Model download and GPU time depend on the actual runtime; no measured GPU speed or score is implied by these settings.

## 6. Run QLoRA and the source-checked MAS

```text
python -m masheart run --mode t4 --config configs/t4.json --output results/my_t4_run
```

The intended sequence is:

1. Load one quantized pretrained checkpoint and attach trainable LoRA parameters.
2. Encode 95 training and 30 validation examples. Prompt/padding target positions are masked with `-100`; only JSON label-response tokens contribute to loss.
3. Train the adapter; choose the saved checkpoint with minimum validation loss.
4. Compare base versus adapter Direct predictions on the same validation set. Higher macro-F1 chooses the Analyst mode; a tie keeps base.
5. Use base instruct for Reviewer and Verifier with the adapter disabled. Read validation traces before proceeding with the frozen protocol.
6. Freeze model/configuration/prompt/adapter and split fingerprints, then evaluate the official 134-row test.
7. Save raw responses, parsed packets, gate outcomes, errors, token counts, time and result tables.

A poor adapter is a possible outcome, not a reason to hide the comparison. Its training cost remains part of the experiment even if Analyst uses base afterward. A GPU exception is not a valid numeric model result. The supplied [GPU status](../results/gpu/STATUS.json) records the local unexecuted T4 status until a real run exists.

## 7. Interpret Direct, MAS and review flags

Direct is the first Analyst draft in the MAS trace: one role call. MAS uses the Analyst, Reviewer and Verifier plus a structural quote gate. A change to a valid initial label requires Reviewer and Verifier to agree on the new label and provide nonempty, unique, verbatim statement quotations.

The gate may retain a draft and mark `review=True`. This flag requests inspection; it does not remove the row from the full-test table. When no usable answer exists, INVALID counts as an error. Quote validity confirms source wording, not correctness of the reasoning.

Compare Direct and MAS on all test rows first. Then inspect the selective table using the **same** `review=False` mask for both. Read accepted count and coverage, not only the selected accuracy. Inspect beneficial and harmful revisions and the cost of the extra calls.

## 8. Read plots and their source tables

Use confusion matrices to locate NO→YS and YS→NO errors; include INVALID where present. Read per-class precision/recall/F1 and the fixed two-class macro average. Use group summaries to identify source groups that are hard for the model.

Paired cluster bootstrap resamples entire source groups with replacement and keeps both methods paired. Only 10 test groups exist, so the 500-draw classroom interval is an illustration with substantial uncertainty. It does not create new independent studies.

PNG figures are convenient for slides; SVG figures remain scalable for reports. Keep the CSV source tables with exported graphics. A polished plot does not replace a complete denominator or a run record.

Regenerate the scientific figures from a completed CPU run:

```text
python -m masheart visualize --run-dir results/my_cpu_run
```

This command validates the completed run's artifacts and regenerates figures from its recorded evidence; it does not refit a model or substitute a fresh test experiment.

## 9. Load the CPU model and try a new question/statement

```text
python -m masheart predict --model results/cpu_reference/models/tfidf_logreg.joblib --question "Does treatment X reduce outcome Y?" --statement "The synthetic study did not report a reduction in outcome Y."
```

This is a synthetic language example, not a medical finding. The saved Pipeline includes both vectorizers and classifier; load artifacts created by this workflow or another trusted source. A demonstration without an annotated gold label cannot produce a meaningful accuracy score.

Try changing only the asked outcome or negation and explain the behavior. The corpus is English; performance on another language is not established by translating the interface.

## 10. Verify the project and retain evidence

```text
python scripts/verify_project.py
```

The verifier checks project/data/artifact consistency as implemented. Passing software checks does not mean the GPU experiment ran or that a model is medically correct. Keep source hashes, exact split IDs, configuration, environment, model files, raw traces, tables and figures for each reported run.

## 11. Use the standalone Colab notebook

Upload [notebooks/MASHeart_Step_by_Step.ipynb](../notebooks/MASHeart_Step_by_Step.ipynb) through Colab's **File → Upload notebook**. The English notebook embeds the required project assets and bootstraps them into its runtime; it does not require a checkout of the parent research repository. Choose T4 for the QLoRA/MAS path and run cells in order.

The companion [notebooks/01_Data_Exploration.ipynb](../notebooks/01_Data_Exploration.ipynb) focuses on the real data, provenance, grouped split and CPU visualizations. Use it for an initial data exploration lesson; use the main notebook for the full training/evaluation/MAS workflow.

Read each cell's input, output and function explanation before running it. Without CUDA, only the supported CPU steps are actual measurements; GPU outputs remain unexecuted. Download the result archive before a temporary runtime disappears. Resume/cache behavior depends on preserving the matching configuration and artifacts; runtime storage is not automatically persistent cloud storage.

For a one-hour lesson, use [TEACHING_GUIDE_60MIN.md](TEACHING_GUIDE_60MIN.md). For interpretation, use [RESULTS_AND_LIMITATIONS.md](RESULTS_AND_LIMITATIONS.md).
