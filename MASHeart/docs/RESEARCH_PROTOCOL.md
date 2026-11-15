# MASHeart research protocol

This protocol describes the standalone MASHeart experiment. It defines the question, data boundaries, model selection, comparison and reporting rules before the final test is evaluated. The companion [methods](DATA_AND_METHODS.md) document explains the implementation; [results and limitations](RESULTS_AND_LIMITATIONS.md) distinguishes measured results from work that has not run.

## 1. Research question

Does a three-stage, source-checked workflow improve the classification of a biomedical statement's **stance toward a supplied research question**, compared with the same Analyst's direct answer? What additional inference cost and failure modes does review introduce?

The target is `YS` when the statement affirms the proposition asked by the question, and `NO` when it negates **or does not affirm** that proposition. These labels measure a relation between two texts. They do not label the medical claim as clinically true or false, grade the underlying study, or establish an appropriate treatment.

There is no assumption that fine-tuning or adding agents must improve performance. Reporting a weaker adapter or a harmful revision is a valid experimental result.

## 2. Relationship to the original research

The earlier HeartMAS v4 study used Qwen3-8B with prompt-assigned roles and did not fine-tune on these examples. It included both question-conditioned stance classification, Task A, and pairwise aspect/condition extraction, Task B. MASHeart implements a new, smaller **Task A educational dev-fit** experiment with a CPU baseline and a Qwen3-4B QLoRA path. It does not reproduce the complete v4 question-plan, ledger, patch candidate or revision-gate protocol.

PCC and BioClaim share source statements. Their tasks are linked observations, not independent biomedical benchmarks. The original project has already used development and test data. The present train/validation partition is new for this executable workflow, but it does not make the historical data exposure disappear. No result should be described as a previously unseen community benchmark.

## 3. Frozen data plan

The distributed BioClaim membership determines the inherited split: its `val.csv` maps to development and its `test.csv` maps to test. MASHeart ships the linked PCC Task A records and their provenance. There is no independent training set in the inherited protocol.

| Role in MASHeart | Source | Statements | Linked source groups | NO | YS |
|---|---|---:|---:|---:|---:|
| Train | Group subset of inherited development | 95 | 8 | 32 | 63 |
| Validation | Other group subset of inherited development | 30 | 3 | 9 | 21 |
| Final test | Inherited official test | 134 | 10 | 38 | 96 |

`GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=17)` operates on development `group_id`. Its fraction refers to groups, not an exact fraction of rows. The resulting IDs must be recorded in the run artifacts. Only the 95 training examples may fit TF-IDF, Logistic Regression or LoRA parameters. There is no final refit on all development examples.

Verify source bytes/hashes before fitting. Validate development fields, labels, IDs and train/validation separation when preparing development tables. Load and validate the official test after choices are frozen; its labels must not influence selection. Train, validation and test must not share group IDs, PMIDs or normalized statement text. Normalization for the audit changes case and whitespace only; original text remains available for quotations. Do not substitute a random row split or reuse legacy processed splits from another experiment.

## 4. Methods fixed before testing

### CPU controls

1. A majority control learns the most frequent training label.
2. One Pipeline combines word TF-IDF and character TF-IDF using `FeatureUnion`, followed by **one** balanced Logistic Regression classifier. `C=1`, word n-grams `(1,2)`, character `char_wb` n-grams `(3,5)` and seed 17 are fixed. Both vectorizers fit training data only.

The CPU Pipeline is a classifier baseline, not a multi-agent system. There is no CPU voting ensemble, confidence threshold or review gate.

### T4 training and selection

The model is `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The [T4 configuration](../configs/t4.json) is the executable source of parameters.

- Load the base in 4-bit NF4 with double quantization and FP16 computation.
- Train LoRA rank 8, alpha 16, dropout 0.05 on `q_proj` and `v_proj`.
- Use 3 epochs, learning rate `1e-4`, micro-batch 1, gradient accumulation 8 and a 512-token training limit.
- Keep the base weights frozen; train the adapter on label-only JSON responses. Prompt and padding positions do not contribute to loss.
- Select the saved adapter checkpoint with the lowest validation loss.
- Compare base and selected adapter on the same 30 validation statements using macro-F1. Use the adapter for the Analyst only if its macro-F1 is strictly higher; a tie selects the base.

Reviewer and Verifier always use the base instruct model with the adapter disabled. The three roles share a pretrained checkpoint, but do not necessarily share the active adapter. The adapter learned a short label response, not an annotated reasoning or quotation task.

Using validation loss for checkpoint selection and validation macro-F1 for Analyst selection is two-stage selection on a small, reused validation set. It is not two independent confirmations. Training cost must still be reported if the adapter is not selected.

## 5. Inference and comparison

The model receives only `question` and `text`; gold labels, IDs, PMIDs and group metadata do not enter the inference prompt. IDs remain available to the experiment runner for checkpointing and evaluation.

The Analyst returns its draft label. The Reviewer reads the source and draft; the Verifier reads the source and preceding proposals. The source gate decides whether to apply a revision using the fixed structural policy in [methods](DATA_AND_METHODS.md). The workflow performs three role calls, without hidden retries to obtain a preferred answer.

**Direct** is the Analyst draft already present in the MAS trace. It is not a separate, fourth model call. Direct and MAS therefore share the input and selected Analyst mode. Direct uses one role's computation; MAS adds two calls and a gate. This comparison changes workflow and compute together and does not isolate a single causal mechanism.

Freeze configuration, prompts, model revision, selected Analyst mode, adapter hash, training/validation IDs, generation settings and dependency versions before the final test. Cache reuse requires matching protocol and input identity/order. Preserve failures in caches; do not silently retry or delete them.

## 6. Outcomes and failure accounting

The primary classification outcome is **macro-F1 over the fixed labels NO and YS**. The primary T4 comparison is `MAS macro-F1 − Direct macro-F1` on all 134 test statements. Accuracy, per-class precision/recall/F1, support, valid-answer rate, confusion matrices, review flags, token counts and elapsed inference time provide additional context.

A malformed or unavailable answer remains `INVALID` when no usable label can be obtained under the declared gate. INVALID counts as an incorrect prediction and a missed true class. Do not discard it from the denominator or repair it using gold. The source gate's declared fallback to an initial label is part of the method and must remain visible in traces.

`review=True` means a person should inspect the result; it does not remove the label from full-test scoring. The selective table uses the mask `review=False` from MAS and applies **that same mask to Direct and MAS**. Report accepted count, total count and coverage beside selective metrics. If there are no accepted samples, selective metrics are undefined. Do not compare a selected subset's accuracy with another method's full-test accuracy as if the denominators matched.

## 7. Uncertainty and interpretation

Use paired cluster bootstrap by `group_id`: draw groups with replacement, retain all rows from each sampled group, and use the same sampled rows for both methods. The classroom/reference setting uses 500 draws with a recorded seed. The 2.5th and 97.5th percentiles describe an illustrative 95% interval for the macro-F1 difference.

Only 10 test groups are available, so the interval is uncertain and may be unstable. It describes variability under this resampling plan, not uncertainty across pretraining runs or all possible clinical topics. A confidence interval containing zero does not establish equivalence, and a positive point estimate alone does not establish superiority.

Read errors in their original question/statement context. Count or inspect both beneficial and harmful label changes. An exact quotation can still be irrelevant to the requested outcome or misinterpreted. Agreement among shared-checkpoint roles is not independent verification.

## 8. Execution and reporting record

The canonical CPU reference is stored under `results/cpu_reference/`; the authoritative numeric table is [test_metrics.csv](../results/cpu_reference/tables/test_metrics.csv). GPU execution status is recorded in [STATUS.json](../results/gpu/STATUS.json). The local development environment has no CUDA execution evidence, so GPU training and evaluation must be recorded as **NOT RUN** until a real T4 run produces artifacts.

CPU tests, synthetic parser/gate checks and notebook validation establish software properties. They do not establish GPU training success or model quality. Do not fill GPU tables with estimates, original v4 scores or mock outputs.

After inspecting test errors, any model, prompt or gate change begins a new development cycle. Keep the old run, increment the protocol/run identity and disclose test exposure. Use a new evaluation set for new confirmatory claims.

## 9. Minimum material for a report

Include the stance definition, source attribution, split counts and group policy; fixed configurations and both selection stages; full-test metrics with failures; matched-mask selective metrics; paired uncertainty, cost and error examples; execution status, environment and historical data exposure. State whether the claim concerns this small fixed dataset, a software check or a proposed experiment. Do not generalize it to clinical truth.
