# Results, execution evidence and limitations

MASHeart separates executable methods from results actually produced by a run. This file explains how to read the reference artifacts without turning an unexecuted GPU experiment into a reported finding.

## 1. Where the evidence lives

| Artifact | Interpretation |
|---|---|
| [CPU reference test metrics](../results/cpu_reference/tables/test_metrics.csv) | Measured majority and TF-IDF/Logistic Regression classification results from the recorded CPU run |
| `results/cpu_reference/tables/` | Machine-readable predictions, summaries and source tables for plots |
| `results/cpu_reference/figures/` | PNG and SVG figures generated from the recorded run, with supporting source CSVs in tables |
| `results/cpu_reference/models/tfidf_logreg.joblib` | Fitted training-only Pipeline; contains vectorizers and classifier together |
| `results/cpu_reference/reports/CPU_RESEARCH_REPORT.md` | Generated scientific report from the same recorded CPU evidence |
| [GPU status](../results/gpu/STATUS.json) | Whether real QLoRA training/inference/evaluation has run |

The CPU reference table is the authority for numeric CPU scores; this document does not maintain a second manually copied score table. Use the recorded `n`, metric definitions and environment when comparing it with a new run. The generated report, `run_summary.json`, complete manifest and frozen protocol provide context. Eight PNG/SVG figure pairs are backed by CSV source tables. Data counts—95 train, 30 validation, 134 test—describe the fixed split and are not performance estimates.

The local development environment has no CUDA execution evidence. The T4 experiment is **NOT RUN locally** until a real GPU execution is recorded. No GPU accuracy, macro-F1, loss, VRAM, speedup or improvement is supplied from estimates. Parser tests, generated notebook cells and synthetic gate cases do not establish successful GPU training. Original HeartMAS v4 results are not MASHeart results.

## 2. Read the full-test table first

Every full-test method is scored on all 134 statements. A model that cannot produce a usable answer retains an INVALID prediction and is counted as wrong. Review flags remain in the full-test denominator.

Start with majority versus the single CPU Pipeline. If the learned model does not beat the majority control on macro-F1, report that outcome. A majority model may have respectable accuracy while failing to recognize NO. Inspect per-class recall and support before interpreting an overall score.

When a complete T4 run exists, compare Direct and MAS on the same test IDs. Direct is the Analyst draft inside each MAS trace, using the same selected Analyst mode. The difference in final labels measures the effect of the extra review stages plus the gate and their extra computation. A larger score alone does not show which component caused the difference.

## 3. Review flags and selected subsets

Coverage is `count(review=False) / 134`. The selected-subset comparison must apply the **same MAS-derived mask** to Direct and MAS. Show accepted count and support with the scores; if the selected subset is empty, its accuracy and F1 are undefined.

A review flag does not mean the system abstained from full-test scoring. It may preserve an initial label while asking for inspection. Valid-answer rate is different: it measures how often a legal NO/YS label was returned. A legal label may still need review, and a source-valid quote may still support the wrong interpretation.

Do not compare selected-subset MAS accuracy with full-test CPU accuracy as if the task and denominator matched. Do not hide INVALID rows, blocked changes or initial fallbacks.

## 4. What to inspect in a T4 trace

For each source, inspect initial label, Reviewer proposal, Verifier proposal, source quotations, parser flags, applied label and gate status. Identify whether Analyst used the base or adapter; Reviewer/Verifier use the base instruct checkpoint.

Record both a wrong-to-right change and a right-to-wrong change if they occur. A gate can block a useful correction or accept a harmful one because it tests structure and exact source occurrence rather than semantic entailment. Correlated roles can agree on the same mistake. Failure to find a beneficial example is itself reportable; do not select only attractive demonstrations.

Inference cost includes all role calls, input/output tokens and elapsed time. Training, validation selection and failed generation also consume resources even when the adapter is not selected for Analyst. Separate summed inference durations from wall time where concurrency exists; use the run's actual measurement conventions.

## 5. Uncertainty

Rows from the same source group are related. The illustrative paired cluster bootstrap resamples the 10 groups, not 134 supposedly independent statements. Use identical resampled groups for both predictions and report a macro-F1 difference with its percentile interval.

Only 10 test groups and 3 validation groups are available. Intervals can be unstable, and validation-based model selection can overfit its few topics. One fixed seed and checkpoint do not characterize variation across seeds, training runs or model families. An interval including zero does not prove equivalence; a small positive point difference does not prove a dependable benefit.

## 6. Material limitations

- **Scope of labels:** YS/NO represent stance toward a supplied question, not clinical truth or study quality. NO includes lack of affirmation.
- **Small, selected corpus:** sources were gathered around potentially contradictory research claims; findings need not generalize to all cardiovascular literature.
- **Historical exposure:** inherited development and test data have been used by the original project. New grouping and an executable protocol do not restore an untouched research holdout.
- **Educational dev-fit:** the 95 training examples come from inherited development. There is no independent inherited training benchmark and no final refit on all development rows.
- **Repeated validation selection:** minimum validation loss chooses an adapter checkpoint; validation macro-F1 then chooses the Analyst mode. These are reused observations, not independent confirmations.
- **Baseline semantics:** a word/character feature classifier can learn shortcuts and mishandle negation scope or an outcome change. Its coefficients and probabilities are not causal or clinical evidence.
- **Shared-checkpoint roles:** multiple prompts do not create independent expertise. Analyst may use an adapter, while Reviewer/Verifier use the same base instruct checkpoint.
- **Structural evidence only:** an exact unique quote confirms source wording, not that the wording entails a conclusion. The gate can preserve a wrong draft.
- **Software versus model validation:** audit and parser/gate tests establish data or control-flow properties; they cannot establish QLoRA accuracy or GPU compatibility without an actual run.
- **Language and deployment:** English corpus results do not establish performance on Vietnamese input, patient records, newly published literature or individual decisions.

## 7. How to write a defensible conclusion

Use a statement tied to the evidence: “On this fixed group-separated test, method A had the recorded macro-F1 and cost; the paired comparison did/did not provide clear evidence of a difference under this resampling plan.” Include failures and small-sample caveats. If GPU execution has not occurred, say “The T4 workflow is implemented but has not been empirically evaluated in this environment.”

Do not claim that QLoRA must help, that MAS is intrinsically superior, that a quoted response is medically verified or that an unrun configuration achieved a projected score. Reading test errors and changing the method starts a new development cycle; preserve the old artifacts and disclose the new exposure.

For exact definitions and fixed choices, see [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) and [DATA_AND_METHODS.md](DATA_AND_METHODS.md).
