# MASHeart CPU research report

This report records an executed educational Task-A experiment. It predicts the stance of a supplied English statement toward a supplied research question. YS means affirmation; NO means negation or lack of affirmation. Neither label establishes clinical truth, diagnostic accuracy or treatment quality.

## Data and protocol

PCC statements and stance labels originate with Abdulaziz Alamri and Mark Stevenson (2016). Official development/test membership follows the linked BioClaim release preserved in the data manifest. The original release has no independent training set. This run uses 95 training examples in 8 groups and 30 validation examples in 3 groups, both drawn only from official development; all 134 official test examples in 10 groups are retained.

The training and validation split is by linked source group with seed 17. ID, group, PMID and normalized-statement intersections are empty across these partitions and test. The complete source audit and split IDs are exported. The original project used development data during method design and an earlier v1 encountered test; repackaging does not create a historically unseen holdout.

Only question and statement are model features. Word TF-IDF (1–2 grams) and character-within-word TF-IDF (3–5 grams) are fitted inside a Pipeline on training only. Balanced logistic regression uses C=1.0 and max_iter=2000. The majority control learns its class from training labels only. There is no hyperparameter search or full-development refit. Both serialized models were reloaded and checked on validation. Configuration, source checksums, model hashes, environment and split IDs were written before official test records were parsed.

Frozen protocol SHA-256: `f51a607e50f1acde3efcc17e7686ce7850811e504b518fc9826f4ff50611b5f8`.

## Complete-test results

| System | n | Accuracy | Macro-F1 | Recall NO | Recall YS | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| majority | 134 | 0.7164 | 0.4174 | 0.0000 | 1.0000 | 0 |
| tfidf_logreg | 134 | 0.7388 | 0.5629 | 0.1842 | 0.9583 | 0 |

Accuracy uses every example. Macro-F1 always averages NO and YS. INVALID outputs, if present, count as errors rather than being discarded. Per-class precision, recall, F1 and support, confusion counts, predictions with gold, source-group scores and all 35 TF-IDF errors are available in tables. Scores from a different Qwen model or from the original HeartMAS study are not used here.

## Uncertainty and interpretation

The paired bootstrap resamples whole test source groups, gives both systems identical sampled rows and retains both labels in every macro average. Across 500 draws, the observed macro-F1 difference (TF-IDF minus majority) is 0.1456, with a 95% percentile interval [0.0000, 0.2416]. The percentile interval includes zero, so this resampling comparison does not establish a clear improvement.

The majority control's test accuracy is 0.7164; its minority-class recall must be considered alongside that score. The trained baseline achieves accuracy 0.7388 and macro-F1 0.5629. Question wording, negation scope, population, intervention, outcome and study conditions can all matter; a word's fitted coefficient or an uncalibrated score is not medical evidence.

## Artifacts and limits

Eight measured figures are exported as 300-dpi PNG and vector SVG, with exact CSV inputs and hashes in `../figures/figure_data_manifest.json`. Model loading was verified; source and frozen model hashes are preserved. Environment versions and source-code hashes document the implementation used in this run. Training took 0.153 seconds in this environment; timing varies by machine.

This CPU workflow does not train a language model or execute the GPU MAS. It establishes measured controls for the separate QLoRA/MAS workflow. No GPU score, model-call cost, selective coverage or claimed MAS advantage is fabricated. Dataset size, class imbalance, related observations, limited source groups, historical data exposure and corpus annotation choices restrict generalization. The data license and upstream version discrepancy are documented in the data card. Use this experiment for education and text-method research, not treatment decisions.
