# Data and methods

This document describes how text becomes a reproducible experiment in MASHeart. Read the [protocol](RESEARCH_PROTOCOL.md) for decisions fixed before the final test, and [the walkthrough](STEP_BY_STEP.md) for commands.

## 1. Data origin and meaning

The statement source is the corpus of potentially contradictory cardiovascular research claims by **Abdulaziz Alamri and Mark Stevenson (2016)**. It contains 259 statements from 24 review questions, 254 distinct PMIDs and 255 distinct exact texts. The questions concern cardiovascular topics and related outcomes, including some pregnancy, renal and metabolic contexts. It is a small, selected research corpus rather than a representative sample of all cardiology language. See the [PCC publication](https://doi.org/10.1186/s13326-016-0083-z).

**Youngwoo Kim, Razieh Rahimi and James Allan (2023)** used related source statements for BioClaim and conditional natural language inference. The inherited linked split follows the released BioClaim membership, not an independently invented split. The [Cond-NLI paper](https://aclanthology.org/2023.findings-emnlp.456/) explains the pairwise task; this project uses the related **question-conditioned stance task**, not its token extraction benchmark.

| Label | Meaning in this project | What it does not establish |
|---|---|---|
| `YS` | The statement affirms the proposition asked by the question, including a tentative affirmative finding | Clinical truth, treatment efficacy for a patient, high evidence quality |
| `NO` | The statement negates or does not affirm the asked proposition | A guaranteed factual contradiction or misinformation label |
| `INVALID` | No usable prediction under the declared parsing/gate policy | A third gold class |

A statement about sleep does not affirm a question about blood pressure just because both mention the same intervention. A negation about a different outcome should not automatically determine the label. The model must read the question and statement together.

## 2. Record fields and inference boundary

| Field | Use |
|---|---|
| `question`, `text` | The only public model inputs |
| `label` | Supervised target during training; evaluation target after prediction |
| `id` | Join predictions, audit duplicates and checkpoint runs |
| `group_id` | Split linked sources and resample uncertainty by source group |
| `pmid`, `review_id`, `source_url` | Provenance and overlap audits |
| `review_title`, `review_no`, `claim_type`, `split`, `source` | Audit metadata; excluded from features and prompts |

Gold and metadata must not enter the prompt by accidentally serializing an entire row. Changing a record's gold label must not change its public payload. PMID is an identifier, not a model feature.

Original statement text is retained for quotation checks. A separate case/whitespace-normalized key detects duplicates. Do not remove negation, doses, units, numbers or time expressions in preprocessing. No translation or external retrieval is required by the method.

## 3. Split construction and audit

The inherited development set has 125 statements in 11 linked groups; the official test has 134 in 10 groups. No independent training partition exists in the inherited protocol. MASHeart splits development with `GroupShuffleSplit`, seed 17, group validation fraction 0.25:

| Partition | Statements | Groups | NO | YS |
|---|---:|---:|---:|---:|
| Training | 95 | 8 | 32 | 63 |
| Validation | 30 | 3 | 9 | 21 |
| Official test | 134 | 10 | 38 | 96 |

The original linked groups were formed conservatively using repeated PMIDs, duplicate normalized statements, paired source links and similar review questions. Holding a whole group together reduces the risk of teaching a model one version of a source and evaluating it on another. Group labels are audit metadata, not evidence of clinical equivalence.

Integrity verification checks file byte counts and fingerprints without parsing JSONL labels. Table loading checks schema, known labels, unique IDs and nonempty text; split checks inspect intersections of groups, PMIDs and normalized statements. Test table loading and its overlap check occur after choices are frozen. Any source overlap is a failure to investigate, not a reason to quietly move a difficult row. Fit all learned preprocessing on training examples only. Validation and test use the learned transformation.

## 4. CPU controls

The majority control predicts the most frequent label in the 95 training statements. It exposes class imbalance: accuracy alone may look reasonable even when one class is never recognized.

The learned baseline is one `Pipeline`:

```text
question + statement
       │
       ├── word TF-IDF: unigrams and bigrams ─────┐
       └── char_wb TF-IDF: 3–5 character grams ──┤ FeatureUnion
                                               ↓
                            one Logistic Regression classifier
```

TF-IDF emphasizes features that help distinguish texts in the training corpus. With smoothing, scikit-learn's IDF is `log((1+n)/(1+df)) + 1`; the complete vectorizer also applies its declared term-frequency and normalization settings. Word bigrams preserve some local phrases such as “not reduce”. Character features capture local spelling patterns. Neither representation guarantees semantic understanding.

The binary classifier computes a weighted score `z = w·x + b` and converts it with a sigmoid to the second class's probability. The implementation records its class order rather than guessing a column's meaning. Training optimizes a regularized objective with `C=1`, `class_weight="balanced"`, a 2,000-iteration cap and seed 17. `max_iter` is an optimizer limit, not an LLM epoch count.

There are two feature blocks but **only one classifier**. There is no word/character vote, two-model probability average, CPU MAS or CPU abstention gate. Coefficients show associations learned in this corpus; they do not identify medical causes. Probability output has not been calibrated as a clinical probability.

## 5. T4 QLoRA training

### Model identity and memory

Use `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Its [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) describes this checkpoint; it differs from Qwen3-8B in the earlier study. The authoritative executable parameters are in [configs/t4.json](../configs/t4.json).

QLoRA keeps a quantized, frozen pretrained base and trains small low-rank adapters. The base is loaded in NF4 with double quantization; T4 computation uses FP16. Not every tensor, activation or optimizer state becomes 4-bit. Gradient checkpointing recomputes some activations to save memory. Gradient accumulation collects gradients across micro-batches before an update. These techniques trade memory, compute and batch behavior; they do not create additional independent examples. See [QLoRA](https://arxiv.org/abs/2305.14314) and [PEFT's quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).

LoRA represents an update to a large matrix through two smaller matrices. For input/output dimensions `d_in,d_out` and rank `r`, an adapter has approximately `r*(d_in+d_out)` matrix parameters instead of `d_in*d_out` for a full update of that matrix. This is a parameter-count explanation, not a promise of task accuracy. See [LoRA](https://arxiv.org/abs/2106.09685).

| Setting | Fixed value |
|---|---|
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Epochs / learning rate | 3 / `1e-4` |
| Micro-batch / accumulation | 1 / 8 |
| Maximum training sequence | 512 tokens; oversize inputs raise an error |
| Selection | Lowest validation loss among saved epoch checkpoints |
| Inference | Greedy; maximum input 1,536 and new output 192 tokens |

Only 95 training statements update the adapter. Validation evaluates checkpoints; test never updates weights. The tokenizer's subword tokens differ from whitespace words used in descriptive text counts.

### Supervised target and masking

A training sequence consists of a chat-formatted question/statement prompt followed by JSON containing only `label` and an end token. Prompt and padding positions receive the ignored target value `-100`. The loss therefore trains the response, not reproduction of the prompt. `attention_mask` distinguishes real tokens from padding and has a different purpose.

The dataset does not provide gold rationales or quotations for this task. MASHeart does not manufacture rationale targets and present them as expert annotation. Validation loss is a token prediction loss, not classification macro-F1.

### Two selection stages

First select the adapter checkpoint with minimum validation loss. Then generate Direct predictions from the frozen base and that adapter on the same 30 validation statements. Select the adapter for Analyst only if its macro-F1 is strictly greater; select base on a tie. Do not refit on validation or choose prompts using final-test scores. Small validation groups and repeated selection can make validation estimates optimistic.

## 6. Roles, packets and source gate

| Role | Active weights | Information received | Output |
|---|---|---|---|
| Analyst | Base or adapter, selected using validation | Question and statement | JSON stance label |
| Reviewer | Base instruct; adapter disabled | Source and Analyst draft | Label, exact quote, short rationale |
| Verifier | Base instruct; adapter disabled | Source, initial and review | Label, exact quote, short rationale |
| Program gate | No learned weights | Parsed packets and structural validity flags | Applied label, review flag, gate status |

These are three role calls on a shared pretrained checkpoint. Reviewer/Verifier do not use the label-only adapter. Role agreement is correlated evidence, not independent specialist judgment.

The parser accepts only the declared stance labels and JSON structure. A source quote is valid only if it is a nonempty string occurring **exactly once, verbatim in the statement**. No fuzzy matching, paraphrase repair or gold-based correction is allowed. Exact quotation establishes lexical provenance; it does not establish entailment, relevance or clinical correctness.

The simplified gate follows this policy:

| Situation | Applied result |
|---|---|
| Initial invalid; final has valid label and quote | Recover final, mark review |
| Initial valid; valid final keeps its label | Keep initial; mark review if Reviewer evidence is invalid or Reviewer disagrees |
| Initial valid; Reviewer and Verifier agree on a new label and both quotes are valid | Apply the sourced revision |
| Initial valid; conditions for a change/final are not met | Keep initial and mark review |
| No usable initial or source-valid recovery | `INVALID`, mark review |

This declared initial fallback differs from the original full v4 protocol. It must be visible in the trace. Even an accepted revision may be wrong; the structural gate is not a medical reasoning oracle.

## 7. Evaluation and reproducibility

Full-test scoring retains 134 gold rows. A prediction failure is counted, not filtered away. Compute class-specific precision, recall and F1 and macro-average the two gold classes; use zero for undefined precision/recall terms. Accuracy and valid rate are separate measures. Include an INVALID prediction column in confusion matrices when needed.

Coverage is the fraction of MAS rows with `review=False`. On the selected subset, score Direct and MAS with the same mask and show the accepted denominator. Review flags do not remove predictions from full-test scoring. If no row is accepted, selective scores are undefined.

Direct is the draft Analyst label from the MAS trace, requiring one call; MAS uses three. Preserve raw generation, parsed packets, gate decisions, tokens, errors and timing. Compare same input and selected Analyst mode. The comparison adds both review and compute and does not isolate a unique cause of any difference.

Paired cluster bootstrap resamples whole test groups with replacement, using identical samples for both methods. Report the seed, 500-draw classroom setting and the small number of groups. Synthetic gate checks test software behavior only; they are not model predictions.

Run artifacts should preserve exact split IDs, data hashes, model revision, adapter hash, parameters, prompts, dependency versions and execution status. Cache reuse must match protocol and ordered input content; failure records must not be silently retried to improve scores. Export figures as PNG and SVG with supporting CSV tables, so a report is inspectable without rerunning a GPU.

## 8. Attribution and permitted interpretation

Dataset attribution and license notices are in [LICENSE_DATA.md](../LICENSE_DATA.md); dataset and model licenses remain separate from the project's software license. The paper's publication license must not be substituted for the corpus license.

The project's measurements concern English question-conditioned stance in this fixed small corpus. They do not establish competence on patient records, Vietnamese text, new medical literature or treatment decisions. Fine-tuning and multi-agent review are experimental methods whose effects must be measured, including harmful changes and extra cost.
