# Teaching MASHeart in 60 minutes

This lesson introduces the complete research process through a small biomedical language experiment: define a question, audit data, prevent leakage, train, validate, freeze decisions, evaluate and inspect mistakes. It uses the English [step-by-step notebook](../notebooks/MASHeart_Step_by_Step.ipynb) and the standalone project commands in [STEP_BY_STEP.md](STEP_BY_STEP.md). No fixed cell numbers are assumed.

## Learning goals and preparation

Students should know Python strings, lists, dictionaries, functions and basic ratios. They do not need calculus for this lesson. By the end, they should distinguish stance from clinical truth, explain group separation, identify trainable parameters, read macro-F1 and understand why an exact quote does not guarantee correct reasoning.

Before class, the teacher should run the CPU workflow and inspect its artifacts. If teaching QLoRA/MAS, prepare a real T4 run and preserve its trace, training log and evaluation tables. Model download, training and full three-call evaluation may exceed the lesson duration. Present prepared artifacts as results of that recorded run, not as work that happened instantly in class. If no GPU run exists, teach the proposed GPU method and mark it NOT RUN; never replace missing measurements with plausible numbers.

The fixed split is train95/validation30/test134, separated by linked source groups. The corpus is English and consists of research statements, not individual patient records.

## Minute-by-minute plan

| Minutes | Activity | Evidence of learning |
|---|---|---|
| 0–6 | Define question-conditioned stance with three synthetic statements | Students explain why NO can mean lack of affirmation |
| 6–14 | Inspect data fields and audit; predict what a random row split could leak | Students identify allowed inputs and forbidden metadata |
| 14–23 | Run or inspect the real CPU baseline and majority control | Students distinguish feature blocks from multiple agents |
| 23–33 | Explain QLoRA; inspect trainable parameters, loss masking and model selection | Students identify frozen base versus trained adapter |
| 33–43 | Read a three-role trace and apply the source gate manually | Students separate exact quotation from semantic support |
| 43–53 | Read full-test metrics, matched-mask coverage, errors and cost | Students state both numerator and denominator |
| 53–60 | Compare conclusions and submit a short exit response | Students make one evidence-limited claim and one limitation |

Use precomputed artifacts to keep discussion focused if a CPU/GPU step has not finished. A smoke test on a handful of rows is a software demonstration, not a full-test result.

## 1. Start with the correct target

Use a fictional intervention and outcome to avoid teaching an invented medical fact:

> Question: Does treatment X reduce outcome Y in adults?
>
> A: In this synthetic example, treatment X reduced outcome Y in adults.
>
> B: In this synthetic example, treatment X did not reduce outcome Y in adults.
>
> C: In this synthetic example, treatment X improved a different outcome Z.

A is YS; B and C are NO toward that question. C lacks “not” yet does not affirm outcome Y. Change the question and ask whether a label might change. The label describes the question–statement relation, not whether treatment X works in real medicine.

## 2. Make data boundaries concrete

Ask students to separate three categories of columns:

- **Input:** question and statement.
- **Target:** label, used for fitting or subsequent scoring.
- **Audit metadata:** group, PMID, ID and provenance.

Two rows may come from the same study or repeat the same statement. A random row split can put related evidence on both sides of the evaluation. Group separation keeps linked sources together. The development data itself has a research history, so this is educational dev-fit rather than a newly discovered untouched benchmark.

Explain `fit` as learning from training data and `transform` as using a representation already learned. Fitting TF-IDF on test text leaks distributional information even without reading its labels. Removing “not”, a dose or a time expression can destroy meaning rather than clean noise.

## 3. Explain the CPU baseline accurately

`FeatureUnion` joins word TF-IDF and character TF-IDF features; **one Logistic Regression** learns weights on the combined vector. It is not two classifiers voting and is not the MAS. The majority control gives a necessary reference for class imbalance.

Word bigrams retain some local combinations; character grams capture spelling patterns. A positive coefficient is an association with a label in this fitted model, not proof that a treatment causes an outcome. `predict_proba` is not a calibrated clinical probability. Keep `C=1` fixed in this lesson instead of searching for a favorable test score.

## 4. Explain training and selection on T4

The pretrained Qwen base already contains learned language patterns. QLoRA keeps its quantized weights frozen and trains small LoRA matrices. The lesson trains only 95 examples, not a new language model from scratch. The response target contains JSON with a stance label; prompt and padding positions use target `-100`, so they do not contribute to response loss. The attention mask serves a different purpose.

The declared configuration uses Qwen3-4B-Instruct-2507 at a pinned revision, NF4/FP16, rank 8 on q/v, 3 epochs, learning rate `1e-4`, micro-batch 1, accumulation 8 and maximum training length 512. Students should read actual run metadata rather than assume every run used the same settings.

Checkpoint selection uses minimum validation loss. A second comparison uses macro-F1 on the same 30 validation statements to select base or adapter for Analyst; a tie keeps base. Reviewer and Verifier always use base instruct with the adapter disabled because the adapter learned short label responses. These roles share a pretrained checkpoint but not always the active adapter.

Fine-tuning can hurt. If base wins, report that outcome and retain the training cost. Decreasing training loss is not sufficient evidence of improved generalization.

## 5. Read a MAS trace as a scientific object

Give three students Analyst, Reviewer and Verifier roles, and a fourth the program gate. The gate sees structured packets rather than an expert answer key. Ask:

1. What proposition does the question ask?
2. What label did Analyst choose, and was its adapter active?
3. Did Reviewer propose a change? Does the exact quote occur once in the statement?
4. Did Verifier agree, and what quote did it provide?
5. Under the fixed gate, is the change applied, blocked or marked for review?
6. Could an exact quote still be irrelevant to the requested outcome?

A change to a valid initial label requires Reviewer/Verifier agreement on the new label and valid exact quotes. Otherwise the workflow can retain the draft and mark review. An invalid initial can be recovered only through the declared valid-final policy; otherwise it remains INVALID. These are structural rules, not access to medical truth.

Direct is the Analyst draft inside this same trace, so the direct comparison does not require a fourth call. MAS uses three calls rather than one. The extra compute is part of the comparison, not a free improvement.

## 6. Teach metrics with denominators

For one class, precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and F1 is their harmonic mean. Macro-F1 averages the F1 values of NO and YS equally. Accuracy is correct labels divided by all labels. Ask why always predicting the majority class can make accuracy look better than minority-class recall.

Full-test results include all 134 rows, review flags and INVALID predictions. Valid rate measures legal labels. Coverage measures the fraction with `review=False`; these are different quantities. The selected-subset Direct/MAS comparison uses the **same** mask and shows accepted count. A higher subset accuracy does not automatically mean the model improved on all questions.

The bootstrap resamples whole groups, using the same sampled sources for both methods. Ten test groups provide limited independent information; 500 resamples do not manufacture 500 new studies. A confidence interval containing zero does not prove equal methods, and a one-point score difference alone does not prove superiority.

## Ten questions with expected answers

1. **Does YS mean clinically true?** No; it means affirmation of the proposition asked by the question.
2. **Can a statement without “not” be NO?** Yes; it may not address or affirm the asked outcome.
3. **Why split groups instead of rows?** Related or repeated sources otherwise cross the evaluation boundary.
4. **Can TF-IDF fit on test text if labels are hidden?** No; its vocabulary/IDF would learn test distribution information.
5. **Are word and character TF-IDF two agents?** No; they are feature blocks for one CPU classifier.
6. **Does QLoRA train all base weights?** No; the quantized base is frozen and adapter parameters are updated.
7. **Do all three LLM roles use the adapter?** No; only Analyst may use it after validation selection.
8. **Does an exact quote prove the label?** No; wording provenance and semantic support are different.
9. **Does review remove a row from the main score?** No; the label/INVALID remains in full-test scoring.
10. **Can we edit prompts after test errors and call the rerun untouched test?** No; that starts a disclosed development cycle.

## Common misconceptions

| Misconception | Correction |
|---|---|
| NO means false medical information | NO includes lack of affirmation toward this question |
| A “not” rule solves stance | Negation scope and the requested outcome matter |
| Different research results imply one study is wrong | Population, intervention, comparator, outcome and time may differ |
| “Associated with” establishes a cause | Association and causation are different; stance labels do not settle causality |
| Three prompts create independent specialists | Shared weights produce correlated errors |
| A valid quote verifies the reasoning | It verifies exact source occurrence only |
| Fine-tuning must beat base | Measure both on validation and report harm if present |
| A plot or test pass proves GPU performance | Only a recorded model execution produces GPU metrics |
| Selected accuracy is comparable with full-test accuracy | Report the subset denominator and use matched masks |

## Exit task and assessment

Ask each group for a one-page report or a two-minute explanation containing: the target definition; the three split roles; one real measured result with its denominator; one trace or error; and one limitation. Students without a GPU run should label that portion unexecuted rather than fabricate a score.

| Criterion | Points out of 10 |
|---|---:|
| Correct stance definition and scope | 2 |
| Data separation and leakage explanation | 2 |
| Correct CPU/QLoRA/role explanation | 2 |
| Metrics, failures, coverage and cost | 2 |
| Evidence-based error analysis and honest limitations | 2 |

Do not award points for the largest F1. A reproducible run with disappointing performance and a careful explanation can receive full credit. Extension projects should propose a new train/validation plan before testing changes and retain the original evidence.
