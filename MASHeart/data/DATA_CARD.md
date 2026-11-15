# MASHeart Task-A data card

## Intended use and task

This package preserves the Task-A data used by the final HeartMAS v4 release for an English educational research project. A model receives a **research question** and a **supplied scientific statement**, then predicts the statement's stance toward that question:

- **YS:** the statement affirms the proposition asked by the question.
- **NO:** the statement negates or does not affirm that proposition. This includes more than direct negation and must not be renamed a medical falsehood label.

The target is a relation between two texts. Changing the question can change the appropriate label even when the statement stays the same. Negation, outcome, population, intervention, comparison, time and conditions can matter. A keyword rule such as “contains not means NO” is insufficient.

The data do not establish clinical truth, diagnostic accuracy, treatment efficacy for an individual, causal validity or the quality of evidence. Use the project to teach text processing, supervised learning, evaluation, source auditing and multi-agent system design. Do not enter private health records or identifying student information into interactive examples.

## Origin, attribution and preserved sources

Task-A questions, statements and stance annotations originate in the PCC corpus by **Abdulaziz Alamri and Mark Stevenson**, described in [their 2016 publication](https://doi.org/10.1186/s13326-016-0083-z). The recorded original corpus URL is [the University of Sheffield resource](https://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/corpus.xml). Its recorded SHA-256 is `c6925524b0117e877e64782af048d18a851bf0127361abe29587858238a180f4`.

The released development/test membership is linked to [BioClaim / Cond-NLI](https://github.com/youngwoo-umass/cond-nli), revision `14e88c344c148c46bb6dbaab2149b252b5e030d3`, described by [Kim, Rahimi and Allan, 2023](https://aclanthology.org/2023.findings-emnlp.456/). The released `val.csv` determines development membership and `test.csv` determines test membership in the linked HeartMAS protocol. Task A does not use invented Task-B labels.

Every file in `raw/` is a byte-identical copy of its preserved source in the earlier educational package, which in turn copied the final HeartMAS release. The original HeartMAS source data and research artifacts are not modified. The normalized `manifest.json` records file sizes, SHA-256 values, source provenance, label semantics and the new educational fit policy. `masheart.data.verify_sources` hashes bytes without parsing the JSONL labels.

| Preserved file | Role |
|---|---|
| `raw/a_dev.jsonl` | Official development: 125 examples, 11 linked source groups, 84 YS and 41 NO. The only permitted source for internal training and validation. |
| `raw/a_test.jsonl` | Official test: 134 examples, 10 linked source groups, 96 YS and 38 NO. Read records only after freezing configuration and models. |
| `raw/pcc_source_provenance.json` | Original PCC provenance, creators, source URL, corpus checksum and license statement. |
| `raw/bioclaim_source_provenance.json` | Original BioClaim repository revision, artifact URLs/checksums and both upstream license statements. |
| `raw/source_split_manifest.json` | Original linked-source split manifest. It retains information about both original tasks to explain grouping; the CPU workflow uses Task A. |

The original provenance files refer to paths and artifacts in the original research repository. Those references document origin; they do not imply that every referenced Task-B, XML or code artifact is distributed in MASHeart.

## Upstream license statements

The preserved PCC provenance records **CC BY-NC-SA 2.0 UK** for the dataset. BioClaim's released README names **CC BY-NC-SA 4.0 International**, while its paper Appendix B.1 describes following PCC's **CC BY-NC-SA 2.0 UK England & Wales** conditions. The original provenance explicitly records this discrepancy. MASHeart preserves both statements and does not invent a replacement license or treat the article's license as the dataset's license.

Retain source attribution and upstream notices when sharing these data. The recorded licenses include noncommercial and share-alike conditions. The project's software license, if provided separately, does not replace the upstream data terms. Clarify ambiguous rights with the data owners when the intended use requires it.

## Full JSONL schema

JSONL contains one independent JSON object per line, encoded in UTF-8. Read IDs as strings and preserve question/statement text unchanged.

| Field | Type | Meaning and permitted role |
|---|---|---|
| `id` | string | Unique example identifier, such as `mc-23435582-00`. Join predictions to sources and audit membership; never a model feature. |
| `review_id` | string | Identifier of the review associated with the research question. Source metadata only. |
| `review_no` | integer | Corpus ordering of the review; not a confidence score. |
| `review_title` | string | Source review title for context and provenance. Excluded from the lesson's model features. |
| `question` | string | Research question in English. One of the two permitted inference fields. |
| `text` | string | Supplied statement in English. The second permitted inference field. |
| `label` | `YS` or `NO` | Supervised target. Training labels may enter fit; validation/test labels enter scoring after predictions. Never include in inference prompts or features. |
| `pmid` | string | PubMed source identifier. Use for provenance and leakage checks, not as a feature. |
| `claim_type` | `CAUS` or `EVAL` | Original statement category. Neither a verified causal claim nor a quality-of-evidence rating. Excluded from model features. |
| `source` | string | Corpus name, here `PCC-2016`. |
| `source_url` | string | PubMed link for source traceability; not a certification of clinical truth. |
| `group_id` | string | Linked source/question component, such as `qg-23435582`. Split and bootstrap by this field; exclude it from model features. |
| `split` | `dev` or `test` | Official linked-source membership. Internal educational train/validation partitions retain `dev` in this original field. |

## Educational train/validation/test protocol

**The original linked release contains no independent official training set.** MASHeart creates an explicitly new, educational **development-fit** experiment. `GroupShuffleSplit(test_size=0.25, random_state=17)` partitions official development into **95 training examples in 8 groups** and **30 validation examples in 3 groups**. The fraction applies to groups, so the example fraction need not equal 25%. The **134-example, 10-group official test** is preserved.

Complete groups stay together. The original grouping links source overlaps, duplicate PMIDs/texts, BioClaim relationships and near research questions. This reduces observable leakage; it does not prove semantic independence. The workflow verifies empty intersections of IDs, groups, PMIDs and case/whitespace-normalized statement text across internal train/validation and official test. Audit normalization is not used to rewrite the source or manufacture quotes.

Fit vocabularies, IDF statistics, classifiers, majority labels and any retrieval index on internal training only. Validation may guide explicitly declared model/checkpoint choices in the separate GPU workflow. The CPU comparator has fixed declared hyperparameters and does not refit on all development. Official test bytes may be checksummed before freezing, but its records and gold labels are parsed only after the source audit, split IDs, environment, configuration, model files and frozen hashes have been saved. Test errors may be inspected after evaluation; using them to change the method creates a subsequent development cycle, not a new unseen test.

Legacy `data/processed` files from the original repository are not included or permitted as training sources. The original audit found source overlap between those earlier splits and the later linked test protocol.

## Dataset limits and historical exposure

The original corpus is small, class-imbalanced and selected around disagreements in the literature. Several statements can share a question, study or review. The 134 test rows are not 134 independent source groups. Report both classes' precision/recall/F1, fixed-label macro-F1, full-sample accuracy, confusion counts and group-level behavior. Use paired group bootstrap when comparing systems and disclose the limited group count.

The original HeartMAS method was developed on development data, and an earlier v1 encountered test. The reported v4 protocol was frozen before its reported final test execution, but redistributing data does not erase prior exposure or create a historically pristine benchmark. The CPU report must disclose this history and must not copy scores from the original Qwen3-8B experiments into a new result table.

The corpus includes cardiovascular topics and associated outcomes, including pregnancy-related hypertension/pre-eclampsia, kidney function and diabetes in relevant contexts. Do not describe every example as a purely cardiovascular intervention trial. Scientific statements and their annotations are not current clinical guidelines. This project neither translates the English source nor assumes that the same labels remain valid after rewriting it.

If a future system marks some examples for review or abstains, report coverage and accepted-example counts alongside selective scores; compare alternative systems on the same subset. A matching source quote verifies the origin of the words, not the correctness of the inference.
