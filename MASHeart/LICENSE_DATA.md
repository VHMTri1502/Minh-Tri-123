# Dataset attribution and model license notices

MASHeart contains software, documentation, derived research records and references to pretrained model weights. These materials do not acquire one common license simply because they share a project folder. This file preserves upstream attribution and identifies the separate notices; it does not relicense the datasets.

## 1. PCC statement corpus

**Creators:** Abdulaziz Alamri and Mark Stevenson.

**Publication:** *A corpus of potentially contradictory research claims from cardiovascular research abstracts*, Journal of Biomedical Semantics 7, article 36 (2016). [Publication and DOI](https://doi.org/10.1186/s13326-016-0083-z).

**Source resource:** [corpus.xml](https://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/corpus.xml).

**Recorded dataset license:** [Creative Commons Attribution–NonCommercial–ShareAlike 2.0 UK: England & Wales](https://creativecommons.org/licenses/by-nc-sa/2.0/uk/), as recorded in the inherited corpus provenance. Preserve attribution, upstream notices and applicable noncommercial/share-alike conditions when distributing corpus material or adaptations. The article's publication license must not be substituted for the dataset's notice.

The original corpus fingerprint recorded by the inherited project is:

```text
SHA-256 corpus.xml
c6925524b0117e877e64782af048d18a851bf0127361abe29587858238a180f4
```

MASHeart's Task A files retain PCC questions, statements, labels and source identifiers, with added linked-group/split metadata. Train/validation is a documented group partition of the inherited development data. Retaining or serializing these records as JSONL does not make them newly authored medical text or remove the upstream license.

## 2. BioClaim and inherited linked split

**Creators/publication:** Youngwoo Kim, Razieh Rahimi and James Allan. *Conditional Natural Language Inference*, Findings of EMNLP 2023. [Paper](https://aclanthology.org/2023.findings-emnlp.456/), DOI `10.18653/v1/2023.findings-emnlp.456`.

**Upstream source:** [BioClaim at revision 14e88c344c148c46bb6dbaab2149b252b5e030d3](https://github.com/youngwoo-umass/cond-nli/tree/14e88c344c148c46bb6dbaab2149b252b5e030d3/BioClaim).

The distributed BioClaim README explicitly identifies [CC BY-NC-SA 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/). Its paper appendix describes following the source PCC CC BY-NC-SA 2.0 UK terms. The inherited provenance records this discrepancy. MASHeart preserves both upstream statements rather than silently choosing a replacement or treating the paper's publication license as the data license.

The inherited official membership is `val.csv → development`, `test.csv → test`. MASHeart uses the linked Task A records; it does not present newly invented BioClaim token annotations. Original released CSV fingerprints are:

```text
val.csv
945fff10e150563368e09dfb1400465f472fa0395b463ecdebf757090ded4566

test.csv
1d724684d5768c4fe6f0e3b7fea06b351407af79bcd67ba1f1d08469b2e29525
```

## 3. Pretrained model weights

The T4 configuration references **Qwen/Qwen3-4B-Instruct-2507**, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Its model distribution identifies **Apache License 2.0**. See the [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and [LICENSE at the pinned revision](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/LICENSE).

Base weights are downloaded separately by the GPU workflow; they are not included in the small project source bundle. Keep upstream notices with any weight redistribution. Quantization and a trained adapter do not change the license of the original corpus, and the model's Apache license does not turn the supplied dataset into Apache-licensed data.

## 4. Software and project citation

Any software license at the project root applies according to its own scope; it must not be read as replacing the dataset/model notices above. Keep data provenance with result archives and cite both original publications when using the supplied records or inherited split. [CITATION.cff](CITATION.cff) provides the software title and original-paper references without inventing a project DOI or named individual authors.

The generic “MASHeart contributors” entry is a collective project credit. It is not a claim that the original dataset authors wrote this software or endorsed its results. Replace it with verified contributor details when preparing an archival software release.
