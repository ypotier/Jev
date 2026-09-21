# Evaluating Jev on ChEMBL assay curation

Does TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
— a "System One" model that returns calibrated, typed decisions instead of text —
earn its place in a life-science curation pipeline?

The task: given a free-text ChEMBL assay description, assign the correct
**BioAssay Ontology (BAO)** assay-format class, the ChEMBL assay type, and a few
supporting judgments — then decide, from the model's own confidence, which records a
human curator still needs to see.

## Data

[`matchbench/ChEMBL-SM`](https://huggingface.co/datasets/matchbench/ChEMBL-SM) —
22,497 usable ChEMBL assay records with descriptions (median 98 characters).

The dataset is published as a *schema*-matching benchmark, but its `matches.txt` is the
identity map over identical column names, so that task is degenerate. We ignore it and
use the CSVs as what they actually are: a corpus of **curator-labelled** assay records.

| Column | Distinct | Role |
| --- | --- | --- |
| `bao_format` | 13 | gold label — BAO ontology class |
| `assay_type` | 5 | gold label — B/F/A/P/U |
| `confidence_score` | 10 | ChEMBL target-assignment confidence (0–9) |
| `curated_by` | 3 | `Expert` (1,383) / `Intermediate` (14,883) / `Autocuration` (6,234) |
| `assay_organism` | 77 | held out as gold for the `names_organism` question |

BAO labels and definitions are resolved from [EBI's Ontology Lookup Service](https://www.ebi.ac.uk/ols4/)
and cached to `data/bao_terms.json`. The ontology's own definition of each class becomes
that option's `criteria` text, so Jev is told what "microsome format" means in BAO's
words rather than being left to guess from an accession number.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

```bash
export TYPESAFE_API_KEY=ts-...
```

## Run

Baselines need no API key and no network beyond the dataset download:

```bash
PYTHONPATH=src ./.venv/bin/python -m jev_eval.cli baselines
```

Inspect exactly what gets sent, without sending it:

```bash
PYTHONPATH=src ./.venv/bin/python -m jev_eval.cli dry-run
```

The evaluation itself. Results stream to `results/predictions.jsonl` and re-runs skip
anything already cached, so an interrupted run resumes:

```bash
PYTHONPATH=src ./.venv/bin/python -m jev_eval.cli eval --per-class 100
```

`--natural --size 1000` samples the real class distribution instead of stratifying —
use it for the coverage/routing numbers, which only mean anything in situ.

## Design notes

**One request, five questions.** Jev ingests the state once and evaluates every question
against it in parallel, so the five judgments cost barely more than one. Two of them
(`in_vivo`, `names_organism`) are speculative consistency probes rather than scored
tasks — they are nearly free on the same pass.

**State is the description alone.** That is what a curator reads, and Jev documents
losing accuracy when the state carries material the question does not need. It also
keeps `assay_organism` clean as a gold label instead of leaking it into the input.

**Arithmetic stays in code.** Per Jev's own
[jaggedness notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13), the model is
unreliable at counting, numeric comparison and dates. Every threshold, aggregation and
rank correlation here is computed in Python.

**We do not ask Jev to predict `confidence_score`.** ChEMBL's 0–9 scale is a taxonomy of
target-assignment specificity, not a true ordinal. Instead we ask a genuinely ordered
Score question about target specificity and report rank correlation against it.

## Results

1,027 stratified records (up to 100 per BAO class), `jev-1.13`, $0.092 total.
Corpus deduplicated to 14,997 unique assays. Both models scored on the **same records**;
TF-IDF trained only on the 13,970 held-out assays. Reproduce with `cli.py rescore`.

| Model | Exact | Macro-F1 | On-branch | Graded | Labels used |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logreg | **0.783** | 0.586 | **0.950** | **0.879** | 13,970 |
| Jev | 0.645 | **0.588** | 0.812 | 0.775 | **0** |

Macro-F1 is a dead heat — 0.002 apart on n=1,027. Jev matches a supervised model
trained on 13,970 labels using none. But it loses clearly on exact accuracy and on both
hierarchy-aware measures, so the tie is not "as good as"; it is "equally good on
average, by being good at the opposite end of the distribution."

### The models are complementary, not competing

Per-class F1 against how much training signal each class carries in the corpus:

| Class | Corpus n | Jev | TF-IDF | Δ |
| --- | --- | --- | --- | --- |
| organism-based format | 8,809 | 0.858 | 0.922 | −0.064 |
| assay format (root) | 1,980 | 0.189 | 0.613 | −0.424 |
| cell based format | 1,922 | 0.861 | 0.971 | −0.110 |
| single protein format | 1,228 | 0.054 | 0.697 | −0.643 |
| tissue-based format | 322 | 0.740 | 0.915 | −0.174 |
| cell membrane format | 163 | 0.943 | 0.964 | −0.021 |
| protein format | 157 | 0.333 | 0.585 | −0.252 |
| small-molecule physicochemical | 155 | 0.901 | 0.975 | −0.075 |
| microsome format | 134 | 0.995 | 0.980 | +0.015 |
| protein complex format | 63 | 0.000 | 0.000 | 0.000 |
| cell-free format | 44 | 0.293 | 0.000 | **+0.293** |
| subcellular format | 11 | 0.471 | 0.000 | **+0.471** |
| nucleic acid format | 9 | 1.000 | 0.000 | **+1.000** |

| Group | Classes | Jev | TF-IDF | Δ |
| --- | --- | --- | --- | --- |
| Rare (<200 in corpus) | 8 | **0.617** | 0.438 | **+0.179** |
| Common (≥200) | 5 | 0.541 | **0.824** | −0.283 |

**TF-IDF scores exactly zero on the three rarest classes** — it never predicts them at
all. Jev, given only the ontology's definition of each class, gets them. That is the
long-tail behaviour the project set out to test, and it holds.

The mirror image is Jev's two bad classes, both granularity failures rather than
comprehension failures: `single protein format` (0.054) loses to its own parent
`protein format`, and the root class (0.189) loses because our instruction tells the
model to commit wherever the description supports it, so it out-commits the curators.

### Error structure

| | Jev | TF-IDF |
| --- | --- | --- |
| exact | 662 (0.645) | 804 (0.783) |
| ancestor | 83 (0.081) | 121 (0.118) |
| descendant | 89 (0.087) | 51 (0.050) |
| sibling | **193 (0.188)** | 51 (0.050) |
| unrelated | **0** | **0** |

**Neither model ever lands on an unrelated branch.** Zero-shot, Jev's mistakes are
always ontologically adjacent. Its distinctive weakness is siblings — 193 errors, 52% of
its total, at nearly 4× the baseline's rate — concentrated in the protein cluster that
BAO nests as `single protein format` → `protein format` → `biochemical format`.

### Other findings

- **Target specificity tracks ChEMBL's curation taxonomy: Spearman 0.854** against
  `confidence_score`, free on the same request. The strongest single result.
- **Assay type: 0.811 accuracy** zero-shot on the 5-way task. (Its 0.640 macro-F1 is an
  artifact — one `Unclassified` record in the sample scores 0 and drags the mean.)
- **Discrimination good, calibration mediocre.** Auto-accepting the top 30% by
  confidence yields 94.2% accuracy, so confidence *ranks* well. But ECE is 0.117 and the
  0.8–0.9 bin is badly overconfident (0.858 confidence → 0.571 accuracy). Route on it;
  do not read it as a probability.
- **Curator back-off: no effect.** Confidence where the curator hedged to the root class
  was 0.723 vs 0.732 where they committed. Hypothesis not supported.
- **The curation-tier breakdown is confounded** under stratified sampling and should not
  be read as-is: `Intermediate` (0.901) is dominated by classes Jev scores 0.86–0.96 on,
  while `Expert` (0.623) is loaded with single-protein format, which it scores 0.054 on.
  Only meaningful on a `--natural` run.

### Open question

Sibling confusion is 52% of Jev's errors. Replacing the flat 13-way Choice with
node-by-node traversal would give sibling discrimination a dedicated question over 2–4
competing options. Untested — and note that ontology-aware scoring was *expected* to
close the gap and did not, so treat it as a hypothesis.

## Cost

~1,390 input tokens per record at $0.042/Mtok. A 1,300-record stratified run is about
1.8M tokens — roughly **$0.08**. Output tokens are free. The full 22,497-record corpus
would be about $1.30.

## Layout

```
src/jev_eval/
  data.py        load and clean the ChEMBL-SM corpus
  ontology.py    resolve BAO classes to labels + definitions via EBI OLS
  sample.py      stratified / natural sampling, train-test split
  questions.py   build the five-question fan-out request
  run.py         threaded runner with resumable on-disk caching
  metrics.py     macro-F1, calibration/ECE, coverage, back-off, rank correlation
  baselines.py   majority class and TF-IDF + logistic regression
  cli.py         baselines / dry-run / eval
```

## Caveats

- Gold labels are ChEMBL's curation, not ground truth. 28% of records are
  `Autocuration` — machine-assigned — so disagreement there is not necessarily a Jev
  error. `by_curation_tier` in the report splits agreement by tier for this reason.
- `BAO_0000019` is the ontology **root**, used when a curator declined to be specific.
  It is a legitimate prediction target here, not a missing value.
- Severe class imbalance; rare classes have as few as 3 test records, so their per-class
  numbers are noisy.
