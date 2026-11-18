# Fraudulent Job Posting Detection

Text mining coursework project: classifying job advertisements as legitimate or
fraudulent from their free-text content.

## Problem

Online recruitment fraud exposes job seekers to identity theft and advance-fee
scams. The task framed here is the one a job board actually faces: given the
text of a submitted posting, flag the ones likely to be fraudulent before
publication.

This is a **binary text classification** problem with **severe class
imbalance** -- roughly 5% of postings are fraudulent. That imbalance is the
methodological centre of the project: a model that predicts "legitimate" for
every input scores ~95% accuracy while being completely useless, so the work
focuses on minority-class precision and recall rather than accuracy.

## Dataset

**EMSCAD** (Employment Scam Aegean Dataset), Laboratory of Information &
Communication Systems Security, University of the Aegean.

| Property | Value |
| --- | --- |
| Rows | 17,880 job postings |
| Fraudulent | 866 (~4.8%) |
| Legitimate | 17,014 |
| Columns | 18 |
| Period | 2012-2014 |

Free-text fields: `title`, `company_profile`, `description`, `requirements`,
`benefits`. Structured fields include `telecommuting`, `has_company_logo`,
`has_questions`, `employment_type`, `required_experience`,
`required_education`, `industry`, `function`.

The dataset is **not tracked in this repository**. Download it and place
`fake_job_postings.csv` in `data/raw/`:

- Kaggle mirror: <https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction>
- Original source: <http://emscad.samos.aegean.gr>

Verify the placement with `uv run fjpd check`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the project root:

```bash
uv sync                      # create .venv and install dependencies
uv sync --extra notebook     # also install JupyterLab
uv run fjpd check            # confirm the dataset is present and valid
```

## Project layout

```
data/
  raw/          fake_job_postings.csv (not tracked)
  interim/      intermediate cleaning artifacts
  processed/    model-ready feature matrices
models/         serialised trained estimators
notebooks/      narrative exploration
reports/
  figures/      generated plots
  tables/       generated CSV/Markdown tables
src/fraudulent_job_posting_detection/
  config.py     paths, dataset schema, experiment constants
  data.py       loading and integrity validation
  cli.py        pipeline entry point
```

## Pipeline

Run the stages in order. Each writes its tables to `reports/tables/` as both
CSV and Markdown, and its figures to `reports/figures/`.

```bash
uv run fjpd check        # validate the raw dataset
uv run fjpd explore      # stage 1: exploratory analysis
uv run fjpd preprocess   # stage 2: clean and persist the corpus
uv run fjpd split        # stage 2: audit the leakage-controlled split
uv run fjpd compare      # stage 4: cross-validate candidate models (resumable)
uv run fjpd imbalance    # stage 4b: compare imbalance strategies
uv run fjpd evaluate     # stage 5: held-out evaluation and interpretability
```

## Key findings

**Two integrity problems shape every result.**

*Duplicate postings.* 4,446 postings (24.9%) share a description with another
posting; the largest single cluster has 380 identical descriptions. A naive
stratified split leaves 22.3% of test postings memorisable from training.
Every split here is therefore grouped on description text, so duplicates
cannot straddle the boundary.

*Redaction placeholders.* EMSCAD masks contact details as `#EMAIL_<sha256>#`.
The hashes survive tokenisation as unique per-entity terms, and a naive
log-odds analysis ranks them among the strongest "fraud indicators" — pure
memorisation. But their *presence* is real signal (`#EMAIL_` appears in 20.1%
of fraudulent postings against 6.8% of legitimate). They are therefore
normalised to single tokens (`emailref`), not stripped.

**The strongest predictor is absence of institutional footprint.**

| Signal | Fraud rate | Baseline 4.84% |
| --- | --- | --- |
| `has_company_logo = 0` | 15.93% | vs 1.99% when present |
| `company_profile` empty | 67.78% of frauds | vs 15.99% of legitimate |
| `has_questions = 0` | 6.78% | vs 2.84% |
| `telecommuting = 1` | 8.34% | vs 4.69% |

**Model comparison** (5-fold grouped stratified CV on the training split):

| Model | Features | F1 (fraud) | Avg. precision |
| --- | --- | --- | --- |
| LinearSVC | tfidf + structured | **0.808** | 0.878 |
| LinearSVC | tfidf only | 0.808 | 0.861 |
| LogisticRegression | tfidf only | 0.752 | 0.844 |
| RandomForest | tfidf + structured | 0.734 | 0.821 |
| MultinomialNB | tfidf + structured | 0.713 | 0.813 |

**Imbalance treatment moves the operating point, not the ranking.** Average
precision is essentially unchanged across no treatment (0.881), class
weighting (0.878) and SMOTE (0.878), while F1 moves from 0.772 to 0.815. The
model's ability to rank postings is the same; what changes is where the
threshold falls.

**Held-out test results** (3,576 postings, touched once):

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Majority baseline | 0.9516 | 0.000 | 0.000 | 0.000 |
| LinearSVC + class weight | 0.9857 | 0.891 | 0.804 | **0.845** |
| LinearSVC + SMOTE | 0.9860 | 0.913 | 0.786 | 0.845 |

The majority baseline scores 95.2% accuracy while catching zero fraud, which
is why F1 on the minority class is the headline metric.

**What the model learns.** The top-weighted terms are off-platform
application patterns (`use link`, `link urlref`, `appli use`), earnings
promises (`earn`, `money`), and low-barrier roles (`data entri`, `cleric`,
`receptionist`, `timejob`). Legitimate postings are marked by concrete
technical vocabulary (`softwar`, `hardwar`, `analyt`, `infrastructur`).

Note when reading coefficients: TF-IDF features have a mean active value of
0.057 against 0.824 for the binary flags, so the blocks are not directly
comparable by coefficient magnitude. Scaled by typical value,
`has_company_logo` is among the most influential features.

## Limitations

- Grouping is on *exact* duplicate descriptions. Near-duplicates from the same
  campaign (reworded templates) can still span the split, so the reported
  figures remain a mild upper bound.
- Company names are not removed. A single impersonation campaign against one
  oil & energy firm accounts for much of that industry's 37.98% fraud rate,
  and the model can key on it. This is arguably realistic — a real job board
  does see repeat offenders — but it does not generalise to new campaigns.
- The data covers 2012-2014. Scam language shifts over time.

## Evaluation note

Accuracy is reported only for comparison against the majority-class baseline.
Model selection uses **F1 and average precision on the fraudulent class**,
because the operational cost of missing a scam differs sharply from the cost of
reviewing a legitimate posting a second time.

## Licence and attribution

Code in this repository is coursework. The EMSCAD dataset remains the property
of its creators at the University of the Aegean and is used here for academic
purposes only.
