# Fraudulent Job Posting Detection

Text mining coursework project (MSc Computer Science — Machine Learning Text
Mining). Classifies job advertisements as legitimate or fraudulent from their
text, with a command-line pipeline and a Flask interface.

**Headline result:** F1 = 0.845 on the fraudulent class, against a majority
baseline of 0.000 at 95.2% accuracy.

---

## Contents

- [The problem](#the-problem)
- [Dataset](#dataset)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Command-line pipeline](#command-line-pipeline)
- [Web interface](#web-interface)
- [Three decisions that shape the results](#three-decisions-that-shape-the-results)
- [Results](#results)
- [What the model learns](#what-the-model-learns)
- [Limitations](#limitations)
- [Attribution](#attribution)

---

## The problem

Online recruitment fraud exposes job seekers to identity theft and advance-fee
scams. The task framed here is the one a job board actually faces: given the
text of a submitted advert, flag the ones likely to be fraudulent before
publication.

This is **binary text classification under severe class imbalance** — roughly
1 posting in 20 is fraudulent. That imbalance is the methodological centre of
the project. A model that predicts "legitimate" for every input scores 95.2%
accuracy while catching nothing, so accuracy is reported only for contrast and
**model selection uses F1 and average precision on the fraudulent class**.

## Dataset

**EMSCAD** (Employment Scam Aegean Dataset), Laboratory of Information &
Communication Systems Security, University of the Aegean.

| Property   | Value           |
| ---------- | --------------- |
| Postings   | 17,880          |
| Fraudulent | 866 (4.84%)     |
| Legitimate | 17,014 (95.16%) |
| Columns    | 18              |
| Period     | 2012–2014       |

Free-text fields: `title`, `company_profile`, `description`, `requirements`,
`benefits`. Structured fields include `telecommuting`, `has_company_logo`,
`has_questions`, `employment_type`, `required_experience`,
`required_education`, `industry`, `function`.

The dataset is **not tracked in this repository**. Download it and place
`fake_job_postings.csv` in `data/raw/`:

- Kaggle mirror: <https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction>
- Original source: <http://emscad.samos.aegean.gr>

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync                  # create .venv and install dependencies
uv run fjpd check        # confirm the dataset is present and valid
uv run fjpd serve        # web interface at http://127.0.0.1:5000
```

Or run the whole pipeline from the command line:

```bash
uv run fjpd explore      # stage 1
uv run fjpd preprocess   # stage 2
uv run fjpd compare      # stages 3–4
uv run fjpd evaluate     # stage 5
```

## Repository structure

```
.
├── data/
│   ├── raw/                     fake_job_postings.csv            (not tracked)
│   ├── interim/                 cv_results.csv, resumable CV runs (not tracked)
│   └── processed/               corpus.parquet + corpus_meta.json (not tracked)
├── models/                      serialised pipelines              (not tracked)
├── reports/
│   ├── figures/                 generated PNGs                    (not tracked)
│   └── tables/                  generated CSV + Markdown tables   (not tracked)
├── src/fraudulent_job_posting_detection/
│   ├── config.py                paths, dataset schema, experiment constants
│   ├── data.py                  loading and integrity validation
│   ├── explore.py               stage 1 — exploratory analysis
│   ├── preprocess.py            stage 2 — text normalisation
│   ├── splits.py                stage 2 — leakage-controlled splitting
│   ├── models.py                stages 3–4 — features, model comparison
│   ├── evaluate.py              stage 5 — held-out evaluation, interpretability
│   ├── predict.py               scoring a single pasted advert
│   ├── plotting.py              shared table/figure helpers
│   ├── cli.py                   command-line entry point
│   └── webapp/
│       ├── app.py               Flask routes (one per stage)
│       ├── jobs.py              background worker for long actions
│       ├── static/style.css
│       └── templates/           base, index, explore, clean, train, predict
├── pyproject.toml
└── uv.lock
```

Everything under `data/`, `models/` and `reports/` is regenerable and therefore
gitignored; the folders are kept via `.gitkeep`. If your submission needs the
figures tracked, uncomment those blocks in `.gitignore`.

**`config.py` is the single source of truth.** No other module hardcodes a path
or a column name, so a renamed field is a one-file change.

## Command-line pipeline

| Command           | Stage | Produces                                                                |
| ----------------- | ----- | ----------------------------------------------------------------------- |
| `fjpd check`      | —     | Validates the raw CSV, prints the class balance                         |
| `fjpd explore`    | 1     | Missingness, text lengths, fraud rates, distinguishing terms, 3 figures |
| `fjpd preprocess` | 2     | `corpus.parquet` + `corpus_meta.json`                                   |
| `fjpd split`      | 2     | Audits the grouped train/test split                                     |
| `fjpd compare`    | 3–4   | 5-fold grouped CV of four models (resumable; `--only NAME`)             |
| `fjpd imbalance`  | 4b    | No treatment vs class weighting vs SMOTE                                |
| `fjpd evaluate`   | 5     | Held-out metrics, threshold tuning, coefficients, 4 figures             |
| `fjpd serve`      | —     | The web interface                                                       |

Tables are written to `reports/tables/` as both CSV and Markdown, so figures
and numbers can be pasted straight into a report.

## Web interface

```bash
uv run fjpd serve
```

Five pages. Each owns one stage and shows only that stage's output.

| Page            | Does                                                                                     |
| --------------- | ---------------------------------------------------------------------------------------- |
| **1 · Import**  | Upload the CSV, validate it, show the class balance                                      |
| **2 · Explore** | Raw preview, missingness, fraud rate by flag, stage-1 figures                            |
| **3 · Clean**   | Toggle normalisation / stopwords / stemming, build the corpus, show what built it        |
| **4 · Train**   | The grouped split, model and metric selection, ranked results, top-3 podium, diagnostics |
| **5 · Predict** | Paste an advert, score it, see the terms that drove the decision                         |

Three design notes:

- **Long actions run on a worker thread** while the page polls `/job/<name>`,
  so no HTTP request is held open for the ~30s cleaning or ~15s-per-model
  training. The job registry is an in-memory dict: this is a single-user local
  tool, not a multi-worker deployment.
- **Results are persisted** to `models/ui_results.json` alongside the saved
  model, so a restart does not strand a trained model with no results.
- **The interface calls the same functions as the CLI**, so the two cannot
  drift. It uses one grouped split for responsiveness; `fjpd compare` runs the
  cross-validation whose numbers belong in a report.

## Three decisions that shape the results

Exploratory analysis surfaced two integrity problems. Both would have inflated
the results silently, and handling them is the substance of this project.

### 1. A quarter of the dataset is duplicated

**4,446 postings (24.9%) share a description with another posting**, across
1,233 distinct texts; the largest single cluster has 380 identical
descriptions. Under an ordinary random split, identical documents land on both
sides of the boundary and the test score measures memorisation.

Measured directly: a naive stratified split leaves **22.3% of test postings
memorisable** from training.

Every split here is therefore **grouped on description text** using
`StratifiedGroupKFold`, so a text seen in training can never reappear in test.
Splits stay stratified because at a 4.84% positive rate an unstratified split
distorts the minority class. Only one duplicate group contains both labels, so
the labelling is internally consistent.

### 2. Redaction placeholders are both a trap and a signal

EMSCAD masks contact details as `#EMAIL_<sha256>#`. The hashes survive
tokenisation as unique per-entity terms, and a naive log-odds analysis ranks
them among the strongest "fraud indicators" — pure memorisation of individual
scammers.

But their _presence_ is real signal:

| Placeholder | In fraudulent | In legitimate |
| ----------- | ------------- | ------------- |
| `#EMAIL_`   | 20.09%        | 6.82%         |
| `#PHONE_`   | 9.12%         | 2.63%         |
| `#URL_`     | 30.83%        | 31.47%        |

So they are **normalised, not stripped**: each family collapses to one token
(`emailref`, `phoneref`, `urlref`). Presence is kept, identity discarded.

### 3. Every transform is fitted inside the fold

Feature extraction lives in a scikit-learn `Pipeline`, so TF-IDF vocabulary and
document frequencies are learned per fold. Vectorising the whole corpus before
splitting is a subtler leak than duplicate rows and just as damaging. SMOTE is
likewise applied inside the pipeline, never before cross-validation.

## Results

### Model comparison (5-fold grouped stratified CV, training split)

| Model              | Features           | F1 (fraud) | Avg. precision |
| ------------------ | ------------------ | ---------- | -------------- |
| **LinearSVC**      | tfidf + structured | **0.808**  | 0.878          |
| LinearSVC          | tfidf only         | 0.808      | 0.861          |
| LogisticRegression | tfidf only         | 0.752      | 0.844          |
| RandomForest       | tfidf + structured | 0.734      | 0.821          |
| MultinomialNB      | tfidf + structured | 0.713      | 0.813          |

### Imbalance treatment moves the operating point, not the ranking

| Strategy        | F1    | Precision | Recall | Avg. precision |
| --------------- | ----- | --------- | ------ | -------------- |
| None            | 0.772 | 0.963     | 0.648  | 0.881          |
| Class weighting | 0.808 | 0.899     | 0.736  | 0.878          |
| SMOTE           | 0.815 | 0.929     | 0.729  | 0.878          |

Average precision is essentially unchanged while F1 moves by four points. The
model's ability to _rank_ postings is the same; only the threshold moves. This
matters because SMOTE is widely reported as "improving" results on this
dataset — here it demonstrably does not improve the ranking.

### Held-out test split (3,576 postings, evaluated once)

| Model                        | Accuracy | Precision | Recall | F1        |
| ---------------------------- | -------- | --------- | ------ | --------- |
| Majority baseline            | 0.9516   | 0.000     | 0.000  | 0.000     |
| **LinearSVC + class weight** | 0.9857   | 0.891     | 0.804  | **0.845** |
| LinearSVC + SMOTE            | 0.9860   | 0.913     | 0.786  | 0.845     |

The decision threshold is tuned on **out-of-fold** predictions. Tuning on
in-sample training scores selects 0.831 and collapses recall to 0.35, because a
fitted model separates its own training data far more confidently than unseen
data.

## What the model learns

**The strongest predictor is absence of institutional footprint**, not
vocabulary:

| Signal                  | Fraud rate       | Baseline 4.84%          |
| ----------------------- | ---------------- | ----------------------- |
| `has_company_logo = 0`  | 15.93%           | vs 1.99% when present   |
| `company_profile` empty | 67.78% of frauds | vs 15.99% of legitimate |
| `has_questions = 0`     | 6.78%            | vs 2.84%                |
| `telecommuting = 1`     | 8.34%            | vs 4.69%                |

Among terms, the heaviest weights fall on off-platform application patterns
(`use link`, `link urlref`, `appli use`), earnings promises (`earn`, `money`)
and low-barrier roles (`data entri`, `cleric`, `receptionist`). Legitimate
postings are marked by concrete technical vocabulary (`softwar`, `hardwar`,
`analyt`, `infrastructur`).

**Reading coefficients correctly:** TF-IDF features have a mean active value of
0.057 against 0.824 for the binary flags, so the two blocks are not comparable
by coefficient magnitude. Scaled by typical value, `has_company_logo` is among
the most influential features despite a modest coefficient. The Predict page
reports contribution as value × weight for this reason.

## Limitations

- Grouping is on **exact** duplicate descriptions. Reworded templates from the
  same campaign can still span the split, so the figures remain a mild upper
  bound.
- Company names are **not** removed. One impersonation campaign against an oil
  & energy firm accounts for much of that industry's 37.98% fraud rate, and the
  model can key on it. Arguably realistic — a real job board does see repeat
  offenders — but it does not generalise to new campaigns.
- The data covers **2012–2014**; scam language shifts over time.
- Bare digits are dropped during cleaning, so "$5,000/month" keeps `month` but
  not the figure. Stage 1 showed numeric tokens were mostly memorisation
  artifacts from one job cluster.
