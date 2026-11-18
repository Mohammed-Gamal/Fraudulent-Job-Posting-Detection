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

| Stage | Purpose |
| --- | --- |
| 1. EDA | Class balance, missingness, text lengths, distinguishing terms |
| 2. Preprocessing | HTML stripping, normalisation, tokenisation, lemmatisation |
| 3. Features | Bag-of-words and TF-IDF (word + character n-grams), structured flags |
| 4. Models | Naive Bayes, Logistic Regression, Linear SVM, Random Forest |
| 5. Imbalance | Baseline vs class weighting vs SMOTE |
| 6. Evaluation | Minority-class P/R/F1, PR curves, threshold tuning, confusion matrices |
| 7. Interpretability | Top weighted features per class from linear models |

## Evaluation note

Accuracy is reported only for comparison against the majority-class baseline.
Model selection uses **F1 and average precision on the fraudulent class**,
because the operational cost of missing a scam differs sharply from the cost of
reviewing a legitimate posting a second time.

## Licence and attribution

Code in this repository is coursework. The EMSCAD dataset remains the property
of its creators at the University of the Aegean and is used here for academic
purposes only.
