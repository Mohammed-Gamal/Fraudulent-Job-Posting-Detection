# Fraudulent Job Posting Detection

Text mining coursework project (MSc Computer Science — Machine Learning Text
Mining). Classifies job advertisements as legitimate or fraudulent from their
text, with a command-line pipeline and a single-page web interface.

**Headline result:** F1 = 0.853 on the fraudulent class, against a majority
baseline of 0.000 at 95.2% accuracy. A learned embedding network reaches
0.768 — second of five models, and the comparison between the two is the most
interesting thing in the project.

https://github.com/user-attachments/assets/43ce2c6f-7335-4d1a-9bca-2f42eca27cf3

---

## Contents

- [The problem](#the-problem)
- [Dataset](#dataset)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Command-line pipeline](#command-line-pipeline)
- [Web interface](#web-interface)
- [Interactive charts](#interactive-charts)
- [Choosing the evaluation protocol](#choosing-the-evaluation-protocol)
- [Keeping the page fast](#keeping-the-page-fast)
- [Starting over](#starting-over)
- [The embedding layer](#the-embedding-layer)
- [Four decisions that shape the results](#four-decisions-that-shape-the-results)
- [Results](#results)
- [What the models learn](#what-the-models-learn)
- [Tests](#tests)
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
uv sync                    # create .venv and install dependencies
uv sync --extra embedding  # add PyTorch, for the embedding model
uv run fjpd serve          # the interface, at http://127.0.0.1:5000
```

**Every stage runs in the interface** — importing, cleaning, training,
cross-validating, the final held-out evaluation, and scoring a pasted advert.
There is no separate command-line pipeline to learn, and nothing that can
only be produced from a terminal.

PyTorch is an **optional** extra. Without it the four sparse models run
normally and the embedding model is hidden from the interface rather than
offered and broken.

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
│   ├── data.py                  loading, validation, caching
│   ├── explore.py               stage 1 — exploratory analysis
│   ├── preprocess.py            stage 2 — text normalisation
│   ├── splits.py                stage 2 — leakage-controlled splitting
│   ├── models.py                stages 3–4 — registry, features, comparison
│   ├── embeddings.py            stage 3b — the trainable embedding layer
│   ├── evaluate.py              stage 5 — held-out evaluation, interpretability
│   ├── predict.py               scoring a single pasted advert
│   ├── plotting.py              shared table/figure helpers
│   ├── progress.py              weighted, nestable progress reporting
│   ├── charts.py                exploratory results as JSON, for the page
│   ├── analysis.py              the report stages, as runnable jobs
│   ├── reset.py                 clearing generated state, safely
│   ├── cli.py                   command-line entry point
│   └── webapp/
│       ├── app.py               Flask: one page + a JSON API
│       ├── jobs.py              background worker for long actions
│       ├── static/              style.css, app.js, charts.js
│       └── templates/index.html the single page
├── tests/                       259 tests, ~110s
├── pyproject.toml
└── uv.lock
```

Everything under `data/`, `models/` and `reports/` is regenerable and therefore
gitignored; the folders are kept via `.gitkeep`. If your submission needs the
figures tracked, uncomment those blocks in `.gitignore`.

**`config.py` is the single source of truth.** No other module hardcodes a path
or a column name, so a renamed field is a one-file change.

## Command line

One command, because everything else lives in the interface:

| Command        | Does                                                  |
| -------------- | ----------------------------------------------------- |
| `fjpd serve`   | Start the interface (also the default with no command) |
| `fjpd check`   | Confirm the raw CSV parses and validates               |

Earlier versions carried a subcommand per stage — `explore`, `preprocess`,
`compare`, `imbalance`, `evaluate`. That meant two surfaces doing the same
work, and the numbers that belong in a report were reachable only from a
terminal. Those stages are now the **Report** stage of the interface, with
their parameters as controls.

## Web interface

```bash
uv run fjpd serve
```

**One page.** The five pipeline stages are panels on a single document, with a
left rail that gates each stage on its real dependency — Train stays locked
until a corpus exists, Predict until a model has been saved. Nothing reloads;
the page keeps one consistent view of the pipeline from a single `/api/state`
call.

| Stage           | Does                                                                                     |
| --------------- | ---------------------------------------------------------------------------------------- |
| **1 · Import**  | Upload the CSV, validate it, show the class balance drawn to scale                       |
| **2 · Explore** | Raw preview, missingness, fraud rate by flag, stage-1 figures                            |
| **3 · Clean**   | Toggle normalisation / stopwords / stemming, build the corpus, show what built it        |
| **4 · Train**   | The grouped split, model and metric selection, ranked results, diagnostics               |
| **5 · Predict** | Paste an advert, score it, see what drove the decision                                   |
| **6 · Report**  | Cross-validation, imbalance comparison, final held-out evaluation, report tables         |

### The progress indicator

Long actions run on a worker thread while the page polls `/api/jobs`. A bar
docks under the header showing **a real percentage**, the current step, elapsed
time, and a cancel button.

The percentage is not a step counter. `progress.py` gives each stage a weighted
slice of the [0, 1] range sized by how long that stage actually takes, and
slices nest: the embedding model reports per-epoch progress into its own
sub-slice, which rolls up into the overall figure without any step doing global
arithmetic. Two pieces of state keep that honest — a per-reporter cursor so
sibling steps lay out end to end, and a **tree-wide floor** so a child
beginning its slice can never drag the bar backwards, which would read as a
fault.

The API surface:

```
GET  /api/state             everything the page renders
POST /api/upload            replace the raw CSV (validated before replacing)
POST /api/explore/charts    data for the interactive charts
POST /api/split/preview     compose and audit a split without training
POST /api/run/<stage>       start a background job
POST /api/cancel/<stage>    ask a running job to stop
GET  /api/jobs              progress, for polling
POST /api/predict           score one pasted advert
GET  /api/reset/preview     what "start over" would delete
POST /api/reset             delete it
GET  /figure/<name>         a generated PNG
```

Three design notes:

- **Only one job runs at a time.** Cleaning and training both load the corpus
  and both draw matplotlib figures, and pyplot is global state. Guarding only
  against starting the *same* job twice let two jobs interleave their figures
  and corrupt both.
- **Results are persisted** to `models/ui_results.json` alongside the saved
  model, so a restart does not strand a trained model with no results.
- **The interface calls the same functions as the CLI**, so the two cannot
  drift. It uses one grouped split for responsiveness; `fjpd compare` runs the
  cross-validation whose numbers belong in a report.

### Interactive charts

The Explore stage draws its charts in the browser from JSON rather than
serving PNGs, so every mark can be hovered for its exact numbers and the
chart list is a set of checkboxes rather than a fixed gallery. Five are
offered: description length by class, fraud rate by binary flag,
distinguishing vocabulary, missing fields by class, and fraud rate by
category.

They are hand-built SVG in `static/charts.js` rather than a charting library,
for the same reason there is no webfont: the app runs on localhost and may
have no network, so a CDN import would be a single point of failure, and
vendoring three megabytes to draw five charts is a poor trade.

Two of them answer a question the static figures could not:

- **Distinguishing vocabulary** has a *placeholders* switch. Left raw, the top
  of the fraudulent list is a column of 64-character hashes — each one a
  single scammer the model would be memorising. Switch it to normalised and
  they collapse. That is decision 2 below, made visible rather than asserted.
- **Fraud rate by category** re-ranks across industry, function, employment
  type, experience and education without regenerating anything.

Charts sit **below** the data tables, after the numbers they summarise, and
each carries **PNG and SVG download buttons**. Both are produced in the
browser from the live SVG, so what is saved is exactly what is on screen —
no server round trip and no second rendering path to keep in step. The PNG
is rasterised at 2x on a canvas; if that fails the button falls back to the
SVG rather than doing nothing.

The static PNG gallery that used to sit at the bottom of this stage is gone:
it showed the same three analyses as the interactive charts, a step behind
and without the hover. The report figures are still written to `reports/` —
by the Report stage, from the same functions, so a figure and a chart cannot
disagree.

The two series colours are checked rather than chosen by eye. The Vega blue
the project started with (`#4C78A8`) sits below the chroma floor for a
categorical series — against white it reads as a desaturated grey rather than
as an identity colour — so it was re-stepped to `#2F6FB5`, which clears the
floor and widens colour-blind separation from the fraud coral (protanopia
dE 12.7 to 15.9, normal vision 25.8 to 29.7). The matplotlib palette was
changed to match, so the generated figures and the live charts use the same
two colours.

### Choosing the evaluation protocol

The split is a control, not a fixed description. A slider sets the training
share between 50% and 95%, and a proportion bar shows the division to scale
with the posting counts on it. Composition and leakage are previewed from
`/api/split/preview` before anything is trained.

Under the hood there are two paths, because a k-fold splitter can only
produce a test set of 1/k:

- A requested fraction that *is* 1/k for some k (50%, 33%, 25%, 20%, 10%...)
  uses `StratifiedGroupKFold`, the well-tested scikit-learn path. This keeps
  the default 80/20 split — and therefore every number in this README —
  exactly as it was.
- Anything else is packed by hand: whole groups are assigned to the test side,
  fraud-bearing groups first until it holds its share of the frauds, then
  all-legitimate groups until it holds its share of the rows. Taking the
  scarce class first is what keeps the ratio close; filling by size first
  leaves the positives to fall where they may. Measured across 10%-50%, it
  lands on the requested fraction exactly and within 0.03 percentage points
  of the dataset's 4.84% class balance.

**Grouping can be switched off**, and this is the most useful control on the
page. Turn it off and the panel reports what it costs:

| Split             | Test postings memorisable from training |
| ----------------- | --------------------------------------- |
| Grouped (default) | 0 (0.0%)                                |
| Ungrouped, 80/20  | 798 (22.3%)                             |
| Ungrouped, 60/40  | 1,512 (21.1%)                           |

Train on an ungrouped split and the scores rise. The results table says so
wherever those numbers appear, not only on the panel where the choice was
made — a table of scores read without its protocol is misleading, so the
protocol travels with the scores into the saved results.

### Keeping the page fast

The first version of the single page asked for everything in one `/api/state`
call and rendered nothing until it arrived. That call took **4.9 seconds**,
every time, and the browser showed a header above an empty page for the
duration. Five things were wrong, and all five are the same mistake —
computing something before knowing whether it is needed:

| Cause                                                                     | Fix                                                        |
| ------------------------------------------------------------------------- | ---------------------------------------------------------- |
| The split audit (4.3s of the 4.9s) ran on every page load, for a panel most visits never open | Moved to `/api/split/preview`, called by the Train panel   |
| `torch` was imported during template rendering, just to decide whether to show one checkbox — 2.5s before the first byte | `importlib.util.find_spec`, which checks without importing |
| `StratifiedGroupKFold` over 14,658 groups, recomputed on every preview     | Splits cached on (frame identity, fraction, grouping, seed) |
| The description column was normalised and factorised three times per audit | Group keys cached on the frame's identity                   |
| Reading the model's *name* unpickled the whole fitted pipeline             | A small JSON sidecar written beside the `.joblib`           |

Plus: the corpus is cached in memory rather than re-read from parquet, the
leakage count is an integer set operation on group ids rather than a pass
over 17,880 strings, and read-only callers skip the defensive copy.

The result, measured on the same machine and the same 17,880 rows:

| Request              | Before    | After          |
| -------------------- | --------- | -------------- |
| `GET /`              | 2,853 ms  | **1 ms**       |
| `GET /api/state`     | 4,900 ms  | **2 ms** warm  |
| `/api/split/preview` | 3,000 ms  | **3 ms** warm  |
| Shell visible        | never, until state arrived | **~250 ms** |

Two principles hold the improvement in place. **Panels fetch their own data**
the first time they are shown, so opening the app pays for the app and
nothing else. And **the shell renders before any request returns** — the rail
and a skeleton appear immediately, which is the difference between a page
that is loading and a page that looks broken.

Every cache is keyed on identity rather than content. Hashing the description
column costs 170 ms, a third of the work being cached, and a sampled
fingerprint could collide and return groups belonging to different data —
which would silently break the leakage guarantee, the one thing the splitting
module exists to provide. Object identity cannot collide.

### Starting over

A **Start over** button in the header clears everything the pipeline has
generated — the cleaned corpus, fitted models, saved results, report tables
and figures, and every in-memory cache — so the whole thing can be run again
from nothing.

It is the only code in the project that deletes anything, so it is
deliberately narrow:

- **The raw dataset is kept by default.** EMSCAD is not redistributable and
  cannot be fetched without a Kaggle session, which makes it the one artifact
  that cannot be regenerated from what remains. Removing it is a separate
  tick in the dialog, with its own warning.
- **The dialog itemises what will go** — each group with a file count and a
  size — before the confirm button, which names the exact number of files.
  A header button that wiped the project on one click would be a trap.
- **`.gitkeep` files survive.** They are what keep the empty directories in
  version control.
- **Symlinks are skipped, not followed.** Every path is resolved and checked
  to be inside the directory it was listed under, so a link in `models/`
  pointing somewhere else cannot be used to delete it.
- **Files outside a target's patterns survive.** Figures are declared as
  `*.png`; a hand-written note in that directory is not generated output.
- **It refuses while a job is running.** A worker part-way through writing a
  corpus would otherwise be left writing into a directory being emptied, and
  the half-file would fail to load later with no hint as to why.

Afterwards the directory tree is recreated, the job registry is cleared — so
nothing reads "training: done" beside a models directory with no model in it
— and the page returns to the Import stage.

## The embedding layer

TF-IDF treats every term as an isolated dimension: `recruit` and `hiring` are
as unrelated as `recruit` and `aardvark`. An embedding gives each vocabulary
term a dense vector trained *for this task*, so terms playing the same role in
fraudulent adverts end up near each other.

```
token ids (padded)                            (batch, 150)
  → nn.Embedding(20000, 64, padding_idx=0)    (batch, 150, 64)
  → pooling over time, padding masked out     (batch, pooled)
  → concat structured flags                   (batch, pooled + 24)
  → Linear → ReLU → Dropout → Linear          (batch, 1)
  → BCEWithLogitsLoss(pos_weight = neg/pos)
```

Five pooling modes are selectable in the interface. Four of them — mean, max,
additive attention, and a gated mean with one learned weight per vocabulary
term — are **order-invariant**: they produce the same vector for a document and
for its shuffled tokens. The fifth is a convolution.

### Getting it to work

The first version scored F1 0.693, well behind the linear baseline. Four
changes closed most of the gap, and each one is a result worth reporting:

| Change                                              | F1        | Avg. precision |
| --------------------------------------------------- | --------- | -------------- |
| Attention pooling, dropout after concatenation       | 0.693     | 0.745          |
| Dropout moved off the structured flags               | 0.676     | 0.756          |
| Gated pooling (one learned weight per term)          | 0.737     | 0.781          |
| Convolutional n-grams, 300 tokens                    | 0.709     | **0.817**      |
| Convolution + 150 tokens + tuned threshold           | **0.768** | **0.817**      |

1. **Dropout was being applied to the structured flags.** It sat after the
   pooled text and the flags were concatenated, so `has_company_logo` — the
   single strongest feature in the dataset — was zeroed 30% of the time while
   the linear models received it intact.

2. **Order-invariant pooling cannot see a phrase.** "No experience", "use
   link", "earn weekly" are exactly the signal, and TF-IDF captures them as
   bigrams. Adding a convolution over widths 1–4 (Kim, 2014) lifted average
   precision from 0.78 to 0.82 — the single largest gain, and direct evidence
   that word order was the missing information.

3. **Truncating to 150 tokens beat 300.** The median advert is 220 tokens; the
   pitch is in the opening, and the tail dilutes it. It also halves training
   time.

4. **Weight decay hurts badly.** Raising it from 1e-5 to 1e-3 cost 8 points of
   F1, and 1e-2 collapsed the model entirely (F1 0.286). Adam's decay shrinks
   every embedding row each step, including the rare terms that carry the most
   signal.

### The threshold is part of the model

`pos_weight` deliberately inflates the loss on the minority class, which shifts
the logit scale. Zero is no longer the F1-optimal boundary:

| Threshold                    | Precision | Recall | F1        |
| ---------------------------- | --------- | ------ | --------- |
| Default (0.0)                | 0.543     | 0.804  | 0.648     |
| Tuned on holdout (3.936)     | 0.899     | 0.671  | **0.768** |

Twelve points of F1 from the threshold alone. The cut is chosen on the
early-stopping holdout — data the model never trained on — so it costs nothing
extra and the test split stays untouched. Every model now carries its own
operating point, and the interface displays it: assuming 0.0 for a margin model
and 0.5 for a probability model mislabels whichever family it guesses wrong.

## Four decisions that shape the results

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
document frequencies are learned per fold. The embedding vocabulary is built
the same way — a vocabulary fitted on the whole corpus would tell the model
which terms exist in the test set. SMOTE is likewise applied inside the
pipeline, never before cross-validation.

### 4. SMOTE is incoherent for an embedding model

SMOTE interpolates between feature vectors. Between two sequences of integer
token ids, the midpoint is a sequence of ids that index arbitrary unrelated
words. The embedding model therefore **refuses SMOTE** and rebalances through
`pos_weight` in its loss instead; the interface records that the requested
treatment was not the one applied, rather than labelling the results with a
treatment that never ran.

## Results

### Single grouped split, all five models (`fjpd train`)

| Model              | Features   | Cut    | F1 (fraud) | Precision | Recall | Avg. precision |
| ------------------ | ---------- | ------ | ---------- | --------- | ------ | -------------- |
| **LinearSVC**      | tf-idf     | 0.000  | **0.8528** | 0.9085    | 0.8035 | 0.9150         |
| Embedding network  | embedding  | 3.936  | 0.7682     | 0.8992    | 0.6705 | 0.8169         |
| RandomForest       | tf-idf     | 0.500  | 0.7632     | 0.8855    | 0.6705 | 0.8563         |
| MultinomialNB      | tf-idf     | 0.500  | 0.7179     | 0.8058    | 0.6474 | 0.8359         |
| LogisticRegression | tf-idf     | 0.000  | 0.6880     | 0.5458    | 0.9306 | 0.8832         |

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

| Model                         | Threshold      | Accuracy | Precision | Recall | F1        | Avg. precision |
| ----------------------------- | -------------- | -------- | --------- | ------ | --------- | -------------- |
| Majority baseline             | n/a            | 0.9516   | 0.000     | 0.000  | 0.000     | 0.048          |
| **LinearSVC + class weight**  | default 0.000  | 0.9866   | 0.909     | 0.804  | **0.853** | 0.915          |
| LinearSVC + class weight      | tuned −0.082   | 0.9852   | 0.866     | 0.821  | 0.843     | 0.915          |
| LinearSVC + SMOTE             | default 0.000  | 0.9869   | 0.938     | 0.780  | 0.852     | **0.916**      |
| LinearSVC + SMOTE             | tuned −0.017   | 0.9866   | 0.931     | 0.780  | 0.849     | 0.916          |
| Embedding net + pos_weight    | default 0.000  | 0.9578   | 0.543     | 0.804  | 0.648     | 0.817          |
| Embedding net + pos_weight    | tuned 3.936    | 0.9804   | 0.899     | 0.671  | 0.768     | 0.817          |

For the linear models the decision threshold is tuned on **out-of-fold**
predictions. Tuning on in-sample training scores selects 0.831 and collapses
recall to 0.35, because a fitted model separates its own training data far more
confidently than unseen data. The embedding network reuses the threshold it
already chose on its early-stopping holdout: deriving it by cross-validation
would retrain the network five times for a value picked the same way.

### The honest headline

**A tuned sparse linear model beats a learned embedding network on this
dataset**, by 8 points of F1 and 10 of average precision. That is the expected
result at this scale and worth stating plainly: 17,880 documents containing 866
positives is far too little signal to fit 1.3M embedding parameters when a
30,000-dimension TF-IDF matrix with bigrams already separates the classes
almost linearly. Embeddings earn their keep on corpora orders of magnitude
larger, or when initialised from pretrained vectors.

The embedding model is still the second-best of the five, and it gets there
from a 64-dimension representation rather than 30,000 sparse features.

## What the models learn

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
the most influential features despite a modest coefficient. The Predict panel
reports contribution as value × weight for this reason.

### Three kinds of explanation

No single explanation covers every model, so the Predict panel offers what each
one can actually support:

- **Coefficients** (linear models) — contribution = feature value × weight.
- **Pooling weights** (embedding model) — which tokens the network attended to.
  On the scam example it concentrates on `home`, `work`, `recruit`, `appli`,
  `use`, `link`.
- **Flag sensitivity** (every model) — each structured flag flipped on its own,
  with the advert otherwise unchanged. Three extra forward passes, and it makes
  the Random Forest — which has neither coefficients nor attention —
  inspectable. It also answers the question a user actually asks: *what if this
  posting had a company logo?* (−0.29 on the margin, for the worked example.)

**A negative result:** the learned embedding neighbourhoods are **not**
semantically interpretable. After training, `recruit` sits beside `class` and
`southwest`; `softwar` beside `cbd` and `click`. Even restricted to terms
appearing in 25+ documents, the geometry carries no recognisable word
similarity — the objective rewards class separation, not similarity, and 866
positives cannot shape 1.3M parameters into a meaning space. This is written to
`reports/tables/embedding_neighbours.csv` by `fjpd evaluate` and deliberately
**not** shown beside a verdict, because displaying it there would imply the
model had learned word meaning when it has not.

## Tests

```bash
uv run pytest                      # 259 tests, ~110s
uv run pytest -m "not slow"        # skip the ones that train a model
```

Every test runs against a synthetic 200-row corpus in a temporary directory,
never the real CSV — so the suite passes on a clean checkout, where
`data/raw/` is empty because the dataset is not redistributable.

Coverage is aimed at the claims this README makes, and at the failures that
would be silent:

- **Leakage control** — duplicate descriptions share a group, no description
  text crosses the split, a deliberately contaminated split is rejected, CV
  folds are group-disjoint. If grouping broke, every number here would rise and
  nothing would look wrong.
- **Preprocessing** — redaction hashes collapse while the family token
  survives; placeholders are not stemmed or dropped as stopwords; a pasted
  advert is cleaned by exactly the same code path as the corpus (train/serve
  skew).
- **The embedding layer** — padding never changes a prediction, an all-padding
  document does not produce NaN in any pooling mode, the vocabulary is fitted
  on training documents only, training is reproducible, and a fitted model
  round-trips through joblib.
- **Input validation** — unknown model names, strategies and metrics are
  rejected rather than reaching a dictionary lookup and producing a 500.
- **The API** — an invalid upload leaves the existing dataset intact, only one
  job runs at a time, figures cannot be traversed out of their directory.
- **The chosen split** — an arbitrary ratio still keeps whole duplicate groups
  on one side and still lands within 0.04 of the class balance; 20% still
  routes through `StratifiedGroupKFold` so the reported numbers cannot drift;
  and the ungrouped split is asserted to *actually leak*, because that claim
  is the argument for the whole protocol.
- **Chart payloads** — every chart is JSON-serialisable and free of NaN, the
  histogram's bins align with its edges, and a term only appears on the side
  its sign supports.
- **Resetting** — the dataset survives unless explicitly included, `.gitkeep`
  survives, a symlink out of the project is not followed, a reset is refused
  while a job runs, and the state endpoint is genuinely empty afterwards.
- **The caches** — a cached corpus cannot be corrupted by a caller mutating
  what it was handed, one frame's group key is never returned for another,
  and the fast group-id leakage count is checked against the obvious
  string comparison. Also that `/api/state` still omits the expensive
  payloads and still does not import torch, because both regressions would
  be invisible except as a slow page.

## Weaknesses found and fixed in this rebuild

Auditing the previous version turned up ten defects. Each has a test:

| Problem                                                                                                   | Fix                                                                     |
| --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Upload wrote over `data/raw/` **before** validating, destroying a working dataset on a bad file            | Validate a staged copy, then rename into place                          |
| `class_balance` zipped a 2-element label list against `value_counts()` — any single-class frame raised     | Reindex over both classes; always two rows                              |
| Model names, strategies and metrics from the form went straight into dict lookups → 500s                   | Validated against the registry, with fallbacks                          |
| Threshold plots labelled 0.0 as "default" even for probability models, mislabelling the operating point    | Score kind carried with the scores; every model reports its own cut     |
| Two jobs could run at once and interleave matplotlib's global state, corrupting both figures               | One global job slot; `FIGURE_LOCK` around every drawing sequence        |
| `load_raw()` re-read a 50 MB CSV on every page render                                                      | Cache keyed on (size, mtime), invalidated by the upload                 |
| The embedding model could not be saved: its progress callback holds a lock and its network classes are local to a function | `__getstate__` drops the callback and persists the `state_dict`         |
| A failed `joblib.dump` overwrote a good model with a truncated file, surfacing later as `EOFError`         | Write to a temp file, rename on success                                 |
| The corpus was written in place, so a crash mid-write left an unreadable parquet                           | Same temp-file-and-rename pattern                                       |
| Progress was `(0, 1)` for the whole 30s cleaning step — a bar stuck at 0%                                  | Weighted nestable reporting, chunked cleaning, per-epoch model progress |
| Two embedding inputs had a `min`/`step`/`value` that no step could reach, so the browser refused to submit the training form and could not focus the offending field to say why (it sits inside a collapsed `<details>`) | Coherent step values, `novalidate` on the form, JS clamping to each input's own range |
| The page computed every panel's data before rendering anything, so it showed an empty screen for about five seconds on load | Panels fetch their own data when shown; the shell paints first (see [Keeping the page fast](#keeping-the-page-fast)) |
| `analysis.py` and the model sidecar bound their paths at import time, so a redirected project directory wrote to the real one | Paths resolved through their module, and patched in the test fixture |

Plus: the hard-coded Flask `SECRET_KEY` is now generated per process, the
figure route serves only PNGs, and `fjpd evaluate` no longer retrains the
network five times to recover a threshold it had already computed.

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
- The embedding layer is trained **from scratch**. Initialising from pretrained
  vectors (GloVe, fastText) is the obvious next step and the most likely way to
  close the remaining gap to the linear baseline.
- The web interface is a **single-user local tool**. The job registry is an
  in-memory dict, so it is not correct behind multiple workers.

## Attribution

Dataset: EMSCAD, Laboratory of Information & Communication Systems Security,
University of the Aegean.

Convolutional sentence classifier: Kim, Y. (2014), *Convolutional Neural
Networks for Sentence Classification*, EMNLP.
