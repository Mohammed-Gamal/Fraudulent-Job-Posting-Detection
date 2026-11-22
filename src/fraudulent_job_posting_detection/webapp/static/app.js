/* ==========================================================================
   Single-page controller.

   The server owns all state. This file does three things:

     1. fetch /api/state and render it,
     2. poll /api/jobs while something is running and drive the progress bar,
     3. post forms and re-render.

   No framework and no build step: the app is served from localhost and may
   have no network, so a CDN import would be a single point of failure for the
   whole interface.

   One rule throughout: text from the server goes into the DOM as `textContent`
   or through `el()`, never as an HTML string. The dataset is user-supplied, so
   a job title containing markup must not become markup.
   ========================================================================== */

"use strict";

const STAGES = [
  { id: "import", name: "Import", panel: "panel-import" },
  { id: "explore", name: "Explore", panel: "panel-explore" },
  { id: "clean", name: "Clean", panel: "panel-clean", job: "clean" },
  { id: "train", name: "Train", panel: "panel-train", job: "train" },
  { id: "predict", name: "Predict", panel: "panel-predict" },
  { id: "report", name: "Report", panel: "panel-report" },
];

const POLL_MS = 900;

const state = {
  server: null,
  stage: "import",
  polling: null,
  lastActive: null,
  // Panel payloads are fetched the first time their panel is shown and kept
  // until something invalidates them. The page used to receive all of this
  // in one /api/state call and render nothing until it arrived.
  panels: {},
  booted: false,
};

/* Fetch a panel's data once, render whenever asked.
   `force` re-fetches after a job changes the thing the panel shows. */
async function panelData(key, url, force = false) {
  if (!force && state.panels[key] !== undefined) return state.panels[key];
  if (state.panels[key] === null) return null; // a fetch is already in flight
  state.panels[key] = null;
  try {
    const payload = await api(url);
    state.panels[key] = payload;
    return payload;
  } catch {
    delete state.panels[key];
    return null;
  }
}

function invalidatePanels(...keys) {
  for (const key of keys) delete state.panels[key];
}

/* ----------------------------------------------------------- DOM helpers -- */
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value; // only ever literal markup
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const $ = (id) => document.getElementById(id);

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

function fmtInt(value) {
  return Number(value || 0).toLocaleString();
}

function fmtSeconds(value) {
  const total = Math.round(Number(value) || 0);
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  return `${minutes}m ${String(total % 60).padStart(2, "0")}s`;
}

function toast(message, tone = "info") {
  const box = $("toast");
  const item = el("div", { "data-tone": tone, text: message });
  box.append(item);
  // Errors stay until the next render; progress messages are transient.
  const life = tone === "error" ? 9000 : 4000;
  setTimeout(() => item.remove(), life);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const message =
      (payload && (payload.error || payload.message)) ||
      `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return payload;
}

/* -------------------------------------------------------------- the rail -- */
/* A stage's state is derived from what the server says exists, so the rail can
   never offer a step whose input has not been produced. */
function stageStatus(stage, server) {
  const jobs = server.jobs || {};
  const job = stage.job ? jobs[stage.job] : null;
  if (job && job.status === "running") return { tone: "running", text: "running" };

  switch (stage.id) {
    case "import":
      return server.dataset
        ? { tone: "done", text: "validated" }
        : { tone: "locked", text: "no dataset" };
    case "explore":
      return server.dataset
        ? { tone: "done", text: "ready" }
        : { tone: "locked", text: "needs a dataset" };
    case "clean":
      if (!server.dataset) return { tone: "locked", text: "needs a dataset" };
      return server.corpus_ready
        ? { tone: "done", text: "corpus built" }
        : { tone: "ready", text: "not built" };
    case "train":
      if (!server.corpus_ready) return { tone: "locked", text: "needs a corpus" };
      return server.results
        ? { tone: "done", text: "results ready" }
        : { tone: "ready", text: "not trained" };
    case "predict":
      return server.model && server.model.name
        ? { tone: "done", text: server.model.label || server.model.name }
        : { tone: "locked", text: "needs a model" };
    case "report": {
      if (!server.corpus_ready) return { tone: "locked", text: "needs a corpus" };
      const ran = Object.keys(server.analysis || {}).length;
      return ran
        ? { tone: "done", text: `${ran} of 4 stages run` }
        : { tone: "ready", text: "nothing generated" };
    }
    default:
      return { tone: "ready", text: "" };
  }
}

function renderRail(server) {
  const list = clear($("rail"));
  STAGES.forEach((stage, index) => {
    const status = stageStatus(stage, server);
    const locked = status.tone === "locked";
    const button = el(
      "button",
      {
        type: "button",
        "aria-current": state.stage === stage.id ? "true" : null,
        disabled: locked && state.stage !== stage.id,
      },
      [
        el("span", { class: "num", text: String(index + 1) }),
        el("span", { class: "name", text: stage.name }),
        el("span", { class: "state", "data-tone": status.tone, text: status.text }),
      ]
    );
    button.addEventListener("click", () => show(stage.id));
    list.append(el("li", {}, button));
  });

  const note = $("rail-note");
  const blocked = STAGES.find((s) => stageStatus(s, server).tone === "locked");
  note.textContent = blocked
    ? `${blocked.name} is waiting on an earlier stage.`
    : "Every stage has what it needs.";
}

function renderMasthead(server) {
  const box = clear($("masthead-status"));
  if (server.dataset) {
    box.append(
      el("span", {
        class: "chip",
        "data-tone": "ok",
        text: `${fmtInt(server.dataset.rows)} postings`,
      }),
      el("span", {
        class: "chip",
        text: `${server.dataset.fraud_pct}% fraudulent`,
      })
    );
  } else {
    box.append(
      el("span", { class: "chip", "data-tone": "missing", text: "no dataset" })
    );
  }
  if (server.corpus_ready) box.append(el("span", { class: "chip", text: "corpus built" }));
  if (server.model && server.model.name) {
    box.append(
      el("span", { class: "chip", text: `model: ${server.model.label || server.model.name}` })
    );
  }
}

function show(stageId) {
  state.stage = stageId;
  for (const stage of STAGES) {
    $(stage.panel).dataset.active = String(stage.id === stageId);
  }
  if (state.server) {
    renderRail(state.server);
    loadPanel(stageId);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* Pull in whatever the newly visible panel needs. Each of these is a few
   hundred milliseconds at most, and none of them runs for a panel the user
   never opens. */
async function loadPanel(stageId) {
  const server = state.server;
  if (!server) return;
  if (stageId === "explore" && server.dataset) {
    const payload = await panelData("explore", "/api/explore/tables");
    if (payload && state.server) renderExplore(state.server);
    refreshCharts();
  }
  if (stageId === "clean" && server.corpus_ready) {
    const payload = await panelData("sample", "/api/corpus/sample");
    if (payload && state.server) renderClean(state.server);
  }
  if (stageId === "train" && server.corpus_ready) previewSplit();
}

/* -------------------------------------------------------------- fragments - */
/* DataFrame columns are snake_case because the CLI prints them too. The page
   shows people words, not identifiers. */
const COLUMN_LABELS = {
  split: "Split",
  n_postings: "Postings",
  pct_of_dataset: "Share of corpus",
  n_fraudulent: "Fraudulent",
  pct_fraudulent: "Fraudulent %",
  n_unique_groups: "Distinct descriptions",
  column: "Column",
  n_missing: "Missing",
  pct_missing: "Missing %",
  field: "Field",
  value: "Value",
  fraud_rate_pct: "Fraud rate %",
  title: "Title",
  location: "Location",
  has_company_logo: "Logo",
  telecommuting: "Remote",
  fraudulent: "Fraudulent",
};

function columnLabel(name) {
  const key = String(name);
  if (COLUMN_LABELS[key]) return COLUMN_LABELS[key];
  return key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

function table(data, caption) {
  if (!data || !data.columns || !data.rows || !data.rows.length) return null;
  const head = el(
    "tr",
    {},
    data.columns.map((column) => el("th", { scope: "col", text: columnLabel(column) }))
  );
  const body = data.rows.map((row) =>
    el(
      "tr",
      {},
      row.map((cell) =>
        el("td", {
          class: typeof cell === "number" ? "num" : null,
          text: cell === null || cell === undefined ? "—" : String(cell),
        })
      )
    )
  );
  const node = el("table", {}, [
    caption ? el("caption", { text: caption }) : null,
    el("thead", {}, head),
    el("tbody", {}, body),
  ]);
  return el("div", { class: "scroller" }, node);
}

function gallery(figures) {
  if (!figures || !figures.length) return null;
  return el(
    "div",
    { class: "gallery" },
    figures.map((figure) =>
      el("figure", {}, [
        el("img", {
          src: `/figure/${encodeURIComponent(figure.name)}?v=${figure.version}`,
          alt: figure.caption,
          loading: "lazy",
        }),
        el("figcaption", { text: figure.caption }),
      ])
    )
  );
}

function provenance(data) {
  if (!data) {
    return el("div", {
      class: "notice",
      "data-tone": "warn",
      text:
        "No build record for the stored corpus. Rebuild it so the settings " +
        "that produced it are known — otherwise the methodology shown here " +
        "may not match the data on disk.",
    });
  }
  const bits = [`${fmtInt(data.rows)} postings`, `median ${data.median_tokens} tokens`];
  if (data.vocabulary_stems) bits.push(`${fmtInt(data.vocabulary_stems)} distinct stems`);
  return el("div", { class: "provenance" }, [
    el("div", {}, [
      el("b", { text: `Built ${data.built_at}` }),
      el("span", { class: "muted", text: ` — ${bits.join(", ")}` }),
    ]),
    el(
      "ul",
      {},
      (data.options || []).map((option) =>
        el("li", { "data-on": String(option.enabled), text: option.label })
      )
    ),
  ]);
}

/* A placeholder with the same footprint as the card it replaces, so the
   layout does not jump when the real content lands. */
function skeletonCard(label) {
  return el("div", { class: "card skeleton" }, [
    el("div", { class: "skeleton-bar", style: "width:38%" }),
    el("div", { class: "skeleton-bar", style: "width:85%" }),
    el("div", { class: "skeleton-bar", style: "width:72%" }),
    el("p", { class: "muted", style: "margin-top:.9rem", text: label }),
  ]);
}

function empty(title, body, actionLabel, stageId) {
  const node = el("div", { class: "empty" }, [
    el("b", { text: title }),
    el("span", { text: body }),
  ]);
  if (actionLabel && stageId) {
    const button = el("button", {
      class: "action",
      type: "button",
      text: actionLabel,
      style: "margin-top:.9rem",
    });
    button.addEventListener("click", () => show(stageId));
    node.append(el("div", {}, button));
  }
  return node;
}

/* ------------------------------------------------------------ 1  Import -- */
function renderImport(server) {
  const box = clear($("import-body"));

  if (!server.dataset) {
    const card = el("div", { class: "card" }, [
      el("h3", { text: "No valid dataset found" }),
      el("p", { class: "muted" }, [
        "Expected at ",
        el("code", { text: server.raw_path }),
      ]),
    ]);
    const form = el("form", { id: "upload-form" }, [
      el("label", { class: "field" }, [
        el("span", { text: "Upload fake_job_postings.csv" }),
        el("input", { type: "file", name: "csv", accept: ".csv", required: true }),
      ]),
      el("div", { class: "card-foot" }, [
        el("button", { class: "action", type: "submit", text: "Upload and validate" }),
        el("span", {
          class: "muted",
          text: "Checked against the published EMSCAD schema before it replaces anything.",
        }),
      ]),
    ]);
    form.addEventListener("submit", onUpload);
    card.append(form);
    card.append(
      el("p", { class: "muted", style: "margin-top:1rem" }, [
        "Download it from the ",
        el("a", {
          href: "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction",
          target: "_blank",
          rel: "noopener",
          text: "Kaggle mirror of EMSCAD",
        }),
        ", or copy it into data/raw/ directly.",
      ])
    );
    box.append(card);
    return;
  }

  const d = server.dataset;
  const card = el("div", { class: "card" }, [
    el("h3", { text: "Class balance" }),
    el("p", {
      class: "muted",
      text:
        "This ratio is why the project exists in the form it does. Every " +
        "later decision — the metric, the split, the imbalance treatment — " +
        "follows from it.",
    }),
  ]);

  // The hero: the imbalance drawn to scale.
  const fraudPct = Number(d.fraud_pct) || 0;
  const legitPct = Number(d.legit_pct) || 0;
  card.append(
    el("div", { class: "imbalance" }, [
      el("div", { class: "imbalance-bar" }, [
        el("span", {
          class: "legit",
          style: `width:${legitPct}%`,
          text: `${fmtInt(d.balance[0].count)} legitimate`,
        }),
        el("span", {
          class: "fraud",
          style: `width:${fraudPct}%`,
          title: `${fmtInt(d.n_fraud)} fraudulent`,
        }),
      ]),
      el("div", { class: "imbalance-legend" }, [
        el("span", {}, [el("b", { text: `${legitPct}%` }), " legitimate"]),
        el("span", {}, [
          el("b", { text: `${fraudPct}%` }),
          ` fraudulent (${fmtInt(d.n_fraud)} postings)`,
        ]),
      ]),
    ])
  );

  card.append(
    el("div", { class: "stat-row" }, [
      el("div", {}, [
        el("strong", { text: fmtInt(d.rows) }),
        el("span", { text: "postings" }),
      ]),
      el("div", {}, [
        el("strong", { text: String(d.columns) }),
        el("span", { text: "columns" }),
      ]),
      el("div", {}, [
        el("strong", { text: fmtInt(d.n_fraud) }),
        el("span", { text: "fraudulent" }),
      ]),
      el("div", {}, [
        el("strong", { text: `${legitPct}%` }),
        el("span", { text: "majority-class accuracy" }),
      ]),
    ])
  );

  card.append(
    el("p", { class: "muted" }, [
      "Accuracy is misleading at this ratio: always predicting ",
      el("em", { text: "legitimate" }),
      ` already scores ${legitPct}%. Model selection uses F1 and average ` +
        "precision on the fraudulent class instead.",
    ])
  );

  const next = el("div", { class: "card-foot" }, [
    el("button", { class: "action", type: "button", text: "Go to Explore" }),
    el("span", { class: "muted", text: "Replace the dataset below if you need to." }),
  ]);
  next.firstChild.addEventListener("click", () => show("explore"));
  card.append(next);
  box.append(card);

  const replace = el("details", { class: "tuning" }, [
    el("summary", { text: "Replace the dataset" }),
  ]);
  const form = el("form", { id: "upload-form" }, [
    el("label", { class: "field" }, [
      el("span", { text: "Upload a different CSV" }),
      el("input", { type: "file", name: "csv", accept: ".csv", required: true }),
    ]),
    el("button", { class: "action", type: "submit", text: "Upload and validate" }),
    el("p", {
      class: "muted",
      style: "margin-top:.7rem",
      text:
        "The upload is parsed and validated first. If it fails, your current " +
        "dataset is left exactly as it is.",
    }),
  ]);
  form.addEventListener("submit", onUpload);
  replace.append(form);
  box.append(replace);
}

/* ------------------------------------------------- interactive charts -- */
/* Per-chart options (which categorical field to rank, raw vs normalised
   vocabulary) live here rather than in the server state: they are a view
   preference, and round-tripping them would make every tick of a checkbox
   wait for a full state refresh. */
const chartOptions = {};

function selectedCharts() {
  return [...document.querySelectorAll('#chart-picker input[name="charts"]:checked')]
    .map((input) => input.value);
}

let chartToken = 0;

function chartBlock(key, meta, spec, token) {
  const block = el("div", { class: "chart-block" });
  const head = el("div", { class: "chart-head" }, [
    el("h4", { text: (spec && spec.title) || meta.label }),
  ]);
  const tools = el("div", { class: "chart-tools" });

  if (meta.options && meta.options.length) {
    const select = el("select", { "aria-label": `${meta.label} option` });
    for (const option of meta.options) {
      const node = el("option", { value: option, text: option.replace(/_/g, " ") });
      if ((chartOptions[key] || meta.options[0]) === option) node.selected = true;
      select.append(node);
    }
    select.addEventListener("change", () => {
      chartOptions[key] = select.value;
      refreshCharts();
    });
    tools.append(
      el("label", { class: "chart-control" }, [
        el("span", { text: key === "terms" ? "placeholders" : "field" }),
        select,
      ])
    );
  }
  head.append(tools);
  block.append(head);

  if (!spec || spec.error) {
    block.append(
      el("div", {
        class: "notice",
        "data-tone": "error",
        text: (spec && spec.error) || "No data for this chart.",
      })
    );
    return block;
  }

  if (spec.note) block.append(el("p", { class: "chart-note", text: spec.note }));

  const drawn = window.FJPDCharts.renderChart(spec);
  if (!drawn) {
    block.append(
      el("div", {
        class: "notice",
        "data-tone": "warn",
        text: `No renderer for a ${spec.kind} chart.`,
      })
    );
    return block;
  }

  block.append(drawn.svg);
  if (drawn.legend) block.append(drawn.legend);

  // Download the chart as drawn, rendered in the browser from the live SVG
  // so what is saved is exactly what is on screen.
  const filename = `${key}${chartOptions[key] ? `-${chartOptions[key]}` : ""}`;
  const png = el("button", { class: "action", type: "button", "data-variant": "quiet", text: "Download PNG" });
  png.addEventListener("click", () => window.FJPDCharts.downloadPng(drawn.svg, filename));
  const svg = el("button", { class: "action", type: "button", "data-variant": "quiet", text: "SVG" });
  svg.addEventListener("click", () => window.FJPDCharts.downloadSvg(drawn.svg, filename));
  tools.append(png, svg);
  return block;
}

async function refreshCharts() {
  const area = $("chart-area");
  if (!state.server || !state.server.dataset || state.stage !== "explore") {
    return;
  }
  const keys = selectedCharts();
  if (!keys.length) {
    clear(area).append(
      el("div", { class: "card" }, [
        empty("No charts selected", "Tick one above to draw it.", null, null),
      ])
    );
    return;
  }

  const token = ++chartToken;
  window.FJPDCharts.hideTip();
  const card = clear(area).appendChild(el("div", { class: "card" }));
  const catalogue = Object.fromEntries(
    (window.FJPD.charts || []).map((chart) => [chart.key, chart])
  );

  // One request per chart, rendered as it lands. Asking for all five at once
  // meant six seconds of blank card on a cold cache; this way the first
  // chart appears in well under a second and the rest fill in behind it.
  for (const key of keys) {
    const meta = catalogue[key] || { label: key, options: [] };
    const placeholder = el("div", { class: "chart-block" }, [
      el("div", { class: "chart-head" }, [el("h4", { text: meta.label })]),
      el("div", { class: "skeleton-bar", style: "width:100%;height:9rem" }),
    ]);
    card.append(placeholder);

    let payload;
    try {
      payload = await api("/api/explore/charts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ charts: [key], options: chartOptions }),
      });
    } catch (error) {
      if (token !== chartToken) return;
      placeholder.replaceWith(
        el("div", { class: "chart-block" }, [
          el("div", { class: "notice", "data-tone": "error", text: error.message }),
        ])
      );
      continue;
    }
    if (token !== chartToken) return;
    placeholder.replaceWith(chartBlock(key, meta, payload.charts[key], token));
  }
}

/* ----------------------------------------------------------- 2  Explore -- */
function renderExplore(server) {
  const box = clear($("explore-body"));
  const picker = $("chart-picker-card");
  picker.querySelectorAll("input").forEach((node) => {
    node.disabled = !server.dataset;
  });
  picker.style.display = server.dataset ? "" : "none";

  if (!server.dataset) {
    clear($("chart-area"));
    box.append(
      empty(
        "Nothing to explore yet",
        "Import and validate the dataset first.",
        "Go to Import",
        "import"
      )
    );
    return;
  }

  const payload = state.panels.explore;
  if (!payload) {
    box.append(skeletonCard("Reading the dataset…"));
    return;
  }

  const preview = table(payload.preview, "A sample of raw postings");
  if (preview) box.append(el("div", { class: "card" }, preview));
  for (const [name, data] of Object.entries(payload.tables || {})) {
    const node = table(data, name);
    if (node) box.append(el("div", { class: "card" }, node));
  }
}

/* ------------------------------------------------------------- 3  Clean -- */
function renderClean(server) {
  const box = clear($("clean-body"));
  const form = $("clean-form");
  const button = $("clean-run");
  button.textContent = server.corpus_ready ? "Rebuild corpus" : "Build corpus";

  if (!server.dataset) {
    form.querySelectorAll("input,button").forEach((node) => (node.disabled = true));
    box.append(
      empty(
        "No dataset to clean",
        "Import and validate the dataset first.",
        "Go to Import",
        "import"
      )
    );
    return;
  }
  form.querySelectorAll("input,button").forEach((node) => (node.disabled = false));

  // Reflect what actually built the corpus, so the controls cannot imply
  // settings that differ from the data on disk.
  if (server.provenance) {
    for (const option of server.provenance.options || []) {
      const input = form.querySelector(`input[name="${option.key}"]`);
      if (input) input.checked = option.enabled;
    }
  }

  if (!server.corpus_ready) {
    box.append(
      empty("No corpus yet", "Choose your settings above and build it.", null, null)
    );
    return;
  }

  box.append(
    el("div", { class: "card" }, [
      el("h3", { text: "What is on disk" }),
      provenance(server.provenance),
    ])
  );

  const samples = (state.panels.sample && state.panels.sample.sample) || [];
  if (samples.length) {
    const card = el("div", { class: "card" }, [
      el("h3", { text: "Cleaned output" }),
      el("p", {
        class: "muted",
        text: "Three postings at random, as the model will read them.",
      }),
    ]);
    for (const item of samples) {
      card.append(
        el("blockquote", { "data-label": item.label }, [
          el("span", { class: "tag", text: item.label }),
          `${item.text}…`,
        ])
      );
    }
    const foot = el("div", { class: "card-foot" }, [
      el("button", { class: "action", type: "button", text: "Go to Train" }),
    ]);
    foot.firstChild.addEventListener("click", () => show("train"));
    card.append(foot);
    box.append(card);
  }
}

/* ------------------------------------------------------------- 4  Train -- */
/* ------------------------------------------------- evaluation protocol -- */
/* The split controls own their own state: the slider, the grouping toggle and
   the seed are read on demand and previewed against the server, so the
   composition shown is always the one training would actually use. */
function splitConfig() {
  const trainPct = Number($("split-slider").value);
  return {
    test_fraction: Number(((100 - trainPct) / 100).toFixed(4)),
    grouped: $("split-grouped").checked,
    seed: Number($("split-seed").value) || 0,
  };
}

function paintProportion(trainPct, nTrain, nTest) {
  const testPct = 100 - trainPct;
  $("split-bar").querySelector(".train").style.width = `${trainPct}%`;
  $("split-bar").querySelector(".test").style.width = `${testPct}%`;
  $("split-train-pct").textContent = `${trainPct}% train`;
  $("split-test-pct").textContent = `${testPct}% test`;
  $("split-train-n").textContent = nTrain ? `${fmtInt(nTrain)} postings` : "";
  $("split-test-n").textContent = nTest ? `${fmtInt(nTest)} postings` : "";
}

let splitPreviewToken = 0;

async function previewSplit() {
  const config = splitConfig();
  const trainPct = Math.round((1 - config.test_fraction) * 100);
  paintProportion(trainPct, null, null);

  const readout = $("split-readout");
  const warning = clear($("split-warning"));
  const tableBox = clear($("split-table"));

  if (!state.server || !state.server.corpus_ready) {
    readout.textContent = "Build the corpus to preview the split.";
    return;
  }

  readout.textContent = "Composing the split…";
  const token = ++splitPreviewToken;
  let payload;
  try {
    payload = await api("/api/split/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
  } catch (error) {
    if (token !== splitPreviewToken) return;
    readout.textContent = `Could not preview the split: ${error.message}`;
    return;
  }
  // A slower earlier request must not overwrite a newer answer.
  if (token !== splitPreviewToken) return;

  state.split = payload;
  const p = payload.provenance;
  paintProportion(trainPct, payload.n_train, payload.n_test);

  readout.textContent =
    `${p.method}. Test set is ${(p.actual_test_fraction * 100).toFixed(1)}% of ` +
    `the corpus, ${payload.table.rows[1][3]} of them fraudulent ` +
    `(${payload.table.rows[1][4]}% against ${payload.table.rows[0][4]}% in training).`;

  if (p.grouped) {
    warning.append(
      el("div", {
        class: "notice",
        "data-tone": "success",
        text:
          "Group-disjoint: no test advert shares its description with anything " +
          "in training, so nothing in the test score is memorised.",
      })
    );
  } else {
    warning.append(
      el("div", {
        class: "notice",
        "data-tone": "error",
        text:
          `Leaking: ${fmtInt(p.n_memorisable_postings)} test postings ` +
          `(${p.pct_test_memorisable}%) repeat a description the model will ` +
          "have trained on. Scores from this split measure memorisation as " +
          "much as detection — useful to see once, not to report.",
      })
    );
  }
  const node = table(payload.table);
  if (node) tableBox.append(node);
}

function renderTrainProtocol(server) {
  const card = $("protocol-card");
  const controls = card.querySelectorAll("input");
  controls.forEach((node) => (node.disabled = !server.corpus_ready));

  const box = clear($("train-protocol"));
  if (!server.corpus_ready) {
    paintProportion(Number($("split-slider").value), null, null);
    $("split-readout").textContent = "Build the corpus to preview the split.";
    box.append(
      empty(
        "No cleaned corpus",
        "Build the corpus before training.",
        "Go to Clean",
        "clean"
      )
    );
    return false;
  }
  if (server.provenance) box.append(el("div", { class: "card" }, provenance(server.provenance)));
  return true;
}

function renderTrain(server) {
  const ready = renderTrainProtocol(server);
  const form = $("train-form");
  form
    .querySelectorAll("input,button,select")
    .forEach((node) => {
      if (!node.dataset.permanentlyDisabled) node.disabled = !ready;
    });
  // Models the environment cannot provide stay disabled regardless.
  for (const model of window.FJPD.models) {
    if (model.available) continue;
    const input = form.querySelector(`input[value="${model.name}"]`);
    if (input) {
      input.disabled = true;
      input.dataset.permanentlyDisabled = "true";
    }
  }

  updateStrategyNote();
  updateEstimate();

  const box = clear($("train-body"));
  const results = server.results;
  if (!results) {
    if (ready) {
      box.append(
        empty(
          "No results yet",
          "Pick your models above and train. Each one is scored once on the " +
            "held-out split.",
          null,
          null
        )
      );
    }
    return;
  }

  const metricKeys = (results.metrics || ["f1_fraud"]).filter(
    (key) => key in window.FJPD.metrics
  );
  const rankLabel = window.FJPD.metrics[results.rank_by] || results.rank_by;

  for (const note of results.notes || []) {
    box.append(el("div", { class: "notice", "data-tone": "warn", text: note }));
  }

  // A results table read without its protocol is misleading, so an ungrouped
  // run is labelled wherever its scores appear, not only where it was chosen.
  const protocol = results.split;
  if (protocol && !protocol.grouped) {
    box.append(
      el("div", {
        class: "notice",
        "data-tone": "error",
        text:
          "These scores come from an ungrouped split: " +
          `${fmtInt(protocol.n_memorisable_postings)} test postings ` +
          `(${protocol.pct_test_memorisable}%) repeated a description from ` +
          "training. They are inflated and should not be reported.",
      })
    );
  }

  /* Podium */
  const podiumCard = el("div", { class: "card" }, [
    el("h3", { text: "Ranking" }),
    el("p", {
      class: "muted",
      text:
        `Ranked by ${rankLabel}, imbalance treatment ` +
        `"${window.FJPD.strategyLabels[results.strategy] || results.strategy}", ` +
        `trained on ${fmtInt(results.n_train)} and scored on ` +
        `${fmtInt(results.n_test)} held-out postings` +
        (protocol ? ` (${protocol.method}).` : "."),
    }),
  ]);
  podiumCard.append(
    el(
      "ol",
      { class: "podium" },
      (results.podium || []).map((row, index) =>
        el("li", {}, [
          el("span", { class: "rank", text: String(index + 1) }),
          el("span", { class: "who" }, [
            el("b", { text: row.label || row.model }),
            el("span", {
              text: [
                row.family === "embedding" ? "learned embedding" : "sparse features",
                window.FJPD.strategyLabels[row.strategy] || row.strategy,
                `cut at ${row.threshold}`,
              ].join(" · "),
            }),
          ]),
          el("span", { class: "score", text: String(row[results.rank_by]) }),
        ])
      )
    )
  );
  podiumCard.append(
    el("p", { class: "muted" }, [
      "The winner is saved and used by ",
      (() => {
        const link = el("a", { href: "#", text: "Screen an advert" });
        link.addEventListener("click", (event) => {
          event.preventDefault();
          show("predict");
        });
        return link;
      })(),
      ".",
    ])
  );
  box.append(podiumCard);

  /* Full results table */
  const header = el("tr", {}, [
    el("th", { scope: "col", text: "Model" }),
    el("th", { scope: "col", text: "Features" }),
    el("th", { scope: "col", text: "Cut" }),
    ...metricKeys.map((key) =>
      el("th", { scope: "col", text: window.FJPD.metrics[key] })
    ),
  ]);
  const body = (results.rows || []).map((row, index) =>
    el("tr", { "data-best": String(index === 0) }, [
      el("td", { text: row.label || row.model }),
      el("td", { text: row.family === "embedding" ? "embedding" : "tf-idf" }),
      el("td", { class: "num", text: String(row.threshold) }),
      ...metricKeys.map((key) => el("td", { class: "num", text: String(row[key]) })),
    ])
  );
  const resultsCard = el("div", { class: "card" }, [
    el("h3", { text: "All results" }),
    el("p", {
      class: "muted",
      text:
        "“Cut” is the decision threshold each model actually applies. A margin " +
        "model cuts at 0; the embedding network tunes its own cut on a holdout, " +
        "so comparing its score against 0 would misstate it.",
    }),
    el("div", { class: "scroller" }, el("table", {}, [
      el("thead", {}, header),
      el("tbody", {}, body),
    ])),
    el("p", {
      class: "muted",
      style: "margin-top:.9rem",
    }, [
      "One grouped split, for responsiveness. For the cross-validated figures " +
        "that belong in a report, run ",
      el("code", { text: "uv run fjpd compare" }),
      ".",
    ]),
  ]);
  box.append(resultsCard);

  /* Embedding convergence */
  if (results.embedding_history && results.embedding_history.length) {
    const history = results.embedding_history;
    const best = history.reduce(
      (a, b) => (b.val_avg_precision > a.val_avg_precision ? b : a),
      history[0]
    );
    box.append(
      el("div", { class: "card" }, [
        el("h3", { text: "Embedding layer training" }),
        el("p", {
          class: "muted",
          text:
            `${history.length} epochs ran; epoch ${best.epoch} scored best on ` +
            `the holdout (average precision ${best.val_avg_precision}) and its ` +
            "weights were kept. Training stops when the holdout stops improving, " +
            "so the epoch count is chosen by the data rather than fixed.",
        }),
        table(
          {
            columns: ["Epoch", "Training loss", "Validation avg. precision"],
            rows: history.map((row) => [
              row.epoch,
              row.train_loss,
              row.val_avg_precision,
            ]),
          },
          null
        ),
      ])
    );
  }

  /* Diagnostics */
  const gal = gallery(server.train_figures || []);
  if (gal) {
    box.append(
      el("div", { class: "card" }, [
        el("h3", { text: "Diagnostics for the best model" }),
        el("p", {
          class: "muted",
          text:
            "A table of numbers hides the trade-off. The threshold plot shows " +
            "what each point of recall costs in precision.",
        }),
        gal,
      ])
    );
  }
}

function updateStrategyNote() {
  const form = $("train-form");
  const note = $("strategy-note");
  if (!form || !note) return;
  const strategy = form.querySelector('input[name="strategy"]:checked');
  const selected = [...form.querySelectorAll('input[name="models"]:checked')].map(
    (input) => input.value
  );
  const clashing = window.FJPD.models.filter(
    (model) => selected.includes(model.name) && !model.supports_smote
  );
  if (strategy && strategy.value === "smote" && clashing.length) {
    note.textContent =
      `${clashing.map((m) => m.label).join(", ")} cannot use SMOTE — ` +
      "interpolating between token ids produces ids that mean nothing. It will " +
      "use loss weighting instead, and the results table will say so.";
  } else if (strategy && strategy.value === "none") {
    note.textContent =
      "No treatment. At a 4.84% positive rate this favours precision heavily " +
      "and recall collapses.";
  } else {
    note.textContent = "";
  }
}

function updateEstimate() {
  const form = $("train-form");
  const target = $("train-estimate");
  if (!form || !target) return;
  const selected = [...form.querySelectorAll('input[name="models"]:checked')].map(
    (input) => input.value
  );
  if (!selected.length) {
    target.textContent = "Select at least one model.";
    return;
  }
  const cost = window.FJPD.models
    .filter((model) => selected.includes(model.name))
    .reduce((total, model) => total + model.cost, 0);
  // ~15s per unit cost, measured on a laptop CPU.
  const seconds = Math.round(cost * 15);
  const hasEmbedding = window.FJPD.models.some(
    (model) => selected.includes(model.name) && model.family === "embedding"
  );
  target.textContent =
    `${selected.length} model${selected.length > 1 ? "s" : ""}, roughly ` +
    `${fmtSeconds(seconds)}` +
    (hasEmbedding ? ". The embedding layer trains on CPU and is the slow one." : ".");
}

/* ----------------------------------------------------------- 5  Predict -- */
function renderPredict(server) {
  const box = clear($("predict-body"));
  if (!server.model || !server.model.name) {
    box.append(
      empty(
        "No saved model",
        "Screening uses the best model from your last training run.",
        "Go to Train",
        "train"
      )
    );
    return;
  }

  const options = server.cleaning_options_in_use || {};
  const enabled = Object.entries(options)
    .filter(([, on]) => on)
    .map(([key]) => key.replace(/_/g, " "));

  const card = el("div", { class: "card" }, [
    el("h3", { text: `Using ${server.model.label || server.model.name}` }),
    el("p", {
      class: "muted",
      text:
        "The best model from your last training run. Text is cleaned with " +
        (enabled.length ? enabled.join(", ") : "no normalisation") +
        " — the same settings that built the corpus.",
    }),
  ]);

  const form = el("form", { id: "predict-form" });
  const ex = window.FJPD.example;
  form.append(
    el("label", { class: "field" }, [
      el("span", { text: "Job title" }),
      el("input", { type: "text", name: "title", value: ex.title || "" }),
    ])
  );
  const description = el("textarea", { name: "description", rows: "6", required: true });
  description.value = ex.description || "";
  form.append(
    el("label", { class: "field" }, [
      el("span", {}, ["Description ", el("em", { text: "(required)" })]),
      description,
    ])
  );

  const grid = el("div", { class: "grid-2" });
  for (const [field, label, rows] of [
    ["company_profile", "Company profile", 3],
    ["requirements", "Requirements", 3],
    ["benefits", "Benefits", 3],
  ]) {
    const area = el("textarea", { name: field, rows: String(rows) });
    area.value = ex[field] || "";
    grid.append(el("label", { class: "field" }, [el("span", { text: label }), area]));
  }
  form.append(grid);

  const flags = el("fieldset", {}, [
    el("legend", { text: "Structured signals" }),
    el("p", {
      class: "muted",
      text:
        "These matter more than most words: postings without a company logo " +
        "are fraudulent 15.9% of the time against 2.0% with one. Toggle them " +
        "and watch the verdict move.",
    }),
  ]);
  for (const [flag, label] of [
    ["has_company_logo", "Has company logo"],
    ["has_questions", "Has screening questions"],
    ["telecommuting", "Telecommuting"],
  ]) {
    flags.append(
      el("label", { class: "pick" }, [
        el("input", { type: "checkbox", name: flag }),
        label,
      ])
    );
  }
  form.append(flags);

  form.append(
    el("div", { class: "card-foot" }, [
      el("button", { class: "action", type: "submit", text: "Screen this advert" }),
      el("span", { class: "muted", text: "Scored immediately; nothing is stored." }),
    ])
  );
  form.addEventListener("submit", onPredict);
  card.append(form);
  box.append(card);
  box.append(el("div", { id: "predict-result" }));
}

function renderVerdict(payload) {
  const box = clear($("predict-result"));
  const r = payload.result;
  const isFraud = Boolean(r.is_fraud);

  // Place the score on a scale around the model's own cut point, so the number
  // is readable without knowing the model family.
  const cut = Number(r.threshold) || 0;
  const span = Math.max(Math.abs(r.score - cut) * 1.6, 2);
  const low = cut - span;
  const high = cut + span;
  const position = ((r.score - low) / (high - low)) * 100;
  const cutPosition = ((cut - low) / (high - low)) * 100;

  const verdict = el("div", { class: "verdict", "data-fraud": String(isFraud) }, [
    el("div", {}, [
      el("div", { class: "verdict-word", text: r.label }),
      el("div", {
        class: "verdict-sub",
        text:
          `${r.score_kind} ${r.score}, cut at ${cut} · ` +
          `${r.n_tokens} tokens after cleaning · ${payload.model_label}`,
      }),
    ]),
    el("div", { class: "gauge" }, [
      el("div", { class: "gauge-track" }, [
        el("div", { class: "gauge-cut", style: `left:${cutPosition}%` }),
        el("div", {
          class: "gauge-dot",
          style: `left:${Math.max(0, Math.min(100, position))}%`,
        }),
      ]),
      el("div", { class: "gauge-scale" }, [
        el("span", { text: low.toFixed(1) }),
        el("span", { text: "cut" }),
        el("span", { text: high.toFixed(1) }),
      ]),
    ]),
  ]);

  const card = el("div", { class: "card" }, [el("h3", { text: "Verdict" }), verdict]);

  if (r.score_kind === "margin") {
    card.append(
      el("p", {
        class: "muted",
        text:
          `Squashed to ${r.confidence} on a 0–1 scale for readability, where ` +
          "0.5 sits exactly on the cut. That is a rescaled margin, not a " +
          "calibrated probability — do not read it as “a 90% chance of fraud”.",
      })
    );
  }

  /* Linear contributions */
  if (r.contributions && r.contributions.length) {
    card.append(
      el("h4", { text: "What drove this decision", style: "margin-top:1.5rem" }),
      el("p", {
        class: "muted",
        text:
          "Contribution is the feature's value multiplied by its weight, not " +
          "the weight alone: a heavily weighted term that barely appears " +
          "contributes little.",
      }),
      table({
        columns: ["Feature", "Value", "Weight", "Contribution", "Pushes toward"],
        rows: r.contributions.map((item) => [
          item.feature,
          item.value,
          item.coefficient,
          item.contribution,
          item.pushes,
        ]),
      })
    );
  }

  /* Attention / pooling weights */
  if (r.attention && r.attention.length) {
    card.append(
      el("h4", { text: "Where the embedding layer looked", style: "margin-top:1.5rem" }),
      el("p", {
        class: "muted",
        text:
          "Pooling weight per token, summed over repeats. This is the " +
          "embedding model's counterpart to a coefficient: it says which words " +
          "the network attended to, though not which way they pushed.",
      })
    );
    card.append(
      el(
        "div",
        { class: "tokens" },
        r.attention.map((item) =>
          el("span", { class: "token", title: `${item.occurrences} occurrence(s)` }, [
            item.token,
            el("b", { text: `${item.share}%` }),
          ])
        )
      )
    );
  }

  /* Flag sensitivity: works for every model family */
  if (r.sensitivity && r.sensitivity.length) {
    card.append(
      el("h4", { text: "What if the flags were different", style: "margin-top:1.5rem" }),
      el("p", {
        class: "muted",
        text:
          "Each flag flipped on its own, with the advert otherwise unchanged. " +
          "This works for any model, including those with no coefficients.",
      })
    );
    card.append(
      el(
        "ul",
        { class: "sensitivity" },
        r.sensitivity.map((item) => {
          const direction =
            item.direction === "more suspicious"
              ? "up"
              : item.direction === "less suspicious"
                ? "down"
                : "flat";
          return el("li", {}, [
            el("span", {
              text: `${item.label}: ${item.current ? "on" : "off"} → ${
                item.if_flipped ? "on" : "off"
              }`,
            }),
            el("span", { class: "muted", text: item.direction }),
            el("span", {
              class: "delta",
              "data-dir": direction,
              text: `${item.delta > 0 ? "+" : ""}${item.delta}`,
            }),
          ]);
        })
      )
    );
  }

  card.append(
    el("h4", { text: "Text the model actually read", style: "margin-top:1.5rem" }),
    el("blockquote", { text: r.cleaned_text || "(empty after cleaning)" })
  );

  box.append(card);
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}


/* ------------------------------------------------------------ 6  Report -- */
/* The stages that used to be command-line only. Each is a background job with
   its own controls; results and the figures they write are shown inline. */
const reportParams = { folds: 5, restart: false, feature_sets: null, models: null };

function reportControls(stage) {
  const wrap = el("div", { class: "report-controls" });
  const controls = stage.controls || [];

  if (controls.includes("models")) {
    const group = el("fieldset", {}, [el("legend", { text: "Models" })]);
    for (const model of window.FJPD.models) {
      const input = el("input", {
        type: "checkbox",
        value: model.name,
        name: `${stage.key}-models`,
        disabled: !model.available,
      });
      input.checked = model.available;
      group.append(el("label", { class: "pick" }, [input, model.label]));
    }
    wrap.append(group);
  }

  if (controls.includes("folds")) {
    const input = el("input", {
      type: "number", min: "2", max: "10", value: String(reportParams.folds),
      name: `${stage.key}-folds`,
    });
    wrap.append(
      el("label", { class: "field compact" }, [
        el("span", { text: "Cross-validation folds" }),
        input,
      ])
    );
  }

  if (controls.includes("feature_sets")) {
    const group = el("fieldset", {}, [el("legend", { text: "Feature sets" })]);
    for (const label of ["tfidf + structured", "tfidf only"]) {
      const input = el("input", {
        type: "checkbox", value: label, name: `${stage.key}-features`,
      });
      input.checked = true;
      group.append(el("label", { class: "pick" }, [input, label]));
    }
    wrap.append(group);
  }

  if (controls.includes("restart")) {
    const input = el("input", {
      type: "checkbox", name: `${stage.key}-restart`,
    });
    wrap.append(
      el("label", { class: "pick" }, [
        input,
        "Start fresh (discard the resumable results on disk)",
      ])
    );
  }
  return wrap;
}

function readReportParams(stage, root) {
  const params = {};
  const pick = (name) => root.querySelectorAll(`[name="${stage.key}-${name}"]`);
  const models = [...pick("models")].filter((i) => i.checked).map((i) => i.value);
  if (models.length) params.models = models;
  const folds = root.querySelector(`[name="${stage.key}-folds"]`);
  if (folds) params.folds = Number(folds.value) || 5;
  const features = [...pick("features")].filter((i) => i.checked).map((i) => i.value);
  if (features.length) params.feature_sets = features;
  const restart = root.querySelector(`[name="${stage.key}-restart"]`);
  if (restart) params.restart = restart.checked;
  return params;
}

function renderReport(server) {
  const box = clear($("report-body"));
  if (!server.corpus_ready) {
    box.append(
      empty(
        "No cleaned corpus",
        "These analyses all run over the cleaned corpus. Build it first.",
        "Go to Clean",
        "clean"
      )
    );
    return;
  }

  const stored = server.analysis || {};
  const jobsState = server.jobs || {};
  const busy = Boolean(server.active_job);

  for (const stage of window.FJPD.analysisStages) {
    const card = el("div", { class: "card" }, [
      el("div", { class: "chart-head" }, [
        el("h3", { text: stage.label }),
        el("span", { class: "muted", text: stage.cost }),
      ]),
      el("p", { class: "muted", text: stage.blurb }),
    ]);

    const controls = reportControls(stage);
    if (controls.childElementCount) card.append(controls);

    const run = el("button", {
      class: "action",
      type: "button",
      text: stored[stage.key] ? "Run again" : "Run",
      disabled: busy,
    });
    run.addEventListener("click", () =>
      startJob(stage.key, readReportParams(stage, card))
    );

    const foot = el("div", { class: "card-foot" }, [run]);
    const job = jobsState[stage.key];
    if (stored[stage.key]) {
      foot.append(
        el("span", { class: "muted", text: `Last run ${stored[stage.key].ran_at}` })
      );
    } else if (job && job.status === "error") {
      foot.append(el("span", { class: "muted", text: `Failed: ${job.error}` }));
    }
    card.append(foot);

    const result = stored[stage.key];
    if (result && result.table) {
      const node = table(result.table);
      if (node) card.append(el("div", { style: "margin-top:1.1rem" }, node));
    }
    if (result && result.figures && result.figures.length) {
      const figures = (server.analysis_figures || []).filter((item) =>
        result.figures.includes(item.name)
      );
      const gal = gallery(figures);
      if (gal) card.append(el("div", { style: "margin-top:1.1rem" }, gal));
    }
    if (result && result.tables && result.tables.length) {
      card.append(
        el("p", { class: "muted", style: "margin-top:.9rem" }, [
          `${result.tables.length} tables written to reports/tables — `,
          el("code", { text: "CSV and Markdown" }),
          ", ready to paste into a write-up.",
        ])
      );
    }
    box.append(card);
  }
}

/* ------------------------------------------------------------- progress -- */
function renderJobbar(server) {
  const jobs = server.jobs || {};
  const activeName = server.active_job;
  const bar = $("jobbar");
  const job = activeName ? jobs[activeName] : null;

  if (!job || job.status !== "running") {
    bar.dataset.visible = "false";
    return;
  }

  bar.dataset.visible = "true";
  $("jobbar-stage").textContent = activeName;
  $("jobbar-step").textContent = job.label ? `— ${job.label}` : "";
  $("jobbar-elapsed").textContent = fmtSeconds(job.elapsed_seconds);

  const percent = Number(job.percent) || 0;
  $("jobbar-percent").textContent = `${percent.toFixed(0)}%`;
  const fill = $("jobbar-fill");
  // Zero percent with no label yet means the worker has not reported: show
  // motion rather than an empty bar that looks stalled.
  const indeterminate = percent <= 0;
  fill.dataset.indeterminate = String(indeterminate);
  if (!indeterminate) fill.style.width = `${percent}%`;

  const track = $("jobbar-track");
  track.setAttribute("aria-valuenow", String(Math.round(percent)));
  track.setAttribute("aria-valuetext", `${Math.round(percent)}% — ${job.label || ""}`);

  const cancel = $("jobbar-cancel");
  cancel.disabled = Boolean(job.cancel_requested);
  cancel.textContent = job.cancel_requested ? "Cancelling…" : "Cancel";
  cancel.onclick = async () => {
    cancel.disabled = true;
    try {
      const result = await api(`/api/cancel/${activeName}`, { method: "POST" });
      toast(result.message || "Cancelling…");
    } catch (error) {
      toast(error.message, "error");
      cancel.disabled = false;
    }
  };
}

function ensurePolling(server) {
  const running = Boolean(server.active_job);
  if (running && !state.polling) {
    state.polling = setInterval(poll, POLL_MS);
  } else if (!running && state.polling) {
    clearInterval(state.polling);
    state.polling = null;
  }
}

async function poll() {
  let payload;
  try {
    payload = await api("/api/jobs");
  } catch {
    return; // transient; the next tick will try again
  }
  const previous = state.lastActive;
  state.server.jobs = payload.jobs;
  state.server.active_job = payload.active;
  renderJobbar(state.server);
  renderRail(state.server);

  if (previous && !payload.active) {
    // A job just finished: report how it ended and reload everything it touched.
    const job = payload.jobs[previous];
    if (job) {
      if (job.status === "error") toast(`${previous} failed: ${job.error}`, "error");
      else if (job.status === "cancelled") toast(`${previous} cancelled.`);
      else toast(`${previous} finished in ${fmtSeconds(job.elapsed_seconds)}.`, "success");
    }
    state.lastActive = null;
    // A finished job changes what the panels show, so their cached payloads
    // are dropped rather than left stale.
    if (previous === "clean") invalidatePanels("sample", "explore");
    if (previous === "report_eda") invalidatePanels("explore");
    state.panels = previous === "clean" ? {} : state.panels;
    await refresh();
    if (previous === "clean" || previous.startsWith("report_")) refreshCharts();
    return;
  }
  state.lastActive = payload.active || previous;
  ensurePolling(state.server);
}

/* --------------------------------------------------------------- actions -- */
async function startJob(stage, body) {
  try {
    const result = await api(`/api/run/${stage}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    toast(result.message || `${stage} started.`);
    state.lastActive = stage;
    await refresh();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function onUpload(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const input = form.querySelector('input[type="file"]');
  if (!input.files || !input.files.length) {
    toast("Choose a CSV first.", "error");
    return;
  }
  const data = new FormData();
  data.append("csv", input.files[0]);
  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "Validating…";
  try {
    const result = await api("/api/upload", { method: "POST", body: data });
    toast(result.message, "success");
    await refresh();
  } catch (error) {
    toast(error.message, "error");
    button.disabled = false;
    button.textContent = "Upload and validate";
  }
}

function onCleanSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const options = {};
  for (const input of form.querySelectorAll('input[type="checkbox"]')) {
    options[input.name] = input.checked;
  }
  startJob("clean", { options });
}

const EMBEDDING_FIELDS = [
  "pooling",
  "embedding_dim",
  "sequence_length",
  "vocabulary_size",
  "max_epochs",
  "dropout",
  "learning_rate",
  "n_filters",
];

/* Read the embedding controls, correcting anything out of range in place.
   The form is `novalidate` on purpose: these inputs live inside a collapsed
   <details>, and a browser refusing to submit because of one of them cannot
   focus it to say so — it just reports "an invalid form control is not
   focusable" and nothing happens. The server clamps every value anyway, so
   the honest behaviour is to snap the field to its own range, show the user
   what it became, and submit. */
function readEmbeddingSettings(form) {
  const settings = {};
  const corrected = [];
  for (const name of EMBEDDING_FIELDS) {
    const field = form.querySelector(`[name="${name}"]`);
    if (!field) continue;
    if (field.type !== "number") {
      if (field.value !== "") settings[name] = field.value;
      continue;
    }
    if (field.value === "") continue;
    const min = field.min === "" ? -Infinity : Number(field.min);
    const max = field.max === "" ? Infinity : Number(field.max);
    let value = Number(field.value);
    if (!Number.isFinite(value)) continue;
    if (value < min || value > max) {
      value = Math.min(Math.max(value, min), max);
      field.value = String(value);
      corrected.push(`${name.replace(/_/g, " ")} → ${value}`);
    }
    settings[name] = value;
  }
  if (corrected.length) {
    toast(`Adjusted to the allowed range: ${corrected.join(", ")}.`, "warn");
  }
  return settings;
}

function onTrainSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const models = [...form.querySelectorAll('input[name="models"]:checked')].map(
    (input) => input.value
  );
  if (!models.length) {
    toast("Select at least one model.", "error");
    return;
  }
  const metrics = [...form.querySelectorAll('input[name="metrics"]:checked')].map(
    (input) => input.value
  );
  const strategy =
    (form.querySelector('input[name="strategy"]:checked') || {}).value || "class_weight";
  const embedding = readEmbeddingSettings(form);
  startJob("train", {
    models,
    metrics,
    strategy,
    rank_by: form.querySelector('[name="rank_by"]').value,
    embedding,
    split: splitConfig(),
  });
}

async function onPredict(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button");
  const body = {};
  for (const field of window.FJPD.textFields) {
    const node = form.querySelector(`[name="${field}"]`);
    body[field] = node ? node.value : "";
  }
  for (const flag of window.FJPD.binaryFlags) {
    const node = form.querySelector(`[name="${flag}"]`);
    body[flag] = Boolean(node && node.checked);
  }
  if (!String(body.description || "").trim()) {
    toast("A description is required.", "error");
    return;
  }
  button.disabled = true;
  button.textContent = "Screening…";
  try {
    const payload = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderVerdict(payload);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Screen this advert";
  }
}


/* ------------------------------------------------------------- start over -- */
/* Deletion, so the dialog shows exactly what will go before asking, and the
   dataset is a separate tick because it is the one thing that cannot be
   regenerated from what remains. */
function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / 1048576).toFixed(1)} MB`;
}

async function openReset() {
  const dialog = $("reset-dialog");
  const list = clear($("reset-targets"));
  const warning = clear($("reset-warning"));
  const confirm = $("reset-confirm");
  const datasetBox = $("reset-dataset");

  datasetBox.checked = false;
  confirm.disabled = true;
  confirm.textContent = "Checking…";
  list.append(el("div", { class: "skeleton-bar", style: "width:100%" }));
  dialog.showModal();

  let payload;
  try {
    payload = await api("/api/reset/preview");
  } catch (error) {
    clear(list).append(
      el("div", { class: "notice", "data-tone": "error", text: error.message })
    );
    confirm.textContent = "Delete and start over";
    return;
  }

  if (payload.active_job) {
    clear(list).append(
      el("div", {
        class: "notice",
        "data-tone": "warn",
        text:
          `'${payload.active_job}' is still running. Wait for it to finish ` +
          "or cancel it first — resetting now would delete files it is " +
          "part-way through writing.",
      })
    );
    confirm.textContent = "Delete and start over";
    return;
  }

  const generated = payload.targets.filter((t) => !t.guarded);
  const dataset = payload.targets.find((t) => t.guarded);
  const total = generated.reduce((sum, t) => sum + t.n_files, 0);

  const ul = el("ul", { class: "reset-list" });
  for (const target of generated) {
    ul.append(
      el("li", { "data-empty": String(target.n_files === 0) }, [
        el("div", { class: "what" }, [
          target.label,
          el("span", { text: target.description }),
        ]),
        el("div", {
          class: "count",
          text: target.n_files
            ? `${target.n_files} files · ${formatBytes(target.bytes)}`
            : "nothing",
        }),
      ])
    );
  }
  clear(list).append(ul);

  // The dataset row only makes sense when there is a dataset to remove.
  const row = $("reset-dataset-row");
  if (dataset && dataset.n_files) {
    row.style.display = "";
    row.querySelector(".help").textContent =
      `${dataset.examples[0] || "the CSV"} (${formatBytes(dataset.bytes)}). ` +
      "EMSCAD is not redistributable and cannot be re-downloaded without a " +
      "Kaggle session. Leave this unticked unless you have the file to hand.";
  } else {
    row.style.display = "none";
  }

  const describe = () => {
    const withDataset = datasetBox.checked && dataset ? dataset.n_files : 0;
    const count = total + withDataset;
    confirm.disabled = count === 0;
    confirm.textContent = count
      ? `Delete ${count} file${count === 1 ? "" : "s"}`
      : "Nothing to delete";
    clear(warning);
    if (datasetBox.checked) {
      warning.append(
        el("div", {
          class: "notice",
          "data-tone": "error",
          text:
            "The dataset will be deleted too. You will need the CSV from " +
            "Kaggle again before anything else can run.",
        })
      );
    }
  };
  datasetBox.onchange = describe;
  describe();
}

async function confirmReset() {
  const dialog = $("reset-dialog");
  const confirm = $("reset-confirm");
  const includeDataset = $("reset-dataset").checked;
  confirm.disabled = true;
  confirm.textContent = "Deleting…";
  try {
    const result = await api("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ include_dataset: includeDataset }),
    });
    dialog.close();
    // Everything the page was showing is gone; start from a clean slate.
    state.panels = {};
    state.lastActive = null;
    chartToken += 1;
    await refresh();
    show("import");
    toast(result.message, "success");
  } catch (error) {
    confirm.disabled = false;
    confirm.textContent = "Delete and start over";
    toast(error.message, "error");
  }
}

/* ----------------------------------------------------------------- boot -- */
async function refresh() {
  let server;
  try {
    server = await api("/api/state");
  } catch (error) {
    toast(`Could not load state: ${error.message}`, "error");
    return;
  }
  state.server = server;
  renderMasthead(server);
  renderRail(server);
  renderJobbar(server);
  renderImport(server);
  renderExplore(server);
  renderClean(server);
  renderTrain(server);
  renderPredict(server);
  renderReport(server);
  ensurePolling(server);
  if (server.active_job) state.lastActive = server.active_job;
  loadPanel(state.stage);
}

function boot() {
  $("clean-form").addEventListener("submit", onCleanSubmit);
  $("train-form").addEventListener("submit", onTrainSubmit);
  $("train-form").addEventListener("change", () => {
    updateStrategyNote();
    updateEstimate();
  });

  // The slider repaints instantly; the server preview waits for the drag to
  // settle so a sweep does not fire thirty requests.
  const slider = $("split-slider");
  slider.addEventListener("input", () =>
    paintProportion(Number(slider.value), null, null)
  );
  let settle;
  const schedulePreview = () => {
    clearTimeout(settle);
    settle = setTimeout(previewSplit, 180);
  };
  slider.addEventListener("change", schedulePreview);
  $("split-grouped").addEventListener("change", previewSplit);
  $("split-seed").addEventListener("change", schedulePreview);
  $("chart-picker").addEventListener("change", refreshCharts);
  $("reset-open").addEventListener("click", openReset);
  $("reset-confirm").addEventListener("click", confirmReset);
  $("reset-cancel").addEventListener("click", () => $("reset-dialog").close());

  // Show the shell straight away. Waiting for the first request before
  // painting anything is what made the page look broken on load; the rail
  // and a skeleton tell the user where they are while state arrives.
  paintProportion(Number(slider.value), null, null);
  showSkeleton();

  refresh().then(() => {
    if (!state.server) return;
    state.booted = true;
    // Land on the furthest stage the pipeline can actually reach, so a
    // returning user resumes where they left off.
    const reachable = [...STAGES]
      .reverse()
      .find((stage) => stageStatus(stage, state.server).tone === "done");
    show(reachable ? reachable.id : "import");
  });
}

/* Placeholder rail and panel, replaced by the first render. */
function showSkeleton() {
  const list = clear($("rail"));
  STAGES.forEach((stage, index) => {
    list.append(
      el("li", {}, el("button", { type: "button", disabled: true }, [
        el("span", { class: "num", text: String(index + 1) }),
        el("span", { class: "name", text: stage.name }),
        el("span", { class: "state", text: "…" }),
      ]))
    );
  });
  $("import-body").append(skeletonCard("Loading the project…"));
}

document.addEventListener("DOMContentLoaded", boot);
