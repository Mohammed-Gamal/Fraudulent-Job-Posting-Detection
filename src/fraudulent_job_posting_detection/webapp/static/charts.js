/* ==========================================================================
   A small SVG chart kit.

   Hand-built rather than a charting library: the app is served from
   localhost and may have no network, so a CDN import would be a single point
   of failure, and vendoring a 3 MB bundle to draw five charts is a poor
   trade. These are a few hundred lines of SVG with one shared tooltip.

   Conventions, applied to every chart here:

   - Two series only ever mean legitimate and fraudulent, and they always
     take the same two colours. Colour carries identity, never rank, so a
     chart never repaints when its contents are re-sorted.
   - Those two colours are checked for colour-blind separation rather than
     chosen by eye (protanopia ΔE 15.9, normal-vision ΔE 29.7 against a white
     surface), and they are the same values the matplotlib figures use, so a
     generated PNG and a live chart cannot disagree.
   - Identity is never colour alone: every two-series chart carries a legend,
     and the tooltip names the series in text.
   - Values, labels and ticks wear the page's ink tokens, never the series
     colour.
   - Grid lines and axes are recessive; the marks are the only saturated
     thing on the card.
   ========================================================================== */

"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

const CHART_COLOURS = {
  legitimate: "#2f6fb5",
  fraudulent: "#e45756",
  single: "#4a6fa0",
  grid: "#d6dbe4",
  axis: "#8a95a8",
  ink: "#16233a",
  inkSoft: "#5a6880",
};

function svgEl(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    node.setAttribute(key, String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function text(x, y, value, opts = {}) {
  return svgEl(
    "text",
    {
      x,
      y,
      "text-anchor": opts.anchor || "start",
      "dominant-baseline": opts.baseline || "auto",
      fill: opts.fill || CHART_COLOURS.inkSoft,
      "font-size": opts.size || 11,
      "font-weight": opts.weight || 400,
      class: opts.tabular === false ? null : "tnum",
    },
    value
  );
}

/* ------------------------------------------------------------- tooltip -- */
let tipNode = null;

function tooltip() {
  if (!tipNode) {
    tipNode = document.createElement("div");
    tipNode.className = "charttip";
    tipNode.setAttribute("role", "status");
    document.body.append(tipNode);
  }
  return tipNode;
}

function showTip(event, lines) {
  const tip = tooltip();
  tip.replaceChildren(
    ...lines.map((line) => {
      const row = document.createElement("div");
      if (typeof line === "string") {
        row.textContent = line;
        row.className = "charttip-head";
      } else {
        const swatch = document.createElement("i");
        swatch.style.background = line.colour || "transparent";
        if (!line.colour) swatch.style.display = "none";
        const label = document.createElement("span");
        label.textContent = line.label;
        const value = document.createElement("b");
        value.textContent = line.value;
        row.append(swatch, label, value);
      }
      return row;
    })
  );
  tip.dataset.visible = "true";
  moveTip(event);
}

function moveTip(event) {
  const tip = tooltip();
  const pad = 14;
  const rect = tip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}

function hideTip() {
  if (tipNode) tipNode.dataset.visible = "false";
}

/* Attach hover to a mark. Keyboard focus shows the same tooltip, so the
   information is not mouse-only. */
function hoverable(node, lines) {
  node.style.cursor = "default";
  node.addEventListener("mouseenter", (event) => showTip(event, lines));
  node.addEventListener("mousemove", moveTip);
  node.addEventListener("mouseleave", hideTip);
  node.setAttribute("tabindex", "0");
  node.addEventListener("focus", (event) => {
    const box = node.getBoundingClientRect();
    showTip({ clientX: box.left + box.width / 2, clientY: box.top }, lines);
  });
  node.addEventListener("blur", hideTip);
  return node;
}

function legend(items) {
  const wrap = document.createElement("div");
  wrap.className = "chart-legend";
  for (const item of items) {
    const entry = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = item.colour;
    entry.append(swatch, document.createTextNode(item.label));
    wrap.append(entry);
  }
  return wrap;
}

function niceCeiling(value) {
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const scaled = value / magnitude;
  const step = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return step * magnitude;
}

function ticks(max, count = 4) {
  const out = [];
  for (let i = 0; i <= count; i += 1) out.push((max / count) * i);
  return out;
}

function truncate(value, limit) {
  const s = String(value);
  return s.length <= limit ? s : `${s.slice(0, limit - 1)}…`;
}

function shell(width, height) {
  return svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    height: null,
    preserveAspectRatio: "xMinYMin meet",
    class: "chart-svg",
    role: "img",
    xmlns: SVG_NS,
    // Set as an attribute, not from the stylesheet: a downloaded SVG and a
    // canvas rasterisation both lose the page's CSS, and without this the
    // exported chart falls back to a serif default.
    "font-family":
      '"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif',
    "data-width": width,
    "data-height": height,
  });
}

/* ------------------------------------------------------------ download -- */
function serialise(svg) {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", SVG_NS);
  const width = clone.getAttribute("data-width");
  const height = clone.getAttribute("data-height");
  clone.setAttribute("width", width);
  clone.setAttribute("height", height);
  // Hover bands are invisible on screen and meaningless in a file.
  clone.querySelectorAll('rect[fill="transparent"]').forEach((n) => n.remove());
  clone.querySelectorAll("[tabindex]").forEach((n) => n.removeAttribute("tabindex"));
  const background = document.createElementNS(SVG_NS, "rect");
  background.setAttribute("x", "0");
  background.setAttribute("y", "0");
  background.setAttribute("width", width);
  background.setAttribute("height", height);
  background.setAttribute("fill", "#ffffff");
  clone.insertBefore(background, clone.firstChild);
  return new XMLSerializer().serializeToString(clone);
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  // Revoke on the next tick; revoking immediately cancels the download in
  // some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function downloadSvg(svg, name) {
  triggerDownload(
    new Blob([serialise(svg)], { type: "image/svg+xml;charset=utf-8" }),
    `${name}.svg`
  );
}

function downloadPng(svg, name, scale = 2) {
  const markup = serialise(svg);
  const width = Number(svg.getAttribute("data-width"));
  const height = Number(svg.getAttribute("data-height"));
  const image = new Image();
  const url = URL.createObjectURL(
    new Blob([markup], { type: "image/svg+xml;charset=utf-8" })
  );
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const context = canvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    canvas.toBlob((blob) => {
      if (blob) triggerDownload(blob, `${name}.png`);
    }, "image/png");
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    // Falling back to the SVG is better than failing silently: the user
    // still gets the chart, in the format that always works.
    downloadSvg(svg, name);
  };
  image.src = url;
}

/* ===================================================== histogram ======== */
function histogram(spec) {
  const W = 760;
  const H = 320;
  const M = { top: 16, right: 16, bottom: 46, left: 56 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const svg = shell(W, H);
  const edges = spec.edges;
  const nBins = edges.length - 1;
  const maxY = niceCeiling(
    Math.max(...spec.series.flatMap((s) => s.density)) || 0.01
  );
  const xOf = (i) => M.left + (i / nBins) * plotW;
  const yOf = (v) => M.top + plotH - (v / maxY) * plotH;

  for (const t of ticks(maxY)) {
    const y = yOf(t);
    svg.append(
      svgEl("line", {
        x1: M.left, x2: M.left + plotW, y1: y, y2: y,
        stroke: CHART_COLOURS.grid, "stroke-width": 1,
      }),
      text(M.left - 8, y, `${(t * 100).toFixed(0)}%`, {
        anchor: "end", baseline: "middle", size: 10,
      })
    );
  }

  const xStep = Math.max(1, Math.round(nBins / 8));
  for (let i = 0; i <= nBins; i += xStep) {
    svg.append(
      text(xOf(i), M.top + plotH + 16, Math.round(edges[i]), {
        anchor: "middle", size: 10,
      })
    );
  }

  const barW = plotW / nBins;
  // Bars ignore the pointer: the two distributions overlap, so whichever
  // series happened to be drawn second would swallow every hover and the
  // other one would be unreachable. A band per bin handles hover instead and
  // reports both series at once, which is the comparison the chart is for.
  for (const series of spec.series) {
    const colour = CHART_COLOURS[series.label] || CHART_COLOURS.single;
    for (let i = 0; i < nBins; i += 1) {
      const value = series.density[i];
      if (!value) continue;
      const y = yOf(value);
      svg.append(
        svgEl("rect", {
          x: xOf(i) + 1,
          y,
          width: Math.max(barW - 2, 1),
          height: M.top + plotH - y,
          fill: colour,
          "fill-opacity": 0.55,
          rx: 2,
          "pointer-events": "none",
        })
      );
    }
  }

  // Median markers, staggered so the two labels cannot collide when the
  // medians are close.
  spec.series.forEach((series, index) => {
    const colour = CHART_COLOURS[series.label] || CHART_COLOURS.single;
    const position = series.median / (edges[edges.length - 1] || 1);
    if (position < 0 || position > 1) return;
    const x = M.left + position * plotW;
    const labelY = M.top + 11 + index * 14;
    const toLeft = position > 0.55;
    svg.append(
      svgEl("line", {
        x1: x, x2: x, y1: M.top, y2: M.top + plotH,
        stroke: colour, "stroke-width": 2, "stroke-dasharray": "4 3",
        "pointer-events": "none",
      }),
      text(x + (toLeft ? -5 : 5), labelY, `median ${Math.round(series.median)}`, {
        size: 10, fill: colour, weight: 600, anchor: toLeft ? "end" : "start",
      })
    );
  });

  // Invisible hover bands, one per bin, on top of everything.
  for (let i = 0; i < nBins; i += 1) {
    const band = svgEl("rect", {
      x: xOf(i), y: M.top, width: barW, height: plotH,
      fill: "transparent",
    });
    const lines = [`${Math.round(edges[i])}–${Math.round(edges[i + 1])} words`];
    for (const series of spec.series) {
      lines.push({
        label: series.label,
        colour: CHART_COLOURS[series.label] || CHART_COLOURS.single,
        value:
          `${(series.density[i] * 100).toFixed(1)}% · ` +
          `${series.counts[i].toLocaleString()}`,
      });
    }
    band.addEventListener("mouseenter", () => band.setAttribute("fill", "#16233a0f"));
    band.addEventListener("mouseleave", () => band.setAttribute("fill", "transparent"));
    hoverable(band, lines);
    svg.append(band);
  }

  const yTitle = text(16, M.top + plotH / 2, spec.y_label, {
    anchor: "middle", size: 11, fill: CHART_COLOURS.inkSoft,
  });
  yTitle.setAttribute("transform", `rotate(-90 16 ${M.top + plotH / 2})`);
  svg.append(
    svgEl("line", {
      x1: M.left, x2: M.left + plotW, y1: M.top + plotH, y2: M.top + plotH,
      stroke: CHART_COLOURS.axis, "stroke-width": 1,
    }),
    text(M.left + plotW / 2, H - 6, spec.x_label, {
      anchor: "middle", size: 11, fill: CHART_COLOURS.inkSoft,
    }),
    yTitle
  );

  return {
    svg,
    legend: legend(
      spec.series.map((s) => ({
        label: `${s.label} (n=${s.n.toLocaleString()}, median ${Math.round(s.median)})`,
        colour: CHART_COLOURS[s.label] || CHART_COLOURS.single,
      }))
    ),
  };
}

/* ================================================== grouped bar ========= */
function groupedBar(spec) {
  const W = 760;
  const H = 300;
  const M = { top: 20, right: 16, bottom: 58, left: 52 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const svg = shell(W, H);

  const maxY = niceCeiling(
    Math.max(...spec.groups.flatMap((g) => g.bars.map((b) => b.rate)))
  );
  const yOf = (v) => M.top + plotH - (v / maxY) * plotH;

  for (const t of ticks(maxY)) {
    const y = yOf(t);
    svg.append(
      svgEl("line", {
        x1: M.left, x2: M.left + plotW, y1: y, y2: y,
        stroke: CHART_COLOURS.grid, "stroke-width": 1,
      }),
      text(M.left - 8, y, `${t.toFixed(0)}%`, {
        anchor: "end", baseline: "middle", size: 10,
      })
    );
  }

  const groupW = plotW / spec.groups.length;
  spec.groups.forEach((group, gi) => {
    const inner = groupW * 0.62;
    const barW = inner / group.bars.length;
    const left = M.left + gi * groupW + (groupW - inner) / 2;
    group.bars.forEach((bar, bi) => {
      const y = yOf(bar.rate);
      // One hue: height already encodes the rate, and colouring the two bars
      // differently would read as the two classes, which they are not.
      const rect = svgEl("rect", {
        x: left + bi * barW + 1,
        y,
        width: barW - 2,
        height: Math.max(M.top + plotH - y, 1),
        fill: CHART_COLOURS.single,
        rx: 4,
      });
      hoverable(rect, [
        `${group.label} = ${bar.label}`,
        { label: "fraud rate", colour: CHART_COLOURS.single, value: `${bar.rate}%` },
        { label: "postings", value: bar.n.toLocaleString() },
        { label: "fraudulent", value: bar.n_fraud.toLocaleString() },
      ]);
      svg.append(
        rect,
        text(left + bi * barW + barW / 2, y - 5, `${bar.rate}%`, {
          anchor: "middle", size: 10, fill: CHART_COLOURS.ink, weight: 600,
        }),
        text(left + bi * barW + barW / 2, M.top + plotH + 14, bar.label, {
          anchor: "middle", size: 10,
        })
      );
    });
    svg.append(
      text(M.left + gi * groupW + groupW / 2, M.top + plotH + 32, group.label, {
        anchor: "middle", size: 11, fill: CHART_COLOURS.ink, weight: 600,
      })
    );
  });

  const baselineY = yOf(spec.baseline);
  svg.append(
    svgEl("line", {
      x1: M.left, x2: M.left + plotW, y1: baselineY, y2: baselineY,
      stroke: CHART_COLOURS.ink, "stroke-width": 1.5, "stroke-dasharray": "5 4",
    }),
    text(M.left + plotW - 2, baselineY - 5, spec.baseline_label, {
      anchor: "end", size: 10, fill: CHART_COLOURS.ink,
    }),
    svgEl("line", {
      x1: M.left, x2: M.left + plotW, y1: M.top + plotH, y2: M.top + plotH,
      stroke: CHART_COLOURS.axis, "stroke-width": 1,
    })
  );
  return { svg, legend: null };
}

/* ================================================ diverging bar ========= */
function divergingBar(spec) {
  /* Two ranked lists facing each other, paired row by row. Sorting both
     sides into one column instead leaves the whole upper-left and lower-right
     of the plot empty, because every fraudulent term outranks every
     legitimate one by construction. Pairing them fills the space and puts the
     two lists at the same scale, which is the comparison worth making. */
  const rowCount = Math.max(spec.fraudulent.length, spec.legitimate.length);
  const rowH = 22;
  const W = 760;
  const M = { top: 28, right: 12, bottom: 34, left: 12 };
  const labelGap = 112;
  const H = M.top + rowCount * rowH + M.bottom;
  const svg = shell(W, H);

  const span =
    Math.max(
      ...spec.fraudulent.map((r) => Math.abs(r.log_odds)),
      ...spec.legitimate.map((r) => Math.abs(r.log_odds)),
      1
    ) || 1;
  const centre = W / 2;
  const half = (W - M.left - M.right) / 2 - labelGap;

  svg.append(
    text(centre - half - 6, M.top - 14, "indicative of legitimate", {
      anchor: "end", size: 10, fill: CHART_COLOURS.legitimate, weight: 600,
    }),
    text(centre + half + 6, M.top - 14, "indicative of fraudulent", {
      anchor: "start", size: 10, fill: CHART_COLOURS.fraudulent, weight: 600,
    })
  );

  const draw = (row, index, side) => {
    if (!row) return;
    const y = M.top + index * rowH;
    const colour = CHART_COLOURS[side];
    const toRight = side === "fraudulent";
    const width = Math.max((Math.abs(row.log_odds) / span) * half, 2);
    const rect = svgEl("rect", {
      x: toRight ? centre + 1 : centre - 1 - width,
      y: y + 4,
      width,
      height: rowH - 9,
      fill: colour,
      rx: 3,
    });
    hoverable(rect, [
      row.term,
      { label: "log-odds", colour, value: row.log_odds.toFixed(3) },
      {
        label: "of fraudulent",
        value: `${row.pct_fraudulent}% (${row.docfreq_fraudulent})`,
      },
      {
        label: "of legitimate",
        value: `${row.pct_legitimate}% (${row.docfreq_legitimate})`,
      },
    ]);
    svg.append(
      rect,
      // Label outside the plot area, so it never sits on a fill.
      text(
        toRight ? centre + half + 8 : centre - half - 8,
        y + rowH / 2,
        truncate(row.term, 16),
        {
          anchor: toRight ? "start" : "end",
          baseline: "middle",
          size: 11,
          fill: CHART_COLOURS.ink,
        }
      )
    );
  };

  for (let i = 0; i < rowCount; i += 1) {
    draw(spec.legitimate[i], i, "legitimate");
    draw(spec.fraudulent[i], i, "fraudulent");
  }

  svg.append(
    svgEl("line", {
      x1: centre, x2: centre, y1: M.top - 4, y2: M.top + rowCount * rowH + 4,
      stroke: CHART_COLOURS.ink, "stroke-width": 1.5,
    }),
    text(centre, H - 8, spec.x_label, {
      anchor: "middle", size: 11, fill: CHART_COLOURS.inkSoft,
    })
  );
  return {
    svg,
    legend: legend([
      { label: "pushes legitimate", colour: CHART_COLOURS.legitimate },
      { label: "pushes fraudulent", colour: CHART_COLOURS.fraudulent },
    ]),
  };
}

/* ==================================================== paired bar ======== */
function pairedBar(spec) {
  const rowH = 34;
  const W = 760;
  const M = { top: 10, right: 54, bottom: 34, left: 168 };
  const H = M.top + spec.rows.length * rowH + M.bottom;
  const plotW = W - M.left - M.right;
  const svg = shell(W, H);

  const maxX = 100;
  const xOf = (v) => M.left + (v / maxX) * plotW;

  for (const t of [0, 25, 50, 75, 100]) {
    svg.append(
      svgEl("line", {
        x1: xOf(t), x2: xOf(t), y1: M.top, y2: M.top + spec.rows.length * rowH,
        stroke: CHART_COLOURS.grid, "stroke-width": 1,
      }),
      text(xOf(t), H - 18, `${t}%`, { anchor: "middle", size: 10 })
    );
  }

  spec.rows.forEach((row, index) => {
    const y = M.top + index * rowH;
    svg.append(
      text(M.left - 10, y + rowH / 2, truncate(row.field, 22), {
        anchor: "end", baseline: "middle", size: 11, fill: CHART_COLOURS.ink,
      })
    );
    [
      ["legitimate", row.legitimate, 0],
      ["fraudulent", row.fraudulent, 1],
    ].forEach(([label, value, slot]) => {
      const colour = CHART_COLOURS[label];
      const barH = 11;
      const top = y + 4 + slot * (barH + 2);
      const rect = svgEl("rect", {
        x: M.left, y: top, width: Math.max(xOf(value) - M.left, 1),
        height: barH, fill: colour, rx: 3,
      });
      hoverable(rect, [
        row.field,
        { label, colour, value: `${value}% empty` },
        { label: "gap", value: `${row.difference > 0 ? "+" : ""}${row.difference} pp` },
      ]);
      svg.append(rect);
    });
    svg.append(
      text(W - M.right + 6, y + rowH / 2,
        `${row.difference > 0 ? "+" : ""}${row.difference}`, {
        anchor: "start", baseline: "middle", size: 10,
        fill: row.difference > 0 ? CHART_COLOURS.fraudulent : CHART_COLOURS.legitimate,
        weight: 600,
      })
    );
  });

  svg.append(
    text(M.left + plotW / 2, H - 4, spec.x_label, {
      anchor: "middle", size: 11, fill: CHART_COLOURS.inkSoft,
    })
  );
  return {
    svg,
    legend: legend([
      { label: "legitimate", colour: CHART_COLOURS.legitimate },
      { label: "fraudulent", colour: CHART_COLOURS.fraudulent },
    ]),
  };
}

/* ==================================================== ranked bar ======== */
function rankedBar(spec) {
  const rowH = 24;
  const W = 760;
  const M = { top: 10, right: 68, bottom: 36, left: 190 };
  const H = M.top + spec.rows.length * rowH + M.bottom;
  const plotW = W - M.left - M.right;
  const svg = shell(W, H);

  const maxX = niceCeiling(Math.max(...spec.rows.map((r) => r.rate), spec.baseline));
  const xOf = (v) => M.left + (v / maxX) * plotW;

  for (const t of ticks(maxX)) {
    svg.append(
      svgEl("line", {
        x1: xOf(t), x2: xOf(t), y1: M.top, y2: M.top + spec.rows.length * rowH,
        stroke: CHART_COLOURS.grid, "stroke-width": 1,
      }),
      text(xOf(t), H - 18, `${t.toFixed(0)}%`, { anchor: "middle", size: 10 })
    );
  }

  spec.rows.forEach((row, index) => {
    const y = M.top + index * rowH;
    // Sequential shading by magnitude: one hue, light to dark, so the ramp
    // encodes the rate rather than decorating it.
    const intensity = 0.35 + 0.65 * (row.rate / maxX);
    const rect = svgEl("rect", {
      x: M.left, y: y + 4, width: Math.max(xOf(row.rate) - M.left, 2),
      height: rowH - 9, fill: CHART_COLOURS.fraudulent,
      "fill-opacity": intensity.toFixed(2), rx: 3,
    });
    hoverable(rect, [
      row.value,
      { label: "fraud rate", colour: CHART_COLOURS.fraudulent, value: `${row.rate}%` },
      { label: "postings", value: row.n.toLocaleString() },
      { label: "fraudulent", value: row.n_fraud.toLocaleString() },
    ]);
    svg.append(
      text(M.left - 10, y + rowH / 2, truncate(row.value, 26), {
        anchor: "end", baseline: "middle", size: 11, fill: CHART_COLOURS.ink,
      }),
      rect,
      text(xOf(row.rate) + 6, y + rowH / 2, `${row.rate}%`, {
        anchor: "start", baseline: "middle", size: 10, fill: CHART_COLOURS.ink,
      })
    );
  });

  const bx = xOf(spec.baseline);
  svg.append(
    svgEl("line", {
      x1: bx, x2: bx, y1: M.top, y2: M.top + spec.rows.length * rowH,
      stroke: CHART_COLOURS.ink, "stroke-width": 1.5, "stroke-dasharray": "5 4",
    }),
    text(M.left + plotW / 2, H - 4,
      `${spec.x_label} — dashed line is the ${spec.baseline}% dataset baseline`, {
      anchor: "middle", size: 11, fill: CHART_COLOURS.inkSoft,
    })
  );
  return { svg, legend: null };
}

/* ======================================================= dispatch ======= */
const RENDERERS = {
  histogram,
  grouped_bar: groupedBar,
  diverging_bar: divergingBar,
  paired_bar: pairedBar,
  ranked_bar: rankedBar,
};

/** Render one chart spec into a detached element, or null if unsupported. */
function renderChart(spec) {
  const renderer = RENDERERS[spec.kind];
  if (!renderer) return null;
  try {
    return renderer(spec);
  } catch (error) {
    console.error("chart render failed", spec.kind, error);
    return null;
  }
}

window.FJPDCharts = {
  renderChart,
  COLOURS: CHART_COLOURS,
  hideTip,
  downloadPng,
  downloadSvg,
};
