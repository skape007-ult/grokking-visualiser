// Grokking Visualiser front end. Consumes only site/data/ (see the Data
// Contract in the README); all paths are relative so it works under a
// GitHub Pages subpath and from `python -m http.server`.
"use strict";

const state = {
  manifest: null,
  runId: null,
  run: null,          // the loaded run JSON for runId
  index: 0,           // slider position = index into run.steps
  embLimits: null,    // fixed embedding axis limits for the current run
  hover: null,        // number under the pointer (tooltip)
  pinned: null,       // number selected by click/tap, highlighted across frames
};

const runCache = new Map();  // run id -> Promise<run JSON>, lazily filled

const els = {
  toggle: document.getElementById("run-toggle"),
  slider: document.getElementById("step-slider"),
  readout: document.getElementById("readout"),
  error: document.getElementById("error"),
  embBox: document.getElementById("emb-box"),
  embCanvas: document.getElementById("embedding"),
  tooltip: document.getElementById("emb-tooltip"),
  frameLabel: document.getElementById("frame-label"),
  legendBar: document.getElementById("legend-bar"),
  legendMax: document.getElementById("legend-max"),
};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmtInt = (n) => n.toLocaleString("en-US");
const fmtPct = (x) => `${Math.round(x * 100)}%`;

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

function loadRun(id) {
  if (!runCache.has(id)) {
    const entry = state.manifest.runs.find((r) => r.id === id);
    const p = fetchJSON(`data/${entry.file}`);
    p.catch(() => runCache.delete(id));  // allow a retry after a failed load
    runCache.set(id, p);
  }
  return runCache.get(id);
}

// ---- Colour ramp (viridis, interpolated; no library) ----------------------

const VIRIDIS = [
  [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141], [42, 120, 142],
  [33, 145, 140], [34, 168, 132], [68, 191, 112], [122, 209, 81], [189, 223, 38],
  [253, 231, 37],
];

function viridis(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const [a, b] = [VIRIDIS[i], VIRIDIS[i + 1]];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

// ---- Accuracy chart ------------------------------------------------------

// Draws a vertical line at the slider's current step.
const markerPlugin = {
  id: "stepMarker",
  afterDatasetsDraw(chart) {
    if (!state.run) return;
    const x = chart.scales.x.getPixelForValue(state.run.steps[state.index]);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = css("--marker");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.restore();
  },
};

function logTick(value) {
  const exp = Math.log10(value);
  if (Math.abs(exp - Math.round(exp)) > 1e-9) return "";  // label powers of ten only
  return value >= 1000 ? `${value / 1000}k` : String(value);
}

let curveChart = null;

function buildCurveChart() {
  curveChart = new Chart(document.getElementById("curve"), {
    type: "line",
    data: { datasets: [
      { label: "Train accuracy", data: [], borderColor: css("--train"), backgroundColor: css("--train"),
        borderWidth: 2.5, pointRadius: 0, tension: 0 },
      { label: "Test accuracy (unseen examples)", data: [], borderColor: css("--test"), backgroundColor: css("--test"),
        borderWidth: 2.5, borderDash: [7, 5], pointRadius: 0, tension: 0 },
    ] },
    options: {
      animation: false,
      maintainAspectRatio: false,
      parsing: false,
      normalized: true,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      scales: {
        x: { type: "logarithmic", min: 1,
             title: { display: true, text: "Training step (log scale)" },
             ticks: { callback: logTick, autoSkip: false, maxRotation: 0 } },
        y: { min: 0, max: 1,
             title: { display: true, text: "Accuracy" },
             ticks: { stepSize: 0.25, callback: (v) => fmtPct(v) } },
      },
      plugins: {
        legend: { labels: { boxWidth: 28, boxHeight: 2 } },
        tooltip: {
          callbacks: {
            title: (items) => `Step ${fmtInt(items[0].parsed.x)}`,
            label: (item) => `${item.dataset.label}: ${fmtPct(item.parsed.y)}`,
          },
        },
      },
    },
    plugins: [markerPlugin],
  });
}

function showRunOnChart(run) {
  const pts = (ys) => run.steps.map((s, i) => ({ x: s, y: ys[i] }));
  curveChart.data.datasets[0].data = pts(run.train_acc);
  curveChart.data.datasets[1].data = pts(run.test_acc);
  curveChart.options.scales.x.max = run.steps[run.steps.length - 1];
  curveChart.update("none");
}

// ---- Embedding scatter (hand-drawn canvas, crisp on high-DPI) -------------

const EMB_PAD = 14;      // CSS px inside the canvas edge
const POINT_R = 4.5;     // CSS px

// Axis limits fixed across every frame of the run (so the plot doesn't
// "pulse"), with equal scale on x and y and a small symmetric margin.
function embeddingLimits(run) {
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const f of run.embedding_frames) {
    for (const [x, y] of f.coords) {
      x0 = Math.min(x0, x); x1 = Math.max(x1, x);
      y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
  }
  const half = (Math.max(x1 - x0, y1 - y0) / 2) * 1.06 || 1;
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, half };
}

// The frame shown is the latest one at or before the slider's step
// (frames are a subset of steps and include step 1, so one always exists).
function frameIndexFor(run, step) {
  let k = 0;
  run.embedding_frames.forEach((f, i) => { if (f.step <= step) k = i; });
  return k;
}

function embGeometry() {
  const size = els.embBox.clientWidth;
  const { cx, cy, half } = state.embLimits;
  const scale = (size / 2 - EMB_PAD) / half;
  return {
    size,
    toPx: ([x, y]) => [size / 2 + (x - cx) * scale, size / 2 - (y - cy) * scale],
  };
}

function currentFrame() {
  const run = state.run;
  return run.embedding_frames[frameIndexFor(run, run.steps[state.index])];
}

function drawEmbedding() {
  if (!state.run) return;
  const canvas = els.embCanvas;
  const dpr = window.devicePixelRatio || 1;
  const { size, toPx } = embGeometry();
  if (canvas.width !== Math.round(size * dpr)) {
    canvas.width = canvas.height = Math.round(size * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  // Faint crosshair through the run's centre, for orientation.
  ctx.strokeStyle = "#ecebe6";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(size / 2, 0); ctx.lineTo(size / 2, size);
  ctx.moveTo(0, size / 2); ctx.lineTo(size, size / 2);
  ctx.stroke();

  const frame = currentFrame();
  const p = frame.coords.length;
  ctx.lineWidth = 0.75;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
  frame.coords.forEach((xy, n) => {
    const [px, py] = toPx(xy);
    ctx.beginPath();
    ctx.arc(px, py, POINT_R, 0, Math.PI * 2);
    ctx.fillStyle = viridis(n / (p - 1));
    ctx.fill();
    ctx.stroke();
  });

  for (const n of [state.pinned, state.hover]) {
    if (n === null) continue;
    const [px, py] = toPx(frame.coords[n]);
    ctx.beginPath();
    ctx.arc(px, py, POINT_R + 4, 0, Math.PI * 2);
    ctx.strokeStyle = css("--marker");
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  els.frameLabel.textContent = `· step ${fmtInt(frame.step)}`;
  updateTooltip();
}

function updateTooltip() {
  const n = state.hover ?? state.pinned;
  if (n === null || !state.run) { els.tooltip.hidden = true; return; }
  const [px, py] = embGeometry().toPx(currentFrame().coords[n]);
  els.tooltip.textContent = String(n);
  els.tooltip.style.left = `${px}px`;
  els.tooltip.style.top = `${py}px`;
  els.tooltip.hidden = false;
}

function nearestPoint(evt) {
  const rect = els.embCanvas.getBoundingClientRect();
  const mx = evt.clientX - rect.left, my = evt.clientY - rect.top;
  const { toPx } = embGeometry();
  let best = null, bestD = (evt.pointerType === "touch" ? 22 : 12) ** 2;
  currentFrame().coords.forEach((xy, n) => {
    const [px, py] = toPx(xy);
    const d = (px - mx) ** 2 + (py - my) ** 2;
    if (d < bestD) { bestD = d; best = n; }
  });
  return best;
}

function initEmbeddingInteraction() {
  const c = els.embCanvas;
  c.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    const n = nearestPoint(e);
    if (n !== state.hover) { state.hover = n; drawEmbedding(); }
  });
  c.addEventListener("pointerleave", () => { state.hover = null; drawEmbedding(); });
  // Click / tap pins a number so you can follow it through training.
  c.addEventListener("pointerdown", (e) => {
    const n = nearestPoint(e);
    state.pinned = n === state.pinned ? null : n;
    drawEmbedding();
  });
  new ResizeObserver(() => drawEmbedding()).observe(els.embBox);

  const stops = Array.from({ length: 11 }, (_, i) => `${viridis(i / 10)} ${i * 10}%`);
  els.legendBar.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
  els.legendMax.textContent = String(state.manifest.p - 1);
}

// ---- Slider + readout ----------------------------------------------------

function setIndex(i) {
  const run = state.run;
  state.index = Math.max(0, Math.min(run.steps.length - 1, i));
  const step = run.steps[state.index];
  const text = `Step ${fmtInt(step)} — train ${fmtPct(run.train_acc[state.index])}, ` +
               `test ${fmtPct(run.test_acc[state.index])}`;
  els.readout.textContent = text;
  els.slider.value = String(state.index);
  els.slider.setAttribute("aria-valuetext", text);
  curveChart.draw();  // redraw marker only; no data change, no animation
  drawEmbedding();
}

// Index in `run` whose step is nearest (on the log scale) to `step`.
function nearestIndex(run, step) {
  let best = 0;
  run.steps.forEach((s, i) => {
    if (Math.abs(Math.log(s) - Math.log(step)) < Math.abs(Math.log(run.steps[best]) - Math.log(step))) best = i;
  });
  return best;
}

// ---- Run toggle ----------------------------------------------------------

function buildToggle() {
  for (const r of state.manifest.runs) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.run = r.id;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<span class="run-label"></span><span class="run-desc"></span>`;
    b.querySelector(".run-label").textContent = r.label;
    b.querySelector(".run-desc").textContent = r.description;
    b.addEventListener("click", () => {
      history.replaceState(null, "", `#run=${r.id}`);
      selectRun(r.id);
    });
    els.toggle.appendChild(b);
  }
}

function runFromHash() {
  const m = location.hash.match(/run=([\w-]+)/);
  const ids = state.manifest.runs.map((r) => r.id);
  return m && ids.includes(m[1]) ? m[1] : ids[0];
}

let selectToken = 0;

async function selectRun(id) {
  const token = ++selectToken;
  for (const b of els.toggle.children) b.setAttribute("aria-pressed", String(b.dataset.run === id));
  const prevStep = state.run ? state.run.steps[state.index] : null;
  const run = await loadRun(id);
  if (token !== selectToken) return;  // a newer click won the race

  state.runId = id;
  state.run = run;
  state.embLimits = embeddingLimits(run);
  showRunOnChart(run);
  els.slider.max = String(run.steps.length - 1);
  els.slider.disabled = false;
  // Keep the slider at the equivalent step; start at the end on first load.
  setIndex(prevStep === null ? run.steps.length - 1 : nearestIndex(run, prevStep));
  document.body.dataset.run = id;  // the run actually rendered (used by check_site.py)
}

// ---- Boot ----------------------------------------------------------------

function showError(err) {
  console.error(err);
  els.error.hidden = false;
  els.error.textContent =
    "Sorry, the training data couldn't be loaded. If you opened this file directly, " +
    "serve the site folder over HTTP instead (e.g. `python -m http.server`). " +
    `Details: ${err.message}`;
  els.readout.textContent = "";
}

async function main() {
  try {
    if (typeof Chart === "undefined") throw new Error("Chart.js failed to load from the CDN");
    state.manifest = await fetchJSON("data/manifest.json");
    buildCurveChart();
    buildToggle();
    initEmbeddingInteraction();
    els.slider.addEventListener("input", () => setIndex(Number(els.slider.value)));
    window.addEventListener("hashchange", () => selectRun(runFromHash()).catch(showError));
    await selectRun(runFromHash());
  } catch (err) {
    showError(err);
  }
}

main();
