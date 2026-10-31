// Grokking Visualiser front end. Consumes only site/data/ (see the Data
// Contract in the README); all paths are relative so it works under a
// GitHub Pages subpath and from `python -m http.server`.
"use strict";

const state = {
  manifest: null,
  run: null,      // the loaded run JSON
  index: 0,       // slider position = index into run.steps
};

const els = {
  slider: document.getElementById("step-slider"),
  readout: document.getElementById("readout"),
  error: document.getElementById("error"),
};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmtInt = (n) => n.toLocaleString("en-US");
const fmtPct = (x) => `${Math.round(x * 100)}%`;

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
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
  const ctx = document.getElementById("curve");
  curveChart = new Chart(ctx, {
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
        legend: { labels: { usePointStyle: false, boxWidth: 28, boxHeight: 2 } },
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
}

function initSlider() {
  els.slider.addEventListener("input", () => setIndex(Number(els.slider.value)));
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
    const entry = state.manifest.runs.find((r) => r.id === "baseline") || state.manifest.runs[0];
    state.run = await fetchJSON(`data/${entry.file}`);
    buildCurveChart();
    showRunOnChart(state.run);
    els.slider.max = String(state.run.steps.length - 1);
    els.slider.disabled = false;
    initSlider();
    // Start at the end of training so the full story is visible.
    setIndex(state.run.steps.length - 1);
  } catch (err) {
    showError(err);
  }
}

main();
