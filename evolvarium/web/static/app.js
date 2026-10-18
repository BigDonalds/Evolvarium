/* evolvarium dashboard
   No build step and no dependencies. Nothing here knows a game: what to draw
   comes from the run's scene, and every number from the run's own records.

   Runs: the page shows one run at a time, picked from every run in the runs
   directory. The one training in the server's process is live; the rest are
   read back from disk, finished or training in another process. A dashboard
   opened with `serve` can also start runs, one at a time.

   Presence: the live run holds at a generation boundary while this page shows
   it, so a generation cannot be replaced before it has been looked at. That
   hold is a lease, refreshed by the heartbeat below. Only a page showing the
   live run sends them, so closing the tab or showing another run lets the
   trainer resume on its own rather than wait forever.

   Pinning: selecting a generation detaches the view from the live edge. New
   generations keep arriving and the scrubber range grows, but the displayed
   generation does not change until "Follow latest" is turned back on.

   Events: the live run announces each generation, record, hold and note on
   /api/events, each tagged with the run, and the page reacts to those of the
   run it shows. Status is still polled, slowly, for the clocks and rates that
   change all the time, and at the old rate whenever the stream is down. */

const SVG_NS = "http://www.w3.org/2000/svg";
const HEARTBEAT_MS = 3000;
const STATUS_MS = 1200;        // polling while the event stream is down
const STATUS_LIVE_MS = 5000;   // polling while events arrive on their own
const SAVED_MS = 5000;         // polling a run read from disk
const RUNS_MS = 10000;
const JOB_MS = 500;
const RECENT_SECONDS = 120;    // a run written to this recently may still be training
const COMPARE_AT_MOST = 8;
const TICK_MS = 90;

const state = {
  runs: [],              // every run, newest first
  liveRun: null,         // the run training in the server's process
  canStart: false,
  defaultGame: null,
  run: null,             // the run shown
  chosen: false,         // picked on the page, rather than the newest or live one
  wanted: null,          // run named in the address, looked for once
  status: null,
  live: false,           // event stream connected
  curve: null,           // chart series, extended as generations arrive
  chartScale: null,      // how the main chart maps generations to pixels
  generation: null,      // loaded GenerationRecord
  following: true,
  pinned: null,          // generation number when not following
  bounds: { first: 0, last: 0, available: [] },
  frame: 0,
  maxFrames: 0,
  playing: true,
  speed: 1,
  sort: "fitness",
  limit: 60,
  agents: [],
  shown: [],             // agents with a card, in grid order
  scene: null,           // what the game draws, from the generation
  paint: null,           // markup for one frame of that scene
  sheetAgent: null,      // agent open in the side sheet
  exportPick: "average",
  exportHint: "",        // the export panel's text before there is a report
  championReport: null,  // export preview of the champion
  job: null,             // a choice by average for the run shown
  gamesLoaded: false,
  settings: { game: null, rows: [], inputs: new Map() },
  compare: new Map(),    // run id -> colour slot
  compareTouched: false, // until then the list follows the run shown
  compared: null,        // the last comparison fetched
  tickAccumulator: 0,
  lastTick: 0,
};

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- transport */

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

/* `path` asked of the run shown, with `params` that have a value. */
function url(path, params = {}) {
  const query = new URLSearchParams();
  if (state.run) query.set("run", state.run);
  for (const [key, value] of Object.entries(params)) {
    if (value != null && value !== "") query.set(key, String(value));
  }
  const text = query.toString();
  return text ? `${path}?${text}` : path;
}

/* An answer about the run shown, or null when another run was shown before it came. */
async function runJSON(path, params) {
  const run = state.run;
  const data = await getJSON(url(path, params));
  return state.run === run ? data : null;
}

let toastTimer = null;
function toast(message, kind = "") {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast ${kind}`;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 3600);
}

/* ---------------------------------------------------------------- formatting */

const fixed = (v, d) =>
  Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

const fmt = {
  int: (n) => (n == null ? "–" : Math.round(n).toLocaleString()),
  dec: (n, d = 2) => (n == null ? "–" : Number(n).toFixed(d)),
  pct: (n) => (n == null ? "–" : `${(n * 100).toFixed(1)}%`),
  // A score as the game keeps it: whole when it is whole.
  num: (n) => {
    if (n == null || !Number.isFinite(Number(n))) return "–";
    const v = Number(n);
    if (Number.isInteger(v)) return v.toLocaleString();
    return fixed(v, Math.abs(v) >= 100 ? 1 : 2);
  },
  bytes: (n) => {
    if (n == null) return "–";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${(n / 1048576).toFixed(2)} MiB`;
  },
  duration: (s) => {
    if (s == null) return "–";
    const total = Math.round(s);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
    if (m) return `${m}m ${String(sec).padStart(2, "0")}s`;
    return `${sec}s`;
  },
  ago: (s) => {
    if (s == null) return "–";
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  },
  // "2026-10-15T09:39:28" as "2026-10-15 09:39".
  when: (iso) => (iso ? String(iso).replace("T", " ").slice(0, 16) : "–"),
};

function escape(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const recent = (seconds) => seconds != null && seconds < RECENT_SECONDS;
// A saved run that another process may still be training. One that ended in
// this dashboard is known to be done.
const writing = (run) =>
  run != null && !run.live && !run.ended_here && recent(run.updated_seconds_ago);
const lowerIsBetter = (direction) => direction === "lower";
const betterWord = (direction) => (lowerIsBetter(direction) ? "lower is better" : "higher is better");

/* ---------------------------------------------------------------- runs */

async function loadRuns() {
  let data;
  try {
    data = await getJSON("/api/runs");
  } catch {
    setStateChip("offline", "disconnected");
    return;
  }
  state.runs = data.runs || [];
  state.liveRun = data.live;
  state.canStart = Boolean(data.can_start);
  state.defaultGame = data.game;
  $("btn-new").hidden = !state.canStart;

  const known = new Set(state.runs.map((run) => run.run_id));
  if (state.wanted) {
    const wanted = state.wanted;
    state.wanted = null;
    if (known.has(wanted)) show(wanted, true);
  }
  // Until a run is picked the page shows the live one, or else the newest.
  if (!state.chosen || !known.has(state.run)) {
    const pick = state.liveRun || (state.runs.length ? state.runs[0].run_id : null);
    show(pick, false);
  }
  renderRunPicker();
  renderEmpty();
  refreshCompared();
}

function runNote(run) {
  const where = run.live ? "training here"
    : writing(run) ? "recently written"
      : `${fmt.int(run.generations)} gens`;
  return `${run.game} · ${where}`;
}

function renderRunPicker() {
  const select = $("run-select");
  const markup = state.runs
    .map((run) => `<option value="${escape(run.run_id)}" data-note="${escape(runNote(run))}">${escape(run.run_id)}</option>`)
    .join("");
  if (select.dataset.markup !== markup) {
    select.innerHTML = markup;
    select.dataset.markup = markup;
  }
  select.value = state.run || "";
  select.hidden = !state.runs.length;
  syncPicker(select);
}

/* Show `runId`, starting over everything that belongs to the run. */
function show(runId, chosen) {
  state.chosen = Boolean(chosen && runId);
  const address = new URL(window.location.href);
  if (state.chosen) address.searchParams.set("run", runId);
  else address.searchParams.delete("run");
  window.history.replaceState(null, "", address);
  if (runId === state.run) return;

  state.run = runId;
  state.status = null;
  closeSheet();
  state.curve = null;
  state.chartScale = null;
  state.generation = null;
  state.following = true;
  state.pinned = null;
  state.bounds = { first: 0, last: 0, available: [] };
  state.agents = [];
  state.shown = [];
  state.scene = null;
  state.paint = null;
  state.frame = 0;
  state.maxFrames = 0;
  state.championReport = null;
  state.job = null;
  clearTimeout(jobTimer);
  disarmStop();

  $("grid").innerHTML = "";
  $("gen-summary").innerHTML = "";
  $("frame-range").max = "0";
  fillRange($("frame-range"));
  for (const id of ["m-gen", "m-best", "m-mean", "m-stag", "m-eps", "m-rate"]) $(id).textContent = "–";
  $("run-detail").innerHTML = "";
  $("direction").textContent = "";
  renderChart();
  renderExport();
  renderJob();
  syncScrubber();
  renderRunPicker();

  document.body.classList.toggle("no-run", !runId);
  if (!runId) {
    document.title = "Evolvarium";
    return;
  }
  $("run-id").textContent = runId;
  heartbeat();
  pollStatus();
  loadCurve();
  attachJob();
}

function renderEmpty() {
  const empty = !state.runs.length;
  document.body.classList.toggle("no-run", empty);
  if (!empty) return;
  $("run-id").textContent = "no runs";
  setStateChip("offline", "no runs");
  $("empty-text").innerHTML = state.canStart
    ? "Start one with <strong>New run</strong>. Runs other processes train in the same runs directory appear here too."
    : "Runs appear here once one is started with <code>python -m evolvarium train</code>.";
}

/* ---------------------------------------------------------------- status */

async function pollStatus() {
  if (!state.run) return;
  try {
    const status = await runJSON("/api/status");
    if (!status) return;
    state.status = status;
    renderStatus(status);
    renderRunDetail(status);
    renderControl(status);

    const gens = await runJSON("/api/generations");
    if (!gens) return;
    state.bounds = gens;
    syncScrubber();

    const shown = state.generation && state.generation.generation;
    if (state.following && gens.available.length && gens.last !== shown) {
      await loadGeneration(null);
    }
    // A run read from disk sends no events, so its curve is read again when
    // it has grown.
    const newest = status.last_generation ? status.last_generation.generation : null;
    if (!status.live && state.curve && newest != null && newest !== curveEnd()) loadCurve();
  } catch {
    setStateChip("offline", "disconnected");
  }
}

function setStateChip(kind, text) {
  const chip = $("state-chip");
  chip.dataset.state = kind;
  $("state-text").textContent = text;
}

function renderStatus(status) {
  $("run-id").textContent = `${status.game} · ${status.run_id}`;
  document.title = `${status.run_id} · Evolvarium`;
  $("m-gen").textContent = fmt.int(status.generation);
  $("m-best").textContent = fmt.num(status.best_ever_score);
  $("m-stag").textContent = fmt.int(status.stagnation);
  $("m-eps").textContent = fmt.int(status.total_episodes);
  $("m-rate").textContent = fmt.dec(status.generations_per_minute, 1);
  const last = status.last_generation;
  $("m-mean").textContent = last ? fmt.num(last.mean_score) : "–";
  $("direction").textContent = betterWord(status.direction);

  const gate = status.gate || {};
  if (status.problem) {
    setStateChip("problem", "cannot load");
  } else if (!status.live) {
    if (writing(status)) {
      setStateChip("elsewhere", `written ${fmt.ago(status.updated_seconds_ago)}`);
    } else {
      setStateChip("offline", status.ended_here ? "ended" : "saved run");
    }
  } else if (gate.holding) {
    setStateChip("held", `held at gen ${gate.generation}`);
  } else if (!status.running) {
    setStateChip("offline", "stopped");
  } else {
    setStateChip("running", gate.hold_enabled ? "running" : "running free");
  }
}

function renderControl(status) {
  const gate = status.gate || {};
  const active = Boolean(status.live && status.running);
  for (const id of ["btn-continue", "btn-continue-5", "btn-hold", "btn-free"]) $(id).disabled = !active;
  $("btn-hold").setAttribute("aria-pressed", String(Boolean(gate.hold_enabled)));
  $("btn-stop").hidden = !active;
  if (!active) disarmStop();
  $("control-hint").hidden = !status.live;

  const hold = $("hold-status");
  hold.className = "hold-status";
  if (!status.live) {
    hold.textContent = writing(status)
      ? `This run was written to ${fmt.ago(status.updated_seconds_ago)}. If another process ` +
        "is training it, new generations appear here as they are saved."
      : status.ended_here
        ? "This run ended here and every generation is saved. It can still be replayed, " +
          "compared and exported."
        : "This run is not training. Its generations can still be replayed, compared and exported.";
  } else if (gate.holding) {
    hold.className = "hold-status held";
    hold.textContent =
      `Paused at generation ${gate.generation} for ${fmt.duration(gate.held_for)}. ` +
      "Nothing advances until you continue.";
  } else if (!status.running) {
    hold.textContent = "Training has stopped.";
  } else {
    hold.textContent = gate.hold_enabled
      ? (gate.credits > 0
          ? `Running with ${gate.credits} generation${gate.credits === 1 ? "" : "s"} of credit.`
          : "Running. Will pause at the next generation boundary.")
      : "Running freely. Generations will advance without waiting.";
  }
}

let stopTimer = null;

/* Stopping takes two clicks, so a run is not ended by a stray one. */
function stopRun() {
  const button = $("btn-stop");
  if (!button.classList.contains("armed")) {
    button.classList.add("armed");
    button.textContent = "Click again to stop";
    stopTimer = setTimeout(disarmStop, 4000);
    return;
  }
  disarmStop();
  postJSON("/api/runs/stop", { run: state.run })
    .then((result) => {
      if (result.error) toast(result.error, "bad");
      else toast(`${result.run} stops after the generation in progress`);
      pollStatus();
    })
    .catch((error) => toast(`Could not stop: ${error.message}`, "bad"));
}

function disarmStop() {
  clearTimeout(stopTimer);
  const button = $("btn-stop");
  button.classList.remove("armed");
  button.textContent = "Stop run";
}

function renderRunDetail(status) {
  const rows = [
    ["Run", status.run_id],
    ["Game", status.game],
    ["Settings", status.label || "–"],
    ["Seed", status.seed ?? "–"],
    ["Started", fmt.when(status.started_at)],
    ["Encoding", status.encoding ? `${status.encoding} · ${status.input_size} inputs` : "–"],
    ["Actions", status.actions && status.actions.length ? status.actions.join(", ") : "–"],
    ["Genome", status.genome_size == null ? "–" : `${fmt.int(status.genome_size)} params`],
    ["Scene", sceneText(status.scene)],
    ["Population", fmt.int(status.population_size)],
    ["Episodes", fmt.int(status.total_episodes)],
    ["Episodes / s", fmt.dec(status.episodes_per_second, 1)],
    ["Runtime", fmt.duration(status.elapsed_seconds)],
    ["Best at gen", fmt.int(status.best_ever_generation)],
    ["Stagnation", `${fmt.int(status.stagnation)} gens`],
    ["Workers", status.workers ?? "–"],
  ];
  if (status.resumed_from) rows.push(["Resumed from", status.resumed_from]);
  if (status.problem) rows.push(["Problem", status.problem]);
  $("run-detail").innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${escape(v)}</dd>`)
    .join("");
}

/* ---------------------------------------------------------------- charts */

/* The lowest and highest of `lists`, leaving out gaps. A range that sits well
   away from zero on one side starts at zero, so a small rise is not drawn as
   a large one. */
function extent(lists) {
  let lo = Infinity, hi = -Infinity;
  for (const list of lists) {
    for (const v of list) {
      if (v == null || !Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (lo === Infinity) return [0, 1];
  if (lo > 0 && lo <= hi / 2) lo = 0;
  if (hi < 0 && hi >= lo / 2) hi = 0;
  if (lo === hi) {
    if (lo === 0) return [0, 1];
    const pad = Math.abs(lo) * 0.1;
    return [lo - pad, hi + pad];
  }
  return [lo, hi];
}

/* About `count` ticks covering `lo` to `hi`, a step of 1, 2, 2.5 or 5 times a
   power of ten apart, and the digits their labels need. */
function niceScale(lo, hi, count) {
  const raw = (hi - lo) / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const multiple = [1, 2, 2.5, 5, 10].find((m) => m * power >= raw * (1 - 1e-9)) || 10;
  const step = multiple * power;
  const start = Math.floor(lo / step + 1e-9) * step;
  const end = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let i = 0, n = Math.round((end - start) / step); i <= n; i++) ticks.push(start + i * step);
  const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9) + (multiple === 2.5 ? 1 : 0));
  return { lo: start, hi: end, ticks, digits };
}

/* Draw `lines` into `svg` at the size it has on the page and return how it
   maps values to pixels. Each line is {xs, values, cls}, and a null value
   leaves a gap. The first line is drawn on top; with `fill` the area beyond
   it, toward the worse end, is shaded. */
function plot(svg, lines, options = {}) {
  const box = svg.getBoundingClientRect();
  const W = Math.max(60, Math.round(box.width));
  const H = Math.max(24, Math.round(box.height));
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const axis = options.axis !== false;
  const [lo, hi] = extent(lines.map((line) => line.values));
  const scale = axis ? niceScale(lo, hi, options.ticks || 4) : { lo, hi, ticks: [], digits: 0 };
  const labels = scale.ticks.map((v) => fixed(v, scale.digits));
  const widest = labels.reduce((m, text) => Math.max(m, text.length), 1);
  const padL = axis ? Math.ceil(widest * 6.2) + 12 : 2;
  const padR = axis ? 12 : 2, padT = axis ? 12 : 3, padB = axis ? 20 : 3;
  const innerW = Math.max(1, W - padL - padR);
  const innerH = Math.max(1, H - padT - padB);

  let x0 = Infinity, x1 = -Infinity;
  for (const line of lines) {
    if (!line.xs.length) continue;
    x0 = Math.min(x0, line.xs[0]);
    x1 = Math.max(x1, line.xs[line.xs.length - 1]);
  }
  if (x0 === Infinity) { x0 = 0; x1 = 1; }
  const xAt = (x) => padL + (x1 === x0 ? innerW / 2 : ((x - x0) / (x1 - x0)) * innerW);
  const yAt = (v) => padT + innerH - ((v - scale.lo) / (scale.hi - scale.lo)) * innerH;
  const xOf = (px) => (x1 === x0 ? x0 : x0 + ((px - padL) / innerW) * (x1 - x0));
  const at = (x, v) => `${xAt(x).toFixed(1)},${yAt(v).toFixed(1)}`;

  const points = (line) => {
    const found = [];
    line.values.forEach((v, i) => {
      if (v != null && Number.isFinite(v)) found.push([line.xs[i], v]);
    });
    return found;
  };

  let out = "";
  scale.ticks.forEach((v, i) => {
    const y = yAt(v).toFixed(1);
    out += `<line class="grid-line" x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"/>`;
    out += `<text class="axis-label" x="${padL - 6}" y="${(Number(y) + 3).toFixed(1)}" text-anchor="end">${labels[i]}</text>`;
  });

  const first = lines.length ? points(lines[0]) : [];
  if (options.fill && first.length > 1) {
    const id = `fade-${svg.id}`;
    const lower = lowerIsBetter(options.direction);
    const edge = (lower ? padT : padT + innerH).toFixed(1);
    const area = first.map(([x, v], i) => `${i ? "L" : "M"}${at(x, v)}`).join("") +
      `L${xAt(first[first.length - 1][0]).toFixed(1)},${edge}L${xAt(first[0][0]).toFixed(1)},${edge}Z`;
    out += `<defs><linearGradient id="${id}" x1="0" y1="${lower ? 1 : 0}" x2="0" y2="${lower ? 0 : 1}">` +
      `<stop offset="0%" class="fade-strong"/><stop offset="100%" class="fade-clear"/>` +
      `</linearGradient></defs><path class="series-area" fill="url(#${id})" d="${area}"/>`;
  }

  // Drawn last to first, so the first line ends up on top.
  for (let k = lines.length - 1; k >= 0; k--) {
    const line = lines[k];
    let d = "", pen = false, count = 0, lone = null;
    line.values.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${at(line.xs[i], v)}`;
      pen = true;
      count++;
      lone = [line.xs[i], v];
    });
    if (count > 1) out += `<path class="series ${line.cls}" d="${d}"/>`;
    else if (count === 1) {
      out += `<circle class="point ${line.cls}" cx="${xAt(lone[0]).toFixed(1)}" cy="${yAt(lone[1]).toFixed(1)}" r="2.5"/>`;
    }
  }

  for (const dot of options.dots || []) {
    out += `<circle class="record-dot" cx="${xAt(dot.x).toFixed(1)}" cy="${yAt(dot.v).toFixed(1)}" r="2.6"/>`;
  }
  for (const x of options.notches || []) {
    const px = xAt(x).toFixed(1);
    out += `<line class="notch" x1="${px}" y1="${H - 3}" x2="${px}" y2="${H}"/>`;
  }
  if (options.marker != null && options.marker >= x0 && options.marker <= x1) {
    const px = xAt(options.marker).toFixed(1);
    out += `<line class="marker" x1="${px}" y1="${padT}" x2="${px}" y2="${padT + innerH}"/>`;
  }
  if (axis) {
    out += `<text class="axis-label" x="${padL}" y="${H - 5}">gen ${x0}</text>`;
    out += `<text class="axis-label" x="${W - padR}" y="${H - 5}" text-anchor="end">gen ${x1}</text>`;
  }
  svg.innerHTML = out;
  return { xAt, yAt, xOf, padL, padT, innerW, innerH, W, H, x0, x1 };
}

async function loadCurve() {
  if (!state.run) return;
  try {
    const curve = await runJSON("/api/curve");
    if (!curve) return;
    state.curve = curve;
    renderChart();
  } catch { /* the chart is optional */ }
}

function curveEnd() {
  const n = state.curve ? state.curve.generation.length : 0;
  return n ? state.curve.generation[n - 1] : null;
}

function extendCurve(summary) {
  const curve = state.curve;
  if (!curve) return;
  const last = curveEnd();
  if (last != null && summary.generation <= last) return;
  if (last != null && summary.generation > last + 1) {
    // A generation finished between loading the curve and the stream opening.
    loadCurve();
    return;
  }
  for (const key of Object.keys(curve)) {
    curve[key].push(key === "record" ? Boolean(summary.record) : summary[key] ?? 0);
  }
}

const MINIS = [["mean_steps", 1], ["diversity", 3], ["mutation_rate", 3]];

function renderChart() {
  const curve = state.curve;
  const n = curve ? curve.generation.length : 0;
  $("chart-wrap").classList.toggle("has-data", n > 0);
  hideReadout();
  const shown = state.generation ? state.generation.generation : null;
  const direction = state.status ? state.status.direction : "higher";

  if (!n) {
    $("chart").innerHTML = "";
    state.chartScale = null;
  } else {
    const xs = curve.generation;
    const dots = [];
    curve.record.forEach((record, i) => {
      if (record) dots.push({ x: xs[i], v: curve.max_episode_score[i] });
    });
    state.chartScale = plot($("chart"), [
      { xs, values: curve.max_episode_score, cls: "best" },
      { xs, values: curve.mean_score, cls: "mean" },
    ], { fill: true, direction, dots, marker: shown, ticks: 4 });
  }

  const notches = n ? curve.generation.filter((_, i) => curve.record[i]) : [];
  for (const [key, digits] of MINIS) {
    const svg = $(`mini-${key}`);
    const values = n ? curve[key] : [];
    if (!values.length) {
      svg.innerHTML = "";
      $(`mini-${key}-v`).textContent = "–";
      continue;
    }
    plot(svg, [{ xs: curve.generation, values, cls: "trace" }],
      { axis: false, notches, marker: shown });
    $(`mini-${key}-v`).textContent = fmt.dec(values[values.length - 1], digits);
  }
}

/* The index in the ascending `list` closest to `value`. */
function nearest(list, value) {
  let lo = 0, hi = list.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo > 0 && Math.abs(list[lo - 1] - value) <= Math.abs(list[lo] - value) ? lo - 1 : lo;
}

function pointedGeneration(event) {
  const scale = state.chartScale;
  if (!scale || !state.curve || !state.curve.generation.length) return null;
  const px = event.clientX - $("chart").getBoundingClientRect().left;
  return nearest(state.curve.generation, scale.xOf(px));
}

function chartHover(event) {
  const index = pointedGeneration(event);
  if (index == null) return;
  const curve = state.curve, scale = state.chartScale;
  const x = scale.xAt(curve.generation[index]);
  const guide = $("chart-guide");
  guide.style.left = `${x}px`;
  guide.style.top = `${scale.padT}px`;
  guide.style.height = `${scale.innerH}px`;
  guide.hidden = false;

  const readout = $("chart-readout");
  readout.innerHTML =
    `<strong>gen ${curve.generation[index]}</strong>` +
    `<span class="key best">best ${fmt.num(curve.max_episode_score[index])}</span>` +
    `<span class="key mean">mean ${fmt.num(curve.mean_score[index])}</span>` +
    (curve.record[index] ? `<span class="tag ok">record</span>` : "");
  readout.hidden = false;
  const width = readout.offsetWidth;
  readout.style.left = `${Math.max(scale.padL, Math.min(x + 10, scale.W - width - 4))}px`;
}

function hideReadout() {
  $("chart-guide").hidden = true;
  $("chart-readout").hidden = true;
}

/* A click on the chart opens the kept generation nearest to it. */
function chartClick(event) {
  const index = pointedGeneration(event);
  const available = state.bounds.available;
  if (index == null || !available.length) return;
  const generation = state.curve.generation[index];
  goTo(available[nearest(available, generation)]);
}

/* ---------------------------------------------------------------- replay */

/* A game declares its drawing once, as a scene of layers and labels, and a
   replay is every frame it drew, packed as changes by evolvarium/replays.py.
   Each frame has one operation for each layer and label, in scene order: same,
   shift (drop items from the front, append new ones), delta (every item moves
   by a little) or full. Runs of identical operations are stored once, and the
   numbers are zigzag varints in a second stream.

   A cursor decodes forward on demand and keeps its state every KEY_EVERY
   frames, so playing decodes one frame per tick and a seek backward replays
   at most KEY_EVERY frames. */

const FIELDS = { cell: 2, dot: 2, box: 4, line: 4 };
const SHIFT = 1, DELTA = 2, FULL = 3;
const SHIFT_ESCAPE = 7, COUNT_ESCAPE = 63;
const KEY_EVERY = 64;

function byteStream(text) {
  const raw = atob(text);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return { bytes, at: 0 };
}

function unsigned(input) {
  let value = 0, scale = 1, byte;
  do {
    byte = input.bytes[input.at++];
    value += (byte & 0x7f) * scale;
    scale *= 128;
  } while (byte >= 0x80);
  return value;
}

function signed(input) {
  const value = unsigned(input);
  return value % 2 ? -(value + 1) / 2 : value / 2;
}

class ReplayCursor {
  constructor(scene, replay) {
    this.fields = scene.layers.map((layer) => FIELDS[layer.shape])
      .concat(scene.labels.map(() => 1));
    this.frames = replay.frames;
    this.truncated = Boolean(replay.truncated);
    this.ops = byteStream(replay.ops);
    this.data = byteStream(replay.data);
    this.keys = [];
    this.index = -1;
    this.run = 0;
    this.codes = null;
    this.state = this.fields.map(() => []);
    this.start = this.mark();
  }

  // Lists are replaced, never changed in place, so a mark can share them.
  mark() {
    return {
      index: this.index, opsAt: this.ops.at, dataAt: this.data.at,
      run: this.run, codes: this.codes, state: this.state.slice(),
    };
  }

  restore(mark) {
    this.index = mark.index;
    this.ops.at = mark.opsAt;
    this.data.at = mark.dataAt;
    this.run = mark.run;
    this.codes = mark.codes;
    this.state = mark.state.slice();
  }

  /* Each layer's items, in replay units, at `frame`. */
  seek(frame) {
    const target = Math.max(0, Math.min(frame, this.frames - 1));
    const key = this.keys[Math.min(Math.floor(target / KEY_EVERY), this.keys.length - 1)]
      || this.start;
    if (target < this.index || key.index > this.index) this.restore(key);
    while (this.index < target) this.step();
    return this.state;
  }

  item(before) {
    const item = new Array(before.length);
    for (let i = 0; i < before.length; i++) item[i] = before[i] + signed(this.data);
    return item;
  }

  step() {
    if (!this.run) {
      this.run = unsigned(this.ops);
      this.codes = this.ops.bytes.subarray(this.ops.at, this.ops.at + this.fields.length);
      this.ops.at += this.fields.length;
    }
    this.run--;
    const state = this.state;
    for (let c = 0; c < this.codes.length; c++) {
      const code = this.codes[c];
      const kind = code & 3;
      if (kind === SHIFT) {
        let count = code >> 5;
        if (count === SHIFT_ESCAPE) count = unsigned(this.data);
        const items = state[c].slice(code >> 2 & 7);
        for (let k = 0; k < count; k++) items.push(this.item(items[items.length - 1]));
        state[c] = items;
      } else if (kind === DELTA) {
        state[c] = state[c].map((item) => this.item(item));
      } else if (kind === FULL) {
        let count = code >> 2;
        if (count === COUNT_ESCAPE) count = unsigned(this.data);
        const items = [];
        let before = new Array(this.fields[c]).fill(0);
        for (let k = 0; k < count; k++) items.push(before = this.item(before));
        state[c] = items;
      }
    }
    this.index++;
    const slot = this.index / KEY_EVERY;
    if (Number.isInteger(slot) && !this.keys[slot]) this.keys[slot] = this.mark();
  }
}

function cursorOf(agent) {
  if (agent._cursor === undefined) {
    const replay = agent.replay;
    // Replays from before scenes have no operation stream and are not drawn.
    agent._cursor = state.scene && replay && replay.ops != null && replay.frames > 0
      ? new ReplayCursor(state.scene, replay)
      : null;
  }
  return agent._cursor;
}

const trim = (v) => Math.round(v * 1000) / 1000;

/* Markup for one frame of a scene. Positions arrive as whole multiples of the
   scene's unit and are drawn in a viewBox of the scene's size, with y turned
   over for a scene whose y grows upward. */
function makePainter(scene) {
  const { width, height, unit } = scene;
  const [ox, oy] = scene.origin;
  const x = (v) => trim(v * unit - ox);
  // Screen top of something reaching `h` from game y `v`.
  const top = scene.y_up ? (v, h) => oy + height - v * unit - h : (v) => v * unit - oy;

  const layers = scene.layers.map((layer) => {
    const s = layer.size;
    let shape;
    if (layer.shape === "cell") {
      const inset = (1 - s) / 2;
      const rest = `width="${trim(s)}" height="${trim(s)}" rx="${trim(layer.rounded ? s / 2 : s * 0.16)}"`;
      shape = (item) =>
        `<rect x="${trim(item[0] * unit - ox + inset)}" y="${trim(top(item[1], 1) + inset)}" ${rest}`;
    } else if (layer.shape === "dot") {
      shape = (item) => `<circle cx="${x(item[0])}" cy="${trim(top(item[1], 0))}" r="${trim(s)}"`;
    } else if (layer.shape === "box") {
      shape = (item) => {
        const w = item[2] * unit, h = item[3] * unit;
        const rx = layer.rounded ? trim(Math.min(w, h) / 2) : 0;
        return `<rect x="${x(item[0])}" y="${trim(top(item[1], h))}" width="${trim(w)}" height="${trim(h)}" rx="${rx}"`;
      };
    } else {
      shape = (item) =>
        `<line x1="${x(item[0])}" y1="${trim(top(item[1], 0))}" x2="${x(item[2])}" y2="${trim(top(item[3], 0))}" stroke-width="${trim(s)}"`;
    }

    return (items) => {
      const n = items.length;
      if (!n) return "";
      let out = `<g class="tone-${layer.tone}">`;
      for (let i = 0; i < n; i++) {
        out += shape(items[i]);
        if (layer.fade) out += ` opacity="${(0.4 + (0.55 * (i + 1)) / (n + 1)).toFixed(2)}"`;
        out += "/>";
      }
      return `${out}</g>`;
    };
  });

  return (lists) => layers.map((paint, i) => paint(lists[i])).join("");
}

/* The labels of a decoded frame, as text kept to their digits. */
function labelText(scene, lists) {
  const first = scene.layers.length;
  return scene.labels.map((label, j) => {
    const item = lists[first + j][0];
    return (item ? item[0] * 10 ** -label.digits : 0).toFixed(label.digits);
  });
}

function sceneText(scene) {
  if (!scene) return "–";
  const layers = scene.layers.length;
  return `${trim(scene.width)} × ${trim(scene.height)} · ${layers} layer${layers === 1 ? "" : "s"}`;
}

/* ---------------------------------------------------------------- generation */

async function loadGeneration(number) {
  let data;
  try {
    data = await runJSON("/api/generation", { n: number });
  } catch {
    return;
  }
  if (!data) return;
  if (data.error) {
    if (number != null) toast(`Generation ${number} is no longer kept`, "bad");
    return;
  }

  state.generation = data;
  state.scene = data.scene || null;
  state.paint = state.scene ? makePainter(state.scene) : null;
  state.agents = data.agents || [];
  state.maxFrames = state.agents.reduce((m, a) => Math.max(m, cursorOf(a)?.frames ?? 0), 0);
  state.frame = 0;
  $("frame-range").max = String(Math.max(0, state.maxFrames - 1));

  renderGenerationSummary(data.summary);
  renderGrid();
  if (state.sheetAgent) {
    const same = state.agents.find((a) => a.id === state.sheetAgent.id);
    if (same) showAgent(same);
  }
  syncScrubber();
  renderChart();
}

function renderGenerationSummary(summary) {
  if (!summary) { $("gen-summary").innerHTML = ""; return; }
  const ends = summary.ends || summary.deaths || {};
  const endText = Object.entries(ends)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k} ${v}`)
    .join(" · ") || "–";

  const items = [
    ["Generation", summary.generation, summary.record ? "record" : ""],
    ["Best score", fmt.num(summary.max_episode_score), summary.record ? "record" : ""],
    ["Mean score", fmt.num(summary.mean_score), ""],
    ["Best fitness", fmt.int(summary.best_fitness), ""],
    ["Mean steps", fmt.dec(summary.mean_steps, 1), ""],
    ["Diversity", fmt.dec(summary.diversity, 3), ""],
    ["Mutation", fmt.dec(summary.mutation_rate, 3), ""],
    ["Ends", endText, ""],
    ["Wall time", fmt.duration(summary.elapsed), ""],
  ];
  $("gen-summary").innerHTML = items
    .map(([k, v, cls]) => `<div class="item ${cls}"><span>${k}</span><span>${escape(v)}</span></div>`)
    .join("");
}

/* ---------------------------------------------------------------- grid */

function sortedAgents() {
  const key = state.sort;
  // Scores sort best first, which for some games is lowest first.
  const sign = lowerIsBetter(state.status && state.status.direction) ? -1 : 1;
  const copy = state.agents.slice();
  copy.sort((a, b) => {
    if (key === "score") return sign * (b.score - a.score) || b.fitness - a.fitness;
    if (key === "steps") return b.steps - a.steps || b.fitness - a.fitness;
    return b.fitness - a.fitness;
  });
  return state.limit > 0 ? copy.slice(0, state.limit) : copy;
}

function viewBox() {
  const scene = state.scene;
  return scene ? `0 0 ${trim(scene.width)} ${trim(scene.height)}` : "0 0 1 1";
}

function renderGrid() {
  const grid = $("grid");
  grid.innerHTML = "";
  const agents = sortedAgents();
  state.shown = agents;
  if (!agents.length) return;

  const box = viewBox();
  const fragment = document.createDocumentFragment();

  agents.forEach((agent, rank) => {
    const card = document.createElement("div");
    card.className = `agent${rank === 0 ? " top" : ""}`;
    card.dataset.id = agent.id;
    card.tabIndex = 0;

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "stage");
    svg.setAttribute("viewBox", box);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    card.appendChild(svg);

    const cap = document.createElement("div");
    cap.className = "cap";
    cap.innerHTML =
      `<span class="rk">#${rank + 1}</span>` +
      `<span class="sc">${fmt.num(agent.best_episode_score)}</span>` +
      `<span>${Math.round(agent.steps)}s</span>`;
    card.appendChild(cap);

    card.addEventListener("click", () => showAgent(agent));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showAgent(agent); }
    });

    agent._svg = svg;
    fragment.appendChild(card);
  });

  grid.appendChild(fragment);
  paintFrame();
}

/* Draw an agent's replay at the current frame into `svg`, returning its
   cursor, or null when there is nothing to draw. */
function paintAgent(agent, svg) {
  const cursor = cursorOf(agent);
  if (!cursor || !state.paint) { svg.innerHTML = ""; return null; }
  try {
    svg.innerHTML = state.paint(cursor.seek(state.frame));
    return cursor;
  } catch {
    // A replay that cannot be read stays blank rather than stopping playback.
    agent._cursor = null;
    svg.innerHTML = "";
    return null;
  }
}

function paintFrame() {
  for (const agent of state.shown) {
    const svg = agent._svg;
    if (!svg) continue;
    const cursor = paintAgent(agent, svg);
    // A replay that was cut short ends before the game did, so its last frame
    // is not an ending and must not be drawn as one.
    const ended = cursor != null && cursor.index >= cursor.frames - 1;
    svg.parentElement.classList.toggle("dead", ended && !cursor.truncated);
    svg.parentElement.classList.toggle("cut", ended && cursor.truncated);
  }
  if (state.sheetAgent) paintSheet();

  $("frame-range").value = String(state.frame);
  fillRange($("frame-range"));
  $("frame-label").textContent = `${state.frame} / ${Math.max(0, state.maxFrames - 1)}`;
}

function tick(now) {
  if (state.playing && state.maxFrames > 0) {
    if (!state.lastTick) state.lastTick = now;
    state.tickAccumulator += now - state.lastTick;
    const interval = TICK_MS * state.speed;
    const before = state.frame;
    while (state.tickAccumulator >= interval) {
      state.tickAccumulator -= interval;
      state.frame = (state.frame + 1) % state.maxFrames;
    }
    if (state.frame !== before) paintFrame();
  }
  state.lastTick = now;
  requestAnimationFrame(tick);
}

/* ---------------------------------------------------------------- agent sheet */

function showAgent(agent) {
  state.sheetAgent = agent;
  $("agent-sheet").hidden = false;
  $("sheet-title").textContent = agent.id;
  const cursor = cursorOf(agent);
  const reads = ["frame", ...(state.scene ? state.scene.labels.map((l) => l.name) : [])]
    .map((name) => `<div class="read"><span>${escape(name)}</span><span>–</span></div>`)
    .join("");

  const terms = agent.terms || {};
  const termRows = Object.entries(terms)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([k, v]) => `<dt>${escape(k)}</dt><dd>${fmt.dec(v, 1)}</dd>`)
    .join("");

  $("sheet-body").innerHTML = `
    <svg class="stage sheet-stage" id="sheet-stage" viewBox="${viewBox()}"
         preserveAspectRatio="xMidYMid meet"></svg>
    <div class="readout" id="sheet-readout">${reads}</div>
    <dl class="detail">
      <dt>Rank origin</dt><dd>${escape(agent.origin)}</dd>
      <dt>Generation</dt><dd>${agent.generation}</dd>
      <dt>Lineage age</dt><dd>${agent.age}</dd>
      <dt>Fitness</dt><dd>${fmt.dec(agent.fitness, 1)}</dd>
      <dt>Mean score</dt><dd>${fmt.num(agent.score)}</dd>
      <dt>Best episode</dt><dd>${fmt.num(agent.best_episode_score)}</dd>
      <dt>Mean steps</dt><dd>${fmt.dec(agent.steps, 1)}</dd>
      <dt>Ended by</dt><dd>${escape(agent.end ?? agent.death)}</dd>
      <dt>Replay</dt><dd>${cursor ? `${cursor.frames} frames${cursor.truncated ? " (truncated)" : ""}` : "none"}</dd>
    </dl>
    <p class="section-label">Fitness terms</p>
    <dl class="detail">${termRows}</dl>
  `;
  paintSheet();
}

/* The sheet plays the same frame as the grid, larger, with the scene's labels. */
function paintSheet() {
  const svg = $("sheet-stage");
  if (!svg) return;
  const cursor = paintAgent(state.sheetAgent, svg);
  const values = $("sheet-readout").querySelectorAll(".read span:last-child");
  if (!cursor) return;
  values[0].textContent = `${cursor.index} / ${cursor.frames - 1}`;
  labelText(state.scene, cursor.state).forEach((text, j) => { values[j + 1].textContent = text; });
}

function closeSheet() {
  state.sheetAgent = null;
  $("agent-sheet").hidden = true;
}

/* ---------------------------------------------------------------- export */

/* Export offers two picks. The champion is the genome behind the run's best
   score, which one lucky game can put there. The best average plays every
   genome the run holds on fresh games, as the export command does, and runs
   as a job on the server that the page polls. */

let jobTimer = null;

function setPick(pick) {
  state.exportPick = pick;
  for (const input of document.querySelectorAll('input[name="pick"]')) input.checked = input.value === pick;
  $("average-controls").hidden = pick !== "average";
  $("champion-controls").hidden = pick !== "champion";
  renderExport();
  renderJob();
}

/* The report the export button would write, for the pick shown. */
function exportReport() {
  if (state.exportPick === "champion") return state.championReport;
  const job = state.job;
  return job && job.state === "done" ? job.report : null;
}

async function previewExport() {
  $("btn-preview").disabled = true;
  try {
    const report = await runJSON("/api/export/preview", { n: $("export-gen").value.trim() });
    if (!report) return;
    if (report.error) {
      toast(report.error, "bad");
      return;
    }
    state.championReport = report;
    renderExport();
  } catch (error) {
    toast(`Could not calculate: ${error.message}`, "bad");
  } finally {
    $("btn-preview").disabled = false;
  }
}

async function chooseByAverage() {
  $("btn-choose").disabled = true;
  try {
    const result = await postJSON("/api/export/choose", { run: state.run });
    if (result.error) {
      toast(result.error, "bad");
      if (result.job && result.job.run === state.run) attach(result.job);
      return;
    }
    attach(result.job);
  } catch (error) {
    toast(`Could not start choosing: ${error.message}`, "bad");
  } finally {
    renderJob();
  }
}

function attach(job) {
  state.job = job;
  renderJob();
  renderExport();
  clearTimeout(jobTimer);
  if (job && job.state === "running") jobTimer = setTimeout(pollJob, JOB_MS);
}

async function pollJob() {
  const job = state.job;
  if (!job) return;
  try {
    const data = await getJSON(`/api/export/job?id=${encodeURIComponent(job.id)}`);
    if (state.job !== job) return;
    attach(data.job);
  } catch {
    if (state.job === job) jobTimer = setTimeout(pollJob, JOB_MS * 4);
  }
}

/* Pick up a choice already made or under way for the run shown. */
async function attachJob() {
  const run = state.run;
  try {
    const data = await getJSON("/api/export/job");
    if (state.run === run && data.job && data.job.run === run) attach(data.job);
  } catch { /* no choice to show */ }
}

const STAGES = {
  reading: "Gathering candidates",
  screen: (job) => `Screening ${fmt.int(job.candidates)} candidates`,
  final: "Final games for the leaders",
};

function renderJob() {
  const job = state.exportPick === "average" ? state.job : null;
  $("job").hidden = !job;
  $("btn-choose").textContent = state.job && state.job.state === "done" ? "Choose again" : "Choose";
  $("btn-choose").disabled = Boolean(state.job && state.job.state === "running");
  if (!job) return;
  const stage = STAGES[job.stage] || job.stage;
  let text = typeof stage === "function" ? stage(job) : stage;
  if (job.state === "done") text = `Chose from ${fmt.int(job.candidates)} candidates`;
  else if (job.state === "cancelled") text = "Cancelled";
  else if (job.state === "failed") text = "Could not choose";
  $("job-stage").textContent = text;
  $("job-count").textContent = job.games ? `${fmt.int(job.played)} / ${fmt.int(job.games)} games` : "";
  const share = job.state === "done" ? 1 : job.games ? job.played / job.games : 0;
  $("job-bar").style.width = `${(share * 100).toFixed(1)}%`;
  $("job").dataset.state = job.state;
  $("job-time").textContent = job.error || fmt.duration(job.seconds);
  $("job-cancel").hidden = job.state !== "running";
}

async function cancelJob() {
  const job = state.job;
  if (!job) return;
  try {
    const result = await postJSON("/api/export/cancel", { job: job.id });
    if (result.error) toast(result.error, "bad");
    pollJob();
  } catch (error) {
    toast(`Could not cancel: ${error.message}`, "bad");
  }
}

function renderExport() {
  const report = exportReport();
  $("btn-export").disabled = !(report && report.ready);
  const availability = $("export-availability");
  if (report && report.torch_available) {
    availability.textContent = ".json and .pt";
    availability.className = "tag ok";
  } else {
    availability.textContent = report ? ".json" : "";
    availability.className = "tag";
  }
  if (!report) {
    $("export-report").innerHTML = state.exportHint;
    return;
  }
  const t = report.training || {};

  const cards = [
    ["Generation", report.resolved_generation],
    ["Parameters", fmt.int(report.parameter_count)],
    ["Est. size", fmt.bytes(report.estimated_size_bytes)],
    ["Best score", fmt.num(t.best_score)],
    ["At generation", fmt.int(t.best_generation)],
    ["Generations", fmt.int(t.generations)],
    ["Episodes", fmt.int(t.episodes)],
    ["Compute time", fmt.duration(t.compute_seconds)],
    ["Records set", fmt.int(t.records_set)],
    ["Final mean", fmt.num(t.mean_score_final)],
    ["Final mean steps", fmt.dec(t.mean_steps_final, 1)],
    ["Achievements", `${report.achievements_unlocked} / ${report.achievements_total}`],
  ];

  const achievements = (report.achievements || [])
    .map((a) => `<span class="ach ${a.unlocked ? "on" : ""}" title="${escape(a.detail)}">
        ${escape(a.label)}${a.unlocked ? `<span class="gen">gen ${a.generation}</span>` : ""}
      </span>`)
    .join("");

  const ends = Object.entries(t.ends || t.deaths || {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k} ${fmt.int(v)}`)
    .join(" · ");

  let selection = "";
  const chosen = report.selection;
  if (chosen) {
    const rows = [
      ["Picked", chosen.source],
      ["Mean of pick", fmt.num(chosen.selection_mean)],
      ["Champion's mean", fmt.num(chosen.stored_champion_mean)],
      ["Candidates", fmt.int(chosen.candidates)],
      ["Games", fmt.int(chosen.games)],
    ];
    const note = chosen.note ? chosen.note.charAt(0).toUpperCase() + chosen.note.slice(1) : "";
    selection = `
      <p class="section-label">Selection · ${escape(chosen.rule)}</p>
      <dl class="detail">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${escape(v)}</dd>`).join("")}</dl>
      ${note ? `<p class="hint">${escape(note)}.</p>` : ""}`;
  }

  let notice = "";
  if (!report.genome_available) {
    notice = report.pick === "champion"
      ? `<p class="hint warn">No champion genome is archived at or before generation ${escape(report.requested_generation ?? report.resolved_generation)}.</p>`
      : `<p class="hint warn">The chosen genome could not be read back.</p>`;
  } else if (!report.torch_available) {
    notice = `<p class="hint">Only the JSON model is written. <code>pip install torch</code> adds a .pt copy beside it.</p>`;
  }

  $("export-report").innerHTML = `
    <div class="report-grid">
      ${cards.map(([k, v]) => `<div class="report-item"><span class="k">${k}</span><span class="v">${escape(v)}</span></div>`).join("")}
    </div>
    ${selection}
    <p class="section-label">Achievements</p>
    <div class="achievements">${achievements}</div>
    <p class="section-label">How episodes ended</p>
    <p class="hint">${escape(ends || "–")}</p>
    <p class="section-label">Destination</p>
    <p class="hint"><code>${escape(report.destination)}</code></p>
    ${notice}
  `;
}

async function doExport() {
  const report = exportReport();
  if (!report) return;
  const body = state.exportPick === "average"
    ? { run: state.run, job: state.job.id }
    : { run: state.run, generation: report.resolved_generation };
  $("btn-export").disabled = true;
  try {
    const result = await postJSON("/api/export", body);
    if (result.error) {
      toast(result.error, "bad");
      return;
    }
    toast(`Exported generation ${result.generation}`, "good");
    const node = document.createElement("div");
    node.className = "written";
    node.textContent =
      `${result.path} · ${fmt.bytes(result.size_bytes)} · ` +
      `${fmt.int(result.parameter_count)} params · ` +
      `${result.achievements_unlocked} achievements` +
      (result.pt ? ` · ${result.pt}` : "");
    $("export-report").appendChild(node);
  } catch (error) {
    toast(`Export failed: ${error.message}`, "bad");
  } finally {
    $("btn-export").disabled = false;
  }
}

/* ---------------------------------------------------------------- new run */

/* The form is built from the game's own settings, so a new game needs nothing
   here. Every value starts where this dashboard was opened, and only the
   changed ones are sent; the server checks them as the command line does. */

async function openDrawer() {
  $("drawer").hidden = false;
  if (!state.gamesLoaded) await loadGames();
  pickerButtons.get($("new-game")).focus();
}

function closeDrawer() {
  $("drawer").hidden = true;
}

async function loadGames() {
  try {
    const data = await getJSON("/api/games");
    const select = $("new-game");
    select.innerHTML = data.games
      .map((game) => `<option value="${escape(game)}">${escape(game)}</option>`)
      .join("");
    select.value = data.default;
    syncPicker(select);
    state.gamesLoaded = true;
    await loadSettings(select.value);
  } catch (error) {
    toast(`Could not list the games: ${error.message}`, "bad");
  }
}

async function loadSettings(game) {
  const box = $("new-settings");
  box.innerHTML = `<p class="hint">Reading the settings of ${escape(game)}…</p>`;
  showProblems([]);
  state.settings = { game, rows: [], inputs: new Map() };
  let data;
  try {
    data = await getJSON(`/api/settings?game=${encodeURIComponent(game)}`);
  } catch (error) {
    box.innerHTML = `<p class="hint warn">Could not read the settings: ${escape(error.message)}</p>`;
    return;
  }
  if ($("new-game").value !== game) return;
  if (data.error) {
    box.innerHTML = `<p class="hint warn">${escape(data.error)}</p>`;
    return;
  }
  state.settings.rows = data.settings;
  renderSettings();
}

function renderSettings() {
  const { game, rows } = state.settings;
  const box = $("new-settings");
  box.innerHTML = "";
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row.group)) groups.set(row.group, []);
    groups.get(row.group).push(row);
  }
  for (const [group, members] of groups) {
    const details = document.createElement("details");
    details.className = "settings-group";
    details.dataset.group = group;
    details.open = group === "game";
    const summary = document.createElement("summary");
    summary.innerHTML =
      `<span>${escape(group === "game" ? game : group)}</span><span class="changed-count"></span>`;
    details.appendChild(summary);
    for (const row of members) details.appendChild(settingRow(row));
    box.appendChild(details);
  }
  markChanged();
}

function settingRow(row) {
  const node = document.createElement("label");
  node.className = "setting";
  node.dataset.name = row.name;

  let input;
  if (row.input === "flag") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = row.value === "true";
  } else if (row.choices && !row.several) {
    input = document.createElement("select");
    input.className = "select";
    input.dataset.picker = "mono";
    const options = row.choices.includes(row.value) ? row.choices : [row.value, ...row.choices];
    input.innerHTML = options.map((c) => `<option value="${escape(c)}">${escape(c)}</option>`).join("");
    input.value = row.value;
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.value = row.value;
    input.spellcheck = false;
    if (!row.several && row.input === "whole") input.inputMode = "numeric";
    else if (!row.several && row.input === "number") input.inputMode = "decimal";
  }
  input.name = row.name;
  input.title = row.kind;
  input.setAttribute("aria-label", row.name);
  input.addEventListener("input", markChanged);
  input.addEventListener("change", markChanged);
  state.settings.inputs.set(row.name, input);

  // Game settings keep their whole name; the rest are already under their section.
  // Long names break after a dot or underscore rather than mid-word.
  const name = document.createElement("span");
  name.className = "setting-name";
  const shown = row.group === "game" ? row.name : row.name.slice(row.group.length + 1);
  name.innerHTML = escape(shown).replace(/[._]/g, "$&<wbr>");
  name.title = row.name;
  node.append(name, input);
  if (input.tagName === "SELECT") picker(input);

  if (row.several) {
    const hint = document.createElement("small");
    hint.textContent = row.choices ? `any of ${row.choices.join(", ")}, comma separated` : "comma separated";
    node.appendChild(hint);
  }
  return node;
}

function settingValue(row) {
  const input = state.settings.inputs.get(row.name);
  if (!input) return row.value;
  return row.input === "flag" ? (input.checked ? "true" : "false") : input.value.trim();
}

/* Two lists that differ only in spacing are the same setting. */
function sameValue(row, a, b) {
  if (!row.several) return a.trim() === b.trim();
  const items = (text) => text.split(",").map((part) => part.trim()).filter(Boolean).join(",");
  return items(a) === items(b);
}

function changedSettings() {
  const changed = {};
  for (const row of state.settings.rows) {
    const value = settingValue(row);
    if (!sameValue(row, value, row.value)) changed[row.name] = value;
  }
  return changed;
}

function markChanged() {
  const changed = changedSettings();
  const counts = new Map();
  for (const row of state.settings.rows) {
    const on = row.name in changed;
    const node = $("new-settings").querySelector(`.setting[data-name="${CSS.escape(row.name)}"]`);
    if (node) node.classList.toggle("changed", on);
    if (on) counts.set(row.group, (counts.get(row.group) || 0) + 1);
  }
  for (const details of $("new-settings").querySelectorAll(".settings-group")) {
    const count = counts.get(details.dataset.group) || 0;
    details.querySelector(".changed-count").textContent = count ? `${count} changed` : "";
  }
}

function showProblems(problems) {
  const list = $("new-problems");
  list.hidden = !problems.length;
  list.innerHTML = problems.map((p) => `<li>${escape(p)}</li>`).join("");
  for (const node of document.querySelectorAll("#new-run .bad")) node.classList.remove("bad");
  for (const problem of problems) {
    for (const row of state.settings.rows) {
      if (!problem.startsWith(`${row.name} `) && !problem.includes(` ${row.name};`)) continue;
      const node = $("new-settings").querySelector(`.setting[data-name="${CSS.escape(row.name)}"]`);
      if (!node) continue;
      node.classList.add("bad");
      node.closest("details").open = true;
    }
    if (problem.startsWith("generations ")) $("field-generations").classList.add("bad");
    if (problem.startsWith("seed ")) $("field-seed").classList.add("bad");
  }
}

function resetSettings() {
  $("new-generations").value = "";
  $("new-seed").value = "";
  showProblems([]);
  renderSettings();
}

async function startRun(event) {
  event.preventDefault();
  const button = $("new-start");
  button.disabled = true;
  const raw = (id) => $(id).value.trim() || null;
  try {
    const result = await postJSON("/api/runs/start", {
      game: $("new-game").value,
      settings: changedSettings(),
      generations: raw("new-generations"),
      seed: raw("new-seed"),
    });
    if (result.problems) {
      showProblems(result.problems);
      return;
    }
    showProblems([]);
    closeDrawer();
    toast(`Started ${result.run}, seed ${result.seed}`, "good");
    await loadRuns();
    show(result.run, true);
  } catch (error) {
    toast(`Could not start: ${error.message}`, "bad");
  } finally {
    button.disabled = false;
  }
}

/* ---------------------------------------------------------------- compare */

/* Runs set side by side, each in its own colour. The list follows the run
   shown until a box is ticked by hand. */

let comparedKey = "";

function compareSlot() {
  const used = new Set(state.compare.values());
  for (let slot = 0; slot < COMPARE_AT_MOST; slot++) if (!used.has(slot)) return slot;
  return null;
}

function renderComparePick() {
  const known = new Set(state.runs.map((run) => run.run_id));
  if (!state.compareTouched) {
    state.compare = new Map(state.run ? [[state.run, 0]] : []);
  }
  for (const id of [...state.compare.keys()]) if (!known.has(id)) state.compare.delete(id);

  const full = state.compare.size >= COMPARE_AT_MOST;
  const markup = state.runs.map((run) => {
    const slot = state.compare.get(run.run_id);
    const on = slot != null;
    return `<label class="compare-run${on ? " on" : ""}">
        <input type="checkbox" value="${escape(run.run_id)}"${on ? " checked" : ""}${!on && full ? " disabled" : ""}>
        <span class="swatch${on ? ` c${slot}` : ""}"></span>
        <span class="compare-name"><span>${escape(run.run_id)}</span>
          <small>${escape(run.game)} · ${escape(run.label)}</small></span>
      </label>`;
  }).join("");
  const box = $("compare-pick");
  if (box.dataset.markup !== markup) {
    box.innerHTML = markup;
    box.dataset.markup = markup;
  }
}

function toggleCompare(event) {
  const input = event.target;
  if (input.type !== "checkbox") return;
  state.compareTouched = true;
  if (input.checked) {
    const slot = compareSlot();
    if (slot != null) state.compare.set(input.value, slot);
  } else {
    state.compare.delete(input.value);
  }
  refreshCompared(true);
}

/* Fetch the comparison again when the runs in it change, or while one of them
   is still training. */
async function refreshCompared(force = false) {
  renderComparePick();
  const ids = [...state.compare.keys()];
  const key = ids.join(",");
  const moving = state.runs.some((run) =>
    state.compare.has(run.run_id) && (run.live || writing(run)));
  if (!force && key === comparedKey && !moving) return;
  comparedKey = key;
  if (!ids.length) {
    state.compared = null;
    renderCompare();
    return;
  }
  try {
    const data = await getJSON(`/api/compare?runs=${encodeURIComponent(key)}`);
    if (key !== [...state.compare.keys()].join(",")) return;
    state.compared = data;
    renderCompare();
  } catch { /* shown again on the next refresh */ }
}

/* The best score so far at each generation, in the run's direction. */
function bestSoFar(values, direction) {
  const lower = lowerIsBetter(direction);
  let best = null;
  return values.map((v) => {
    if (v != null && (best == null || (lower ? v < best : v > best))) best = v;
    return best;
  });
}

function renderCompare() {
  const runs = state.compared ? state.compared.runs : [];
  const games = new Set(runs.map((run) => run.game));
  const directions = new Set(runs.map((run) => run.direction));
  const note = $("compare-note");
  if (games.size > 1) note.textContent = "different games, scores on different scales";
  else if (runs.length) note.textContent = `best so far · ${betterWord([...directions][0])}`;
  else note.textContent = "";

  const drawn = runs.filter((run) => run.curve.generation.length);
  $("compare-wrap").classList.toggle("has-data", drawn.length > 0);
  if (drawn.length) {
    plot($("compare-chart"), drawn.map((run) => ({
      xs: run.curve.generation,
      values: bestSoFar(run.curve.max_episode_score, run.direction),
      cls: `c${state.compare.get(run.run_id) ?? 0}`,
    })), { ticks: 4 });
  } else {
    $("compare-chart").innerHTML = "";
  }

  // The best of each game is marked, since scores of different games do not compare.
  const lead = new Map();
  for (const run of runs) {
    if (run.best == null) continue;
    const held = lead.get(run.game);
    const lower = lowerIsBetter(run.direction);
    if (held == null || (lower ? run.best < held : run.best > held)) lead.set(run.game, run.best);
  }

  const table = $("compare-table");
  if (!runs.length) {
    table.innerHTML = "";
    return;
  }
  const head = ["", "Run", "Settings", "Game", "Seed", "Gens", "Best", "At gen",
    "Recent mean", "Episodes", "Eps / s", "Time"];
  table.innerHTML = `
    <thead><tr>${head.map((h, i) => `<th${i > 4 ? ' class="n"' : ""}>${h}</th>`).join("")}</tr></thead>
    <tbody>${runs.map((run) => {
      const slot = state.compare.get(run.run_id) ?? 0;
      const leads = run.best != null && run.best === lead.get(run.game) && runs.length > 1;
      return `<tr data-run="${escape(run.run_id)}"${run.run_id === state.run ? ' class="current"' : ""}>
        <td><span class="swatch c${slot}"></span></td>
        <td class="mono">${escape(run.run_id)}${run.live ? ' <span class="tag ok">live</span>' : ""}</td>
        <td>${escape(run.label)}</td>
        <td>${escape(run.game)}</td>
        <td class="mono">${escape(run.seed ?? "–")}</td>
        <td class="n">${fmt.int(run.generations)}</td>
        <td class="n${leads ? " lead" : ""}">${fmt.num(run.best)}</td>
        <td class="n">${fmt.int(run.best_generation)}</td>
        <td class="n">${fmt.num(run.recent_mean)}</td>
        <td class="n">${fmt.int(run.episodes)}</td>
        <td class="n">${fmt.dec(run.episodes_per_second, 1)}</td>
        <td class="n">${fmt.duration(run.seconds)}</td>
      </tr>`;
    }).join("")}</tbody>`;
}

/* ---------------------------------------------------------------- navigation */

function syncScrubber() {
  const range = $("gen-range");
  const { first, last } = state.bounds;
  range.min = String(first);
  range.max = String(last);
  const shown = state.generation ? state.generation.generation : last;
  range.value = String(shown);
  fillRange(range);
  $("gen-jump").placeholder = `${first}–${last}`;
  $("btn-live").setAttribute("aria-pressed", String(state.following));
}

async function goTo(number) {
  if (!state.run) return;
  const { first, last } = state.bounds;
  const target = Math.max(first, Math.min(last, number));
  state.following = false;
  state.pinned = target;
  await loadGeneration(target);
  syncScrubber();
}

function follow() {
  if (!state.run) return;
  state.following = true;
  state.pinned = null;
  loadGeneration(null);
  syncScrubber();
}

/* The filled part of a range's track, which CSS cannot read from the value. */
function fillRange(input) {
  const min = Number(input.min) || 0;
  const max = Number(input.max) || 0;
  const share = max > min ? (Number(input.value) - min) / (max - min) : 0;
  input.style.setProperty("--fill", `${(share * 100).toFixed(2)}%`);
}

function setPlaying(on) {
  state.playing = on;
  const button = $("btn-play");
  button.dataset.playing = String(on);
  button.title = on ? "Pause" : "Play";
  button.setAttribute("aria-label", button.title);
}

/* ---------------------------------------------------------------- pickers */

/* Every select is shown as a button that opens a list drawn here, since a
   native list cannot be styled and looks different on every system. The
   select stays in the button's wrapper, out of sight, and still holds the
   value and fires the change the rest of the page listens for. An option can
   carry a note in data-note and a select a label in data-label. One list
   serves every picker, as only one is open at a time. */

const CHEVRON_ICON = '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 10l5 5 5-5"/></svg>';
const CHECK_ICON = '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M5.5 12.5l4 4 9-9"/></svg>';

const pickerButtons = new WeakMap();
const pickers = { list: null, open: null, active: -1 };

function picker(select) {
  const wrap = document.createElement("span");
  wrap.className = "picker";
  if (select.dataset.picker) wrap.classList.add(select.dataset.picker);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "picker-button";
  button.title = select.title;
  button.setAttribute("role", "combobox");
  button.setAttribute("aria-haspopup", "listbox");
  button.setAttribute("aria-expanded", "false");
  button.innerHTML = `<span class="picker-text"></span>${CHEVRON_ICON}`;
  select.before(wrap);
  wrap.append(button, select);
  select.classList.add("picker-native");
  select.tabIndex = -1;
  select.setAttribute("aria-hidden", "true");

  button.addEventListener("click", () => {
    if (pickers.open === select) closePicker();
    else openPicker(select);
  });
  button.addEventListener("keydown", (e) => pickerKey(e, select));
  pickerButtons.set(select, button);
  syncPicker(select);
}

/* Show the select's value on its button, after the page changed the select. */
function syncPicker(select) {
  const button = pickerButtons.get(select);
  if (!button) return;
  const option = select.options[select.selectedIndex];
  let text = "";
  if (select.dataset.label) text += `<span class="picker-label">${escape(select.dataset.label)}</span>`;
  text += `<span class="picker-main">${escape(option ? option.text : "")}</span>`;
  if (option && option.dataset.note) text += `<span class="picker-note">${escape(option.dataset.note)}</span>`;
  button.querySelector(".picker-text").innerHTML = text;
  button.disabled = select.disabled || !select.options.length;
  button.parentElement.hidden = select.hidden;
  if (pickers.open === select) {
    if (button.disabled || select.hidden) closePicker();
    else renderPickerList();
  }
}

function openPicker(select) {
  closePicker();
  if (!pickers.list) {
    const list = document.createElement("div");
    list.id = "picker-list";
    list.className = "picker-list";
    list.setAttribute("role", "listbox");
    list.hidden = true;
    // Pressing on the list would otherwise take the focus from the button.
    list.addEventListener("mousedown", (e) => e.preventDefault());
    list.addEventListener("click", (e) => {
      const option = e.target.closest(".picker-option");
      if (option) pick(Number(option.dataset.index));
    });
    list.addEventListener("pointermove", (e) => {
      const option = e.target.closest(".picker-option");
      if (option) setActive(Number(option.dataset.index), false);
    });
    document.body.appendChild(list);
    pickers.list = list;
  }
  const button = pickerButtons.get(select);
  pickers.open = select;
  button.setAttribute("aria-expanded", "true");
  button.setAttribute("aria-controls", pickers.list.id);
  pickers.list.hidden = false;
  renderPickerList();
  placePicker();
  setActive(select.selectedIndex, true);
}

function renderPickerList() {
  const select = pickers.open;
  pickers.list.classList.toggle("mono", select.dataset.picker === "mono");
  pickers.list.innerHTML = Array.from(select.options, (option, index) => {
    const note = option.dataset.note ? `<span class="picker-note">${escape(option.dataset.note)}</span>` : "";
    return `<div class="picker-option" role="option" id="picker-option-${index}" data-index="${index}"` +
      ` aria-selected="${index === select.selectedIndex}">${CHECK_ICON}` +
      `<span class="picker-main">${escape(option.text)}</span>${note}</div>`;
  }).join("");
  setActive(Math.min(pickers.active, select.options.length - 1), false);
}

/* Below the button, or above it when there is more room there, and never
   past the edge of the window. */
function placePicker() {
  const list = pickers.list;
  const box = pickerButtons.get(pickers.open).getBoundingClientRect();
  const margin = 8;
  const gap = 4;
  const below = window.innerHeight - box.bottom - margin - gap;
  const above = box.top - margin - gap;
  list.style.minWidth = `${box.width}px`;
  list.style.maxHeight = "";
  const wanted = Math.min(320, list.scrollHeight);
  const up = wanted > below && above > below;
  list.style.maxHeight = `${Math.min(320, up ? above : below)}px`;
  list.style.top = `${up ? box.top - gap - list.offsetHeight : box.bottom + gap}px`;
  const left = Math.min(box.left, window.innerWidth - list.offsetWidth - margin);
  list.style.left = `${Math.max(margin, left)}px`;
}

function pick(index) {
  const select = pickers.open;
  if (!select) return;
  closePicker();
  pickerButtons.get(select).focus();
  if (index < 0 || index === select.selectedIndex) return;
  select.selectedIndex = index;
  syncPicker(select);
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

function closePicker() {
  const select = pickers.open;
  if (!select) return;
  const button = pickerButtons.get(select);
  button.setAttribute("aria-expanded", "false");
  button.removeAttribute("aria-controls");
  button.removeAttribute("aria-activedescendant");
  pickers.list.hidden = true;
  pickers.list.innerHTML = "";
  pickers.open = null;
  pickers.active = -1;
}

function setActive(index, reveal) {
  const list = pickers.list;
  const button = pickerButtons.get(pickers.open);
  const options = list.children;
  if (options[pickers.active]) options[pickers.active].classList.remove("active");
  pickers.active = index;
  const option = options[index];
  if (!option) {
    button.removeAttribute("aria-activedescendant");
    return;
  }
  option.classList.add("active");
  button.setAttribute("aria-activedescendant", option.id);
  if (!reveal) return;
  if (option.offsetTop < list.scrollTop) list.scrollTop = option.offsetTop - gapOf(list);
  else if (option.offsetTop + option.offsetHeight > list.scrollTop + list.clientHeight) {
    list.scrollTop = option.offsetTop + option.offsetHeight - list.clientHeight + gapOf(list);
  }
}

function gapOf(list) {
  return parseFloat(getComputedStyle(list).paddingTop) || 0;
}

/* Arrows open a closed list, as they do a native select. Enter and Space on a
   closed one are the button's own click. */
function pickerKey(e, select) {
  const last = select.options.length - 1;
  let handled = true;
  if (pickers.open !== select) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") openPicker(select);
    else handled = false;
  } else if (e.key === "ArrowDown") setActive(Math.min(last, pickers.active + 1), true);
  else if (e.key === "ArrowUp") setActive(Math.max(0, pickers.active - 1), true);
  else if (e.key === "Home") setActive(0, true);
  else if (e.key === "End") setActive(last, true);
  else if (e.key === "Enter" || e.key === " ") pick(pickers.active);
  else if (e.key === "Escape") closePicker();
  else if (e.key === "Tab") { closePicker(); handled = false; }
  else handled = false;
  if (!handled) return;
  e.preventDefault();
  e.stopPropagation();
}

/* An open list closes on a press anywhere else, and when the page under it
   moves. */
function watchPickers() {
  document.addEventListener("pointerdown", (e) => {
    if (!pickers.open) return;
    if (pickers.list.contains(e.target) || pickerButtons.get(pickers.open).contains(e.target)) return;
    closePicker();
  }, true);
  document.addEventListener("scroll", (e) => {
    if (pickers.open && e.target !== pickers.list) closePicker();
  }, true);
  window.addEventListener("resize", closePicker);
}

/* ---------------------------------------------------------------- wiring */

function control(action, count) {
  postJSON("/api/control", { run: state.run, action, count })
    .then((result) => {
      if (result.error) toast(result.error, "bad");
      else scheduleRefresh();
    })
    .catch((error) => toast(`Could not reach the run: ${error.message}`, "bad"));
}

function bind() {
  $("run-select").addEventListener("change", (e) => show(e.target.value, true));
  $("btn-new").addEventListener("click", openDrawer);

  $("btn-continue").addEventListener("click", () => control("continue", 1));
  $("btn-continue-5").addEventListener("click", () => control("continue", 5));
  $("btn-hold").addEventListener("click", () => control("hold"));
  $("btn-free").addEventListener("click", () => control("run_free"));
  $("btn-stop").addEventListener("click", stopRun);

  const chart = $("chart");
  chart.addEventListener("pointermove", chartHover);
  chart.addEventListener("pointerleave", hideReadout);
  chart.addEventListener("click", chartClick);

  $("gen-first").addEventListener("click", () => goTo(state.bounds.first));
  $("gen-prev").addEventListener("click", () => goTo((state.generation?.generation ?? 0) - 1));
  $("gen-next").addEventListener("click", () => goTo((state.generation?.generation ?? 0) + 1));
  $("gen-last").addEventListener("click", () => goTo(state.bounds.last));
  $("btn-live").addEventListener("click", follow);
  $("gen-range").addEventListener("input", (e) => fillRange(e.target));
  $("gen-range").addEventListener("change", (e) => goTo(Number(e.target.value)));
  $("gen-jump").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { goTo(Number(e.target.value)); e.target.value = ""; }
  });

  $("btn-play").addEventListener("click", () => setPlaying(!state.playing));
  $("btn-restart").addEventListener("click", () => { state.frame = 0; paintFrame(); });
  $("frame-range").addEventListener("input", (e) => {
    setPlaying(false);
    state.frame = Number(e.target.value);
    paintFrame();
  });
  $("speed").addEventListener("change", (e) => { state.speed = Number(e.target.value); });
  $("sort").addEventListener("change", (e) => { state.sort = e.target.value; renderGrid(); });
  $("limit").addEventListener("change", (e) => { state.limit = Number(e.target.value); renderGrid(); });

  for (const input of document.querySelectorAll('input[name="pick"]')) {
    input.addEventListener("change", () => setPick(input.value));
  }
  $("btn-preview").addEventListener("click", previewExport);
  $("btn-choose").addEventListener("click", chooseByAverage);
  $("job-cancel").addEventListener("click", cancelJob);
  $("btn-export").addEventListener("click", doExport);
  $("sheet-close").addEventListener("click", closeSheet);

  $("drawer-close").addEventListener("click", closeDrawer);
  $("new-game").addEventListener("change", (e) => loadSettings(e.target.value));
  $("new-reset").addEventListener("click", resetSettings);
  $("new-run").addEventListener("submit", startRun);

  $("compare-pick").addEventListener("change", toggleCompare);
  $("compare-table").addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-run]");
    if (row && row.dataset.run !== state.run) show(row.dataset.run, true);
  });

  if (window.ResizeObserver) {
    let pending = false;
    const redraw = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        renderChart();
        renderCompare();
      });
    };
    const observer = new ResizeObserver(redraw);
    observer.observe($("chart-wrap"));
    observer.observe($("compare-wrap"));
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!$("drawer").hidden) closeDrawer();
      else closeSheet();
      return;
    }
    const tag = e.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (e.target.closest("#drawer, .picker")) return;
    if (e.key === "ArrowLeft") goTo((state.generation?.generation ?? 0) - 1);
    else if (e.key === "ArrowRight") goTo((state.generation?.generation ?? 0) + 1);
    else if (e.key === " ") { e.preventDefault(); $("btn-play").click(); }
    else if (e.key.toLowerCase() === "c") $("btn-continue").click();
    else if (e.key.toLowerCase() === "l") follow();
  });
}

/* ---------------------------------------------------------------- events */

let refreshTimer = null;

function scheduleRefresh() {
  // A fast run can finish several generations a second; one refresh covers
  // all of them.
  if (refreshTimer) return;
  refreshTimer = setTimeout(() => {
    refreshTimer = null;
    pollStatus();
    renderChart();
  }, 150);
}

let runsTimer = null;

function runsSoon() {
  clearTimeout(runsTimer);
  runsTimer = setTimeout(loadRuns, 300);
}

function listen() {
  if (!window.EventSource) return;
  const source = new EventSource("/api/events");
  // `mine` is whether the event is about the run shown. Events from before
  // they carried a run are taken as the run shown.
  const on = (kind, handler) =>
    source.addEventListener(kind, (e) => {
      const event = JSON.parse(e.data);
      handler(event, event.run == null || event.run === state.run);
    });

  source.addEventListener("open", () => { state.live = true; });
  source.addEventListener("error", () => { state.live = false; });

  on("generation", (event, mine) => {
    if (!mine) return;
    extendCurve(event.summary);
    scheduleRefresh();
  });
  on("record", (event, mine) => {
    if (mine) toast(`New best score ${fmt.num(event.score)} at generation ${event.generation}`, "good");
  });
  on("held", (event, mine) => { if (mine) scheduleRefresh(); });
  on("resumed", (event, mine) => { if (mine) scheduleRefresh(); });
  on("note", (event, mine) => { if (mine) toast(event.message); });
  on("run_started", runsSoon);
  on("run_finished", (event, mine) => {
    runsSoon();
    if (!mine) return;
    scheduleRefresh();
    if (event.error) toast(`Training stopped: ${event.error}`, "bad");
  });
  // Events were missed, so rebuild from the regular endpoints.
  on("resync", () => { loadCurve(); scheduleRefresh(); runsSoon(); });
}

function statusLoop() {
  const status = state.status;
  const saved = status != null && !status.live;
  // A run read from disk that has not been written to for a while is not
  // going to change, and the list of runs notices when it does.
  const entry = state.runs.find((run) => run.run_id === state.run);
  const settled = saved && entry != null && !entry.live && !writing(entry);
  const work = settled ? Promise.resolve() : pollStatus();
  const delay = !state.run || saved ? SAVED_MS : state.live ? STATUS_LIVE_MS : STATUS_MS;
  work.finally(() => setTimeout(statusLoop, delay));
}

function heartbeat() {
  // Presence lease. The trainer holds only while these keep arriving, so a
  // closed tab, or one showing another run, releases the hold instead of
  // stalling the run indefinitely.
  if (!state.run || state.run !== state.liveRun) return;
  postJSON("/api/heartbeat").catch(() => {});
}

function init() {
  state.wanted = new URLSearchParams(window.location.search).get("run");
  state.exportHint = $("export-report").innerHTML;
  for (const select of document.querySelectorAll("select")) picker(select);
  watchPickers();
  bind();
  setPick(state.exportPick);
  listen();
  loadRuns().finally(statusLoop);
  setInterval(loadRuns, RUNS_MS);
  setInterval(heartbeat, HEARTBEAT_MS);
  requestAnimationFrame(tick);
}

document.addEventListener("DOMContentLoaded", init);
