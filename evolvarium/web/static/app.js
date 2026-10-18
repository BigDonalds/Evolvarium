/* evolvarium dashboard
   Vanilla ES modules-free script: no build step, no dependencies.

   Three behaviours drive most of the design.

   Presence: the trainer holds at a generation boundary while this page is open,
   so a generation cannot be replaced before it has been looked at. That hold is
   a lease, refreshed by the heartbeat below; if this page stops sending them the
   trainer resumes on its own rather than waiting forever.

   Pinning: selecting a generation detaches the view from the live edge. New
   generations keep arriving and the scrubber range grows, but the displayed
   generation does not change until "Follow latest" is turned back on.

   Events: the trainer announces each generation, record, hold and note on
   /api/events as it happens, and the page reacts to those. Status is still
   polled, slowly, for the clocks and rates that change all the time, and at the
   old rate whenever the stream is down. */

const SVG_NS = "http://www.w3.org/2000/svg";
const HEARTBEAT_MS = 3000;
const STATUS_MS = 1200;        // polling while the event stream is down
const STATUS_LIVE_MS = 5000;   // polling while events arrive on their own
const TICK_MS = 90;

const state = {
  status: null,
  live: false,           // event stream connected
  curve: null,           // chart series, extended as generations arrive
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
  exportReport: null,
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

const fmt = {
  int: (n) => (n == null ? "–" : Math.round(n).toLocaleString()),
  dec: (n, d = 2) => (n == null ? "–" : Number(n).toFixed(d)),
  pct: (n) => (n == null ? "–" : `${(n * 100).toFixed(1)}%`),
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
};

/* ---------------------------------------------------------------- status */

async function pollStatus() {
  try {
    const status = await getJSON("/api/status");
    state.status = status;
    renderStatus(status);
    renderRunDetail(status);

    const gens = await getJSON("/api/generations");
    state.bounds = gens;
    syncScrubber();

    if (state.following && gens.last !== (state.generation && state.generation.generation)) {
      await loadGeneration(null);
    }
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
  $("run-id").textContent = status.run_id;
  $("m-gen").textContent = fmt.int(status.generation);
  $("m-best").textContent = fmt.int(status.best_ever_score);
  $("m-eps").textContent = fmt.int(status.total_episodes);
  $("m-rate").textContent = fmt.dec(status.generations_per_minute, 1);

  const last = status.last_generation;
  $("m-mean").textContent = last ? fmt.dec(last.mean_score, 3) : "–";
  $("m-eat").textContent = last ? fmt.pct(last.scoring_fraction) : "–";

  const gate = status.gate || {};
  const holdBtn = $("btn-hold");
  holdBtn.setAttribute("aria-pressed", String(!!gate.hold_enabled));

  if (gate.holding) {
    setStateChip("held", `held at gen ${gate.generation}`);
    $("hold-status").className = "hold-status held";
    $("hold-status").textContent =
      `Paused at generation ${gate.generation} for ${fmt.duration(gate.held_for)}. ` +
      `Nothing advances until you continue.`;
  } else if (!status.running) {
    setStateChip("offline", "stopped");
    $("hold-status").className = "hold-status";
    $("hold-status").textContent = "Training has stopped.";
  } else {
    setStateChip("running", gate.hold_enabled ? "running" : "running free");
    $("hold-status").className = "hold-status";
    $("hold-status").textContent = gate.hold_enabled
      ? (gate.credits > 0
          ? `Running with ${gate.credits} generation${gate.credits === 1 ? "" : "s"} of credit.`
          : "Running. Will pause at the next generation boundary.")
      : "Running freely. Generations will advance without waiting.";
  }
}

function renderRunDetail(status) {
  const rows = [
    ["Run", status.run_id],
    ["Game", status.game],
    ["Encoding", `${status.encoding} · ${status.input_size} inputs`],
    ["Genome", `${fmt.int(status.genome_size)} params`],
    ["Scene", sceneText(status.scene)],
    ["Population", fmt.int(status.population_size)],
    ["Episodes", fmt.int(status.total_episodes)],
    ["Episodes / s", fmt.dec(status.episodes_per_second, 1)],
    ["Runtime", fmt.duration(status.elapsed_seconds)],
    ["Best at gen", fmt.int(status.best_ever_generation)],
    ["Stagnation", `${fmt.int(status.stagnation)} gens`],
  ];
  $("run-detail").innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${escape(String(v))}</dd>`)
    .join("");
}

function escape(text) {
  return text.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------------------------------------------------------- chart */

async function loadCurve() {
  try {
    state.curve = await getJSON("/api/curve");
    renderChart(state.curve);
  } catch { /* chart is optional */ }
}

function extendCurve(summary) {
  const curve = state.curve;
  if (!curve) return;
  const n = curve.generation.length;
  const last = n ? curve.generation[n - 1] : null;
  if (last != null && summary.generation <= last) return;
  if (last != null && summary.generation > last + 1) {
    // A generation finished between loading the curve and the stream opening.
    loadCurve();
    return;
  }
  for (const key of Object.keys(curve)) curve[key].push(summary[key] ?? 0);
}

function renderChart(curve) {
  const svg = $("chart");
  const wrap = svg.parentElement;
  const n = curve.generation.length;
  if (!n) { wrap.classList.remove("has-data"); return; }
  wrap.classList.add("has-data");

  const W = 1000, H = 220, padL = 34, padR = 12, padT = 12, padB = 20;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const best = curve.max_episode_score;
  const mean = curve.mean_score;
  const feed = curve.scoring_fraction;
  const maxScore = Math.max(1, ...best);

  const xAt = (i) => padL + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const yScore = (v) => padT + innerH - (v / maxScore) * innerH;
  const yPct = (v) => padT + innerH - v * innerH;

  const path = (values, mapper) =>
    values.map((v, i) => `${i ? "L" : "M"}${xAt(i).toFixed(1)},${mapper(v).toFixed(1)}`).join("");

  const ticks = 4;
  let grid = "";
  for (let i = 0; i <= ticks; i++) {
    const value = (maxScore / ticks) * i;
    const y = yScore(value);
    grid += `<line class="grid-line" x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}"/>`;
    grid += `<text class="axis-label" x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end">${
      maxScore <= 8 ? value.toFixed(0) : Math.round(value)}</text>`;
  }

  let marker = "";
  const shown = state.generation && state.generation.generation;
  if (shown != null) {
    const index = curve.generation.indexOf(shown);
    if (index >= 0) {
      const x = xAt(index).toFixed(1);
      marker = `<line class="marker" x1="${x}" y1="${padT}" x2="${x}" y2="${padT + innerH}"/>`;
    }
  }

  const area = `${path(best, yScore)}L${xAt(n - 1).toFixed(1)},${(padT + innerH).toFixed(1)}L${xAt(0).toFixed(1)},${(padT + innerH).toFixed(1)}Z`;

  svg.innerHTML = `
    <defs>
      <linearGradient id="bestFade" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#4ade80" stop-opacity="0.22"/>
        <stop offset="100%" stop-color="#4ade80" stop-opacity="0"/>
      </linearGradient>
    </defs>
    ${grid}
    <path class="series-area" d="${area}"/>
    <path class="series feed" d="${path(feed, yPct)}"/>
    <path class="series mean" d="${path(mean, yScore)}"/>
    <path class="series best" d="${path(best, yScore)}"/>
    ${marker}
    <text class="axis-label" x="${padL}" y="${H - 5}">gen ${curve.generation[0]}</text>
    <text class="axis-label" x="${W - padR}" y="${H - 5}" text-anchor="end">gen ${curve.generation[n - 1]}</text>
  `;
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
  const query = number == null ? "" : `?n=${number}`;
  let data;
  try {
    data = await getJSON(`/api/generation${query}`);
  } catch {
    return;
  }
  if (data.error) {
    toast(`Generation ${number} is no longer retained`, "bad");
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
  if (state.curve) renderChart(state.curve);
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
    ["Best score", summary.max_episode_score, summary.record ? "record" : ""],
    ["Mean score", fmt.dec(summary.mean_score, 3), ""],
    ["Scored", fmt.pct(summary.scoring_fraction), ""],
    ["Best fitness", fmt.int(summary.best_fitness), ""],
    ["Mean steps", fmt.dec(summary.mean_steps, 1), ""],
    ["Diversity", fmt.dec(summary.diversity, 3), ""],
    ["Mutation", fmt.dec(summary.mutation_rate, 3), ""],
    ["Ends", endText, ""],
    ["Wall time", fmt.duration(summary.elapsed), ""],
  ];
  $("gen-summary").innerHTML = items
    .map(([k, v, cls]) => `<div class="item ${cls}"><span>${k}</span><span>${escape(String(v))}</span></div>`)
    .join("");
}

/* ---------------------------------------------------------------- grid */

function sortedAgents() {
  const key = state.sort;
  const copy = state.agents.slice();
  copy.sort((a, b) => {
    if (key === "score") return b.score - a.score || b.fitness - a.fitness;
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
      `<span class="sc">${agent.best_episode_score}</span>` +
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
      <dt>Mean score</dt><dd>${fmt.dec(agent.score, 3)}</dd>
      <dt>Best episode</dt><dd>${agent.best_episode_score}</dd>
      <dt>Mean steps</dt><dd>${fmt.dec(agent.steps, 1)}</dd>
      <dt>Ended by</dt><dd>${escape(String(agent.end ?? agent.death))}</dd>
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

async function previewExport() {
  const raw = $("export-gen").value.trim();
  const query = raw === "" ? "" : `?n=${encodeURIComponent(raw)}`;
  $("btn-preview").disabled = true;
  try {
    const report = await getJSON(`/api/export/preview${query}`);
    state.exportReport = report;
    renderExportReport(report);
    $("btn-export").disabled = !report.ready;
  } catch (error) {
    toast(`Could not calculate: ${error.message}`, "bad");
  } finally {
    $("btn-preview").disabled = false;
  }
}

function renderExportReport(report) {
  const t = report.training || {};
  const availability = $("export-availability");
  if (report.torch_available) {
    availability.textContent = ".json and .pt";
    availability.className = "tag ok";
  } else {
    availability.textContent = ".json";
    availability.className = "tag";
  }

  const cards = [
    ["Generation", report.resolved_generation],
    ["Parameters", fmt.int(report.parameter_count)],
    ["Est. size", fmt.bytes(report.estimated_size_bytes)],
    ["Best score", fmt.int(t.best_score)],
    ["At generation", fmt.int(t.best_generation)],
    ["Generations", fmt.int(t.generations)],
    ["Episodes", fmt.int(t.episodes)],
    ["Compute time", fmt.duration(t.compute_seconds)],
    ["Records set", fmt.int(t.records_set)],
    ["Final mean", fmt.dec(t.mean_score_final, 3)],
    ["Final scored", fmt.pct(t.scoring_fraction_final)],
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

  let notice = "";
  if (!report.genome_available) {
    notice = `<p class="hint" style="color:var(--warn)">No champion genome is archived at or before generation ${report.requested_generation}.</p>`;
  } else if (!report.torch_available) {
    notice = `<p class="hint">Only the JSON model is written. <code>pip install torch</code> adds a .pt copy beside it.</p>`;
  }

  $("export-report").innerHTML = `
    <div class="report-grid">
      ${cards.map(([k, v]) => `<div class="report-item"><span class="k">${k}</span><span class="v">${escape(String(v))}</span></div>`).join("")}
    </div>
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
  const report = state.exportReport;
  if (!report) return;
  $("btn-export").disabled = true;
  try {
    const result = await postJSON("/api/export", { generation: report.resolved_generation });
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

/* ---------------------------------------------------------------- navigation */

function syncScrubber() {
  const range = $("gen-range");
  const { first, last } = state.bounds;
  range.min = String(first);
  range.max = String(last);
  const shown = state.generation ? state.generation.generation : last;
  range.value = String(shown);
  $("gen-jump").placeholder = `${first}–${last}`;
  $("btn-live").setAttribute("aria-pressed", String(state.following));
}

async function goTo(number) {
  const { first, last } = state.bounds;
  const target = Math.max(first, Math.min(last, number));
  state.following = false;
  state.pinned = target;
  await loadGeneration(target);
  syncScrubber();
}

function follow() {
  state.following = true;
  state.pinned = null;
  loadGeneration(null);
  syncScrubber();
}

/* ---------------------------------------------------------------- wiring */

function bind() {
  $("btn-continue").addEventListener("click", () => postJSON("/api/control", { action: "continue", count: 1 }));
  $("btn-continue-5").addEventListener("click", () => postJSON("/api/control", { action: "continue", count: 5 }));
  $("btn-hold").addEventListener("click", () => postJSON("/api/control", { action: "hold" }));
  $("btn-free").addEventListener("click", () => postJSON("/api/control", { action: "run_free" }));

  $("gen-first").addEventListener("click", () => goTo(state.bounds.first));
  $("gen-prev").addEventListener("click", () => goTo((state.generation?.generation ?? 0) - 1));
  $("gen-next").addEventListener("click", () => goTo((state.generation?.generation ?? 0) + 1));
  $("gen-last").addEventListener("click", () => goTo(state.bounds.last));
  $("btn-live").addEventListener("click", follow);
  $("gen-range").addEventListener("change", (e) => goTo(Number(e.target.value)));
  $("gen-jump").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { goTo(Number(e.target.value)); e.target.value = ""; }
  });

  $("btn-play").addEventListener("click", () => {
    state.playing = !state.playing;
    $("btn-play").textContent = state.playing ? "⏸" : "▶";
  });
  $("btn-restart").addEventListener("click", () => { state.frame = 0; paintFrame(); });
  $("frame-range").addEventListener("input", (e) => {
    state.playing = false;
    $("btn-play").textContent = "▶";
    state.frame = Number(e.target.value);
    paintFrame();
  });
  $("speed").addEventListener("change", (e) => { state.speed = Number(e.target.value); });
  $("sort").addEventListener("change", (e) => { state.sort = e.target.value; renderGrid(); });
  $("limit").addEventListener("change", (e) => { state.limit = Number(e.target.value); renderGrid(); });

  $("btn-preview").addEventListener("click", previewExport);
  $("btn-export").addEventListener("click", doExport);
  $("sheet-close").addEventListener("click", closeSheet);

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") goTo((state.generation?.generation ?? 0) - 1);
    else if (e.key === "ArrowRight") goTo((state.generation?.generation ?? 0) + 1);
    else if (e.key === " ") { e.preventDefault(); $("btn-play").click(); }
    else if (e.key === "Escape") closeSheet();
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
    if (state.curve) renderChart(state.curve);
  }, 150);
}

function listen() {
  if (!window.EventSource) return;
  const source = new EventSource("/api/events");
  const on = (kind, handler) =>
    source.addEventListener(kind, (e) => handler(JSON.parse(e.data)));

  source.addEventListener("open", () => { state.live = true; });
  source.addEventListener("error", () => { state.live = false; });

  on("generation", (event) => { extendCurve(event.summary); scheduleRefresh(); });
  on("record", (event) =>
    toast(`New best score ${event.score} at generation ${event.generation}`, "good"));
  on("held", scheduleRefresh);
  on("resumed", scheduleRefresh);
  on("note", (event) => toast(event.message));
  on("run_finished", (event) => {
    scheduleRefresh();
    if (event.error) toast(`Training stopped: ${event.error}`, "bad");
  });
  // Events were missed, so rebuild from the regular endpoints.
  on("resync", () => { loadCurve(); scheduleRefresh(); });
}

function statusLoop() {
  pollStatus().finally(() => setTimeout(statusLoop, state.live ? STATUS_LIVE_MS : STATUS_MS));
}

function heartbeat() {
  // Presence lease. The trainer holds only while these keep arriving, so a
  // closed tab releases the hold instead of stalling the run indefinitely.
  postJSON("/api/heartbeat").catch(() => {});
}

function init() {
  bind();
  heartbeat();
  listen();
  loadCurve();
  statusLoop();
  setInterval(heartbeat, HEARTBEAT_MS);
  requestAnimationFrame(tick);
}

document.addEventListener("DOMContentLoaded", init);
