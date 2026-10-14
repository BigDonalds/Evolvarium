/* evolvarium dashboard
   Vanilla ES modules-free script: no build step, no dependencies.

   Two behaviours drive most of the design.

   Presence: the trainer holds at a generation boundary while this page is open,
   so a generation cannot be replaced before it has been looked at. That hold is
   a lease, refreshed by the heartbeat below; if this page stops sending them the
   trainer resumes on its own rather than waiting forever.

   Pinning: selecting a generation detaches the view from the live edge. New
   generations keep arriving and the scrubber range grows, but the displayed
   generation does not change until "Follow latest" is turned back on. */

const SVG_NS = "http://www.w3.org/2000/svg";
const HEARTBEAT_MS = 3000;
const STATUS_MS = 1200;
const TICK_MS = 90;

const state = {
  status: null,
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
  board: { width: 14, height: 14 },
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
    ["Encoding", `${status.feature_spec} · ${status.input_size} inputs`],
    ["Genome", `${fmt.int(status.genome_size)} params`],
    ["Board", `${status.board.width} × ${status.board.height}`],
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
    const curve = await getJSON("/api/curve");
    renderChart(curve);
  } catch { /* chart is optional */ }
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

const STEP_DELTAS = [[0, -1], [1, 0], [0, 1], [-1, 0]];

/* Rebuild an agent's episode from the compact encoding the server sends: a
   starting cell, one direction byte per step, and sparse lists of food changes
   and growth points. Expanding once on load keeps the per-frame paint cheap. */
function expandReplay(replay) {
  if (!replay || !replay.steps) return { path: [], food: [], length: [], steps: 0 };

  const bytes = atob(replay.dirs || "");
  const path = new Array(replay.steps);
  let [x, y] = replay.start;
  path[0] = [x, y];
  for (let i = 0; i < bytes.length && i + 1 < replay.steps; i++) {
    const [dx, dy] = STEP_DELTAS[bytes.charCodeAt(i) & 3];
    x += dx; y += dy;
    path[i + 1] = [x, y];
  }

  // Forward-fill food position and body length across every step.
  const food = new Array(replay.steps);
  let cursor = 0;
  let current = replay.food.length ? [replay.food[0][1], replay.food[0][2]] : [0, 0];
  for (let i = 0; i < replay.steps; i++) {
    while (cursor < replay.food.length && replay.food[cursor][0] <= i) {
      current = [replay.food[cursor][1], replay.food[cursor][2]];
      cursor++;
    }
    food[i] = current;
  }

  const length = new Array(replay.steps);
  let grown = 0;
  let size = replay.start_length || 3;
  for (let i = 0; i < replay.steps; i++) {
    while (grown < replay.grow.length && replay.grow[grown] <= i) { size++; grown++; }
    length[i] = size;
  }

  return { path, food, length, steps: replay.steps };
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
  state.board = data.board;
  state.agents = data.agents || [];
  for (const agent of state.agents) agent.episode = expandReplay(agent.replay);
  state.maxFrames = state.agents.reduce((m, a) => Math.max(m, a.episode.steps), 0);
  state.frame = 0;
  $("frame-range").max = String(Math.max(0, state.maxFrames - 1));

  renderGenerationSummary(data.summary);
  renderGrid();
  syncScrubber();
  loadCurve();
}

function renderGenerationSummary(summary) {
  if (!summary) { $("gen-summary").innerHTML = ""; return; }
  const deaths = summary.deaths || {};
  const deathText = Object.entries(deaths)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k} ${v}`)
    .join(" · ") || "–";

  const items = [
    ["Generation", summary.generation, summary.record ? "record" : ""],
    ["Best score", summary.max_episode_score, summary.record ? "record" : ""],
    ["Mean score", fmt.dec(summary.mean_score, 3), ""],
    ["Feeding", fmt.pct(summary.scoring_fraction), ""],
    ["Best fitness", fmt.int(summary.best_fitness), ""],
    ["Mean steps", fmt.dec(summary.mean_steps, 1), ""],
    ["Diversity", fmt.dec(summary.diversity, 3), ""],
    ["Mutation", fmt.dec(summary.mutation_rate, 3), ""],
    ["Deaths", deathText, ""],
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

function renderGrid() {
  const grid = $("grid");
  grid.innerHTML = "";
  const agents = sortedAgents();
  if (!agents.length) return;

  const { width, height } = state.board;
  const fragment = document.createDocumentFragment();

  agents.forEach((agent, rank) => {
    const card = document.createElement("div");
    card.className = `agent${rank === 0 ? " top" : ""}`;
    card.dataset.id = agent.id;
    card.tabIndex = 0;

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
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

function paintFrame() {
  const agents = sortedAgents();
  const { width, height } = state.board;
  const frame = state.frame;

  for (const agent of agents) {
    const svg = agent._svg;
    if (!svg) continue;
    const episode = agent.episode;
    if (!episode || !episode.steps) { svg.innerHTML = ""; continue; }

    const index = Math.min(frame, episode.steps - 1);
    const [hx, hy] = episode.path[index];
    const [fx, fy] = episode.food[index];
    const bodyLength = Math.min(episode.length[index], index + 1);

    let markup = `<rect x="0" y="0" width="${width}" height="${height}" fill="#080b11"/>`;
    markup += `<rect class="cell-food" x="${fx + 0.12}" y="${fy + 0.12}" width="0.76" height="0.76" rx="0.38"/>`;

    // Every step is stored, so the preceding `bodyLength` head positions are
    // exactly the cells the body occupied at this moment.
    for (let i = index - bodyLength + 1; i < index; i++) {
      if (i < 0) continue;
      const cell = episode.path[i];
      if (!cell) continue;
      const age = (i - (index - bodyLength)) / bodyLength;
      markup += `<rect class="cell-body" x="${cell[0] + 0.06}" y="${cell[1] + 0.06}" width="0.88" height="0.88" rx="0.14" opacity="${(0.4 + age * 0.55).toFixed(2)}"/>`;
    }
    markup += `<rect class="cell-head" x="${hx + 0.03}" y="${hy + 0.03}" width="0.94" height="0.94" rx="0.18"/>`;

    // A replay that was cut short ends before the game did, so its last frame
    // is not a death and must not be drawn as one.
    const ended = index >= episode.steps - 1;
    const cut = Boolean(agent.replay && agent.replay.truncated);
    svg.parentElement.classList.toggle("dead", ended && !cut);
    svg.parentElement.classList.toggle("cut", ended && cut);
    svg.innerHTML = markup;
  }

  $("frame-range").value = String(frame);
  $("frame-label").textContent = `${frame} / ${Math.max(0, state.maxFrames - 1)}`;
}

function tick(now) {
  if (state.playing && state.maxFrames > 0) {
    if (!state.lastTick) state.lastTick = now;
    state.tickAccumulator += now - state.lastTick;
    const interval = TICK_MS * state.speed;
    while (state.tickAccumulator >= interval) {
      state.tickAccumulator -= interval;
      state.frame = (state.frame + 1) % state.maxFrames;
    }
    paintFrame();
  }
  state.lastTick = now;
  requestAnimationFrame(tick);
}

/* ---------------------------------------------------------------- agent sheet */

function showAgent(agent) {
  $("agent-sheet").hidden = false;
  $("sheet-title").textContent = agent.id;

  const terms = agent.terms || {};
  const termRows = Object.entries(terms)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([k, v]) => `<dt>${escape(k)}</dt><dd>${fmt.dec(v, 1)}</dd>`)
    .join("");

  $("sheet-body").innerHTML = `
    <dl class="detail">
      <dt>Rank origin</dt><dd>${escape(agent.origin)}</dd>
      <dt>Generation</dt><dd>${agent.generation}</dd>
      <dt>Lineage age</dt><dd>${agent.age}</dd>
      <dt>Fitness</dt><dd>${fmt.dec(agent.fitness, 1)}</dd>
      <dt>Mean score</dt><dd>${fmt.dec(agent.score, 3)}</dd>
      <dt>Best episode</dt><dd>${agent.best_episode_score}</dd>
      <dt>Mean steps</dt><dd>${fmt.dec(agent.steps, 1)}</dd>
      <dt>Death</dt><dd>${escape(String(agent.death))}</dd>
      <dt>Replay steps</dt><dd>${agent.episode ? agent.episode.steps : 0}${agent.replay && agent.replay.truncated ? " (truncated)" : ""}</dd>
    </dl>
    <p class="section-label">Fitness terms</p>
    <dl class="detail">${termRows}</dl>
  `;
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
    ["Final feeding", fmt.pct(t.scoring_fraction_final)],
    ["Achievements", `${report.achievements_unlocked} / ${report.achievements_total}`],
  ];

  const achievements = (report.achievements || [])
    .map((a) => `<span class="ach ${a.unlocked ? "on" : ""}" title="${escape(a.detail)}">
        ${escape(a.label)}${a.unlocked ? `<span class="gen">gen ${a.generation}</span>` : ""}
      </span>`)
    .join("");

  const deaths = Object.entries(t.deaths || {})
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
    <p class="section-label">Deaths across the run</p>
    <p class="hint">${escape(deaths || "–")}</p>
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
  $("sheet-close").addEventListener("click", () => { $("agent-sheet").hidden = true; });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") goTo((state.generation?.generation ?? 0) - 1);
    else if (e.key === "ArrowRight") goTo((state.generation?.generation ?? 0) + 1);
    else if (e.key === " ") { e.preventDefault(); $("btn-play").click(); }
    else if (e.key === "Escape") $("agent-sheet").hidden = true;
    else if (e.key.toLowerCase() === "c") $("btn-continue").click();
    else if (e.key.toLowerCase() === "l") follow();
  });
}

function heartbeat() {
  // Presence lease. The trainer holds only while these keep arriving, so a
  // closed tab releases the hold instead of stalling the run indefinitely.
  postJSON("/api/heartbeat").catch(() => {});
}

function init() {
  bind();
  heartbeat();
  pollStatus();
  loadCurve();
  setInterval(heartbeat, HEARTBEAT_MS);
  setInterval(pollStatus, STATUS_MS);
  requestAnimationFrame(tick);
}

document.addEventListener("DOMContentLoaded", init);
