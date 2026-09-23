/*
 * The bluffing explorer.  One data file per card P2 holds carries the public
 * game graph and, at every node, the set of results P2 can guarantee (vectors
 * over P1's card).  The page follows a line with P2's belief p and the
 * guarantee y it is committed to, and asks BluffCore for each side's mix.
 */
"use strict";

const Core = window.BluffCore;
const CARDS = ["D", "S", "C", "T"];
const NAME = { D: "Duke", S: "Assassin", C: "Captain", T: "Contessa" };
// the card each P1 claim asserts
const CLAIMS = { "Tax": "D", "Steal": "C", "Assassinate": "S",
                 "block Duke": "D", "block Captain": "C", "block Contessa": "T" };
const EPS = 1e-9;

const $ = (id) => document.getElementById(id);
const pct = (x) => `${(100 * x).toFixed(Math.abs(x - Math.round(x)) < 1e-9 ? 0 : 1)}%`;
const prob = (x) => (Math.abs(x) < 5e-4 ? 0 : x).toFixed(3);

const shards = {};
let data = null;      // the shard in use
let setup = null;     // {p1, p2, up:[..], p1Index}
let line = [];        // [{i, p, y, text, auto, over}]

/* ---- loading ----------------------------------------------------------- */

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url} (${res.status})`);
  const buf = await res.arrayBuffer();
  const bytes = new Uint8Array(buf);
  // A host may or may not have decompressed this for us: 1f 8b is gzip's magic.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (typeof DecompressionStream === "undefined") {
      throw new Error("this browser cannot decompress the data (needs DecompressionStream)");
    }
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function shard(card) {
  if (!shards[card]) shards[card] = fetchJson(`data/bluff/${card}.json.gz`);
  return shards[card];
}

/* ---- setup ------------------------------------------------------------- */

function fillSelects() {
  const defaults = { c0: "S", c1: "D", d0: "D", d1: "S" };
  for (const id of ["c0", "c1", "d0", "d1"]) {
    $(id).innerHTML = CARDS.map((c) => `<option value="${c}">${NAME[c]}</option>`).join("");
    $(id).value = defaults[id];
  }
}

function readSetup() {
  const p1 = $("c0").value, p2 = $("c1").value;
  const up = [$("d0").value, $("d1").value];
  const coins = [Number($("n0").value), Number($("n1").value)];
  for (const n of coins) {
    if (!Number.isInteger(n) || n < 0 || n > 12) throw new Error("coins must be whole numbers from 0 to 12");
  }
  for (const c of CARDS) {
    const copies = [p1, p2, ...up].filter((x) => x === c).length;
    if (copies > 3) throw new Error(`only three ${NAME[c]}s exist`);
  }
  // P2 knows its own card and the upcards; every other copy could be P1's.
  const unseen = CARDS.map((c) => 3 - [p2, ...up].filter((x) => x === c).length);
  const total = unseen.reduce((s, x) => s + x, 0);
  const prior = unseen.map((x) => x / total);
  return { p1, p2, up, coins, first: Number($("first").value),
           prior, t: CARDS.indexOf(p1) };
}

async function begin() {
  $("err").textContent = "";
  let s;
  try { s = readSetup(); } catch (e) { $("err").textContent = e.message; return; }
  $("start").disabled = true;
  try {
    data = await shard(s.p2);
  } catch (e) {
    $("err").textContent = e.message; $("start").disabled = false; return;
  } finally {
    $("start").disabled = false;
  }
  setup = s;
  const i = data.starts[`${s.coins[0]}.${s.coins[1]}.${s.first}`];
  line = [{ i, p: s.prior, y: Core.argminVertex(data.sets[i], s.prior), text: null }];
  settle();
  $("game").hidden = false;
  $("histPanel").hidden = false;
  render();
}

/* ---- walking the line -------------------------------------------------- */

const here = () => line[line.length - 1];
const node = (i) => data.nodes[i];

/** Take forced moves automatically; they are recorded but not stopped at. */
function settle() {
  for (;;) {
    const cur = here();
    if (cur.over !== undefined || node(cur.i)[0] !== "f") return;
    const [, mover, , moves] = node(cur.i);
    const [w, child] = moves[0];
    line.push({ i: child, p: cur.p, y: cur.y, auto: true,
                text: `${who(mover)}: ${data.vocab[w]}` });
  }
}

function choose(step, mover) {
  const cur = here();
  if (step.child === -1) {
    const [, , , , claim] = node(cur.i);
    const honest = claim === setup.t;
    line.push({ i: cur.i, p: cur.p, y: cur.y, over: honest ? 0 : 1,
                text: `P2: challenge — P1 ${honest ? "shows" : "cannot show"} a ${NAME[CARDS[claim]]}` });
  } else {
    const tag = mover === 0 && isBluff(step.label) ? " (bluff)" : "";
    line.push({ i: step.child, p: step.belief || cur.p, y: step.target,
                text: `${who(mover)}: ${step.label}${tag}` });
    settle();
  }
  render();
}

function undo() {
  while (line.length > 1 && here().auto) line.pop();
  if (line.length > 1) line.pop();
  render();
}

const who = (seat) => (seat === 0 ? "P1" : "P2");
const isBluff = (label) => label in CLAIMS && CLAIMS[label] !== setup.p1;

/* ---- rendering --------------------------------------------------------- */

function phaseText(info) {
  const [, , , mover, phase, act, actor, blk, blocker] = info;
  switch (phase) {
    case "ACTION": return `${who(mover)} to choose an action`;
    case "ACTION_CHALLENGE": return `${who(actor)} claims ${act} — ${who(mover)} may challenge`;
    case "BLOCK": return `${who(actor)} plays ${act} — ${who(mover)} may block`;
    case "BLOCK_CHALLENGE": return `${who(blocker)} blocks with ${blk} — ${who(mover)} may challenge`;
    case "LOSE_INFLUENCE": return `${who(mover)} loses an influence`;
    default: return phase;
  }
}

function winner() {
  const cur = here();
  if (cur.over !== undefined) return cur.over;
  if (node(cur.i)[0] === "t") return data.sets[cur.i][0][0] === 1 ? 0 : 1;
  return null;
}

function renderBoard() {
  const cur = here();
  const info = node(cur.i)[2];
  const mover = info[3];
  const w = winner();
  const seat = (s, title, card, note) => `
    <div class="seat${w === null && mover === s ? " turn" : ""}">
      <div class="who">${title} &middot; ${info[s]} coin${info[s] === 1 ? "" : "s"}</div>
      <div class="card">${NAME[card]}</div>
      <div class="note">${note}</div>
    </div>`;
  $("board").innerHTML =
    seat(0, "P1 (bluffer)", setup.p1, "hidden from P2") +
    seat(1, "P2 (honest)", setup.p2, "known to P1") +
    `<div class="seat"><div class="who">Upcards</div>
      <div class="card">${setup.up.map((c) => NAME[c]).join(", ")}</div>
      <div class="note">seen by both</div></div>`;
}

function renderBelief() {
  const { p } = here();
  $("belief").innerHTML = `<div class="title">P2's belief about P1's card</div>` +
    CARDS.map((c, t) => `
      <div class="brow${t === setup.t ? " actual" : ""}">
        <span>${NAME[c]}${t === setup.t ? " ← actual" : ""}</span>
        <span class="bbar"><i style="width:${(100 * p[t]).toFixed(1)}%"></i></span>
        <span class="num">${pct(p[t])}</span>
      </div>`).join("");
}

function mixLine(sigma, p) {
  const parts = CARDS.map((c, t) => (p[t] > EPS && sigma[t] > EPS ? `${NAME[c]} ${pct(sigma[t])}` : null))
    .filter(Boolean);
  return parts.length ? `Played by: ${parts.join(" · ")}` : "Played by no card — off the equilibrium path";
}

function beliefLine(b) {
  return CARDS.map((c, t) => (b[t] > 5e-4 ? `${NAME[c]} ${pct(b[t])}` : null)).filter(Boolean).join(" · ");
}

function li(cls, name, ev, bar, detail, p1) {
  const el = document.createElement("li");
  el.className = `move ${cls}`;
  el.dataset.p1 = p1;
  el.innerHTML = `<span class="name">${name}</span>
    <span class="bar"><i style="width:${(100 * bar).toFixed(1)}%"></i></span>
    <span class="ev">${ev}</span>
    <span class="detail">${detail}</span>`;
  return el;
}

/**
 * The belief the strategies are computed from.  It is P2's belief, except
 * after P1 has made a move its real card never makes: P2 has then ruled the
 * card out, and a card with no weight gets no strategy.  A trace of weight
 * (a tremble) gives it the limit of best replies while changing nothing shown.
 */
function working(p) {
  const t = setup.t;
  if (p[t] > 1e-7) return p;
  const e = 1e-5;
  return p.map((x, k) => (1 - e) * x + (k === t ? e : 0));
}

function renderMoves() {
  const ul = $("moves");
  ul.innerHTML = "";
  const cur = here();
  const t = setup.t;
  const w = winner();
  if (w !== null) {
    $("meta").innerHTML = `<span class="over">${who(w)} wins.</span>`;
    $("meta").dataset.p1 = w === 0 ? 1 : 0;
    return;
  }
  const [kind, , info, , claim] = node(cur.i);
  const pw = working(cur.p);
  const fooled = pw !== cur.p;
  const items = [];
  let p1Value, note, p2Value = 1 - Core.dot(cur.p, cur.y);

  // an option the equilibrium leaves unused can still be worth as much
  const card = NAME[setup.p1];
  if (kind === "1") {
    const steps = Core.p1Step(data, cur.i, pw, cur.y);
    p1Value = Math.max(...steps.map((s) => s.target[t]));
    note = " Values are P1's win chance with the card it really holds.";
    for (const s of steps) {
      const played = s.sigma[t] > EPS;
      let name = s.label;
      if (isBluff(s.label)) name += `<span class="tag bluff">bluff</span>`;
      if (played) {
        name += `<span class="tag freq">plays ${pct(s.sigma[t])}</span>`;
      } else {
        name += `<span class="tag off">leaves equilibrium</span>`;
        if (s.target[t] >= p1Value - 1e-9) name += `<span class="tag zero">equally good</span>`;
      }
      let detail = `${mixLine(s.sigma, pw)}. P2 then believes ${beliefLine(s.belief)}` +
        (s.onPath ? "" : " (the belief that keeps this move unprofitable)") + ".";
      if (!played && s.onPath) {
        detail = `<b>Leaves the equilibrium:</b> a ${card} never makes this move here, so P2 ` +
          `would conclude P1 cannot hold one. ` + detail;
      } else if (!played) {
        detail = `<b>Leaves the equilibrium:</b> no card makes this move here. P2 then believes ` +
          `${beliefLine(s.belief)} (the belief that keeps this move unprofitable).`;
      }
      items.push([li(played ? "best" : "off", name, prob(s.target[t]), s.target[t], detail, s.target[t]),
                  () => choose(s, 0)]);
    }
  } else {
    const steps = Core.p2Step(data, cur.i, pw, cur.y);
    p1Value = steps.reduce((acc, s) => acc + s.mu * s.target[t], 0);
    note = "";
    if (kind === "c") {
      note += ` P1 claims ${NAME[CARDS[claim]]}: by P2's belief it is a lie with probability ` +
        `<b>${pct(1 - cur.p[claim])}</b>. P2 challenges when that beats what passing is worth.`;
    }
    note += " Values are P2's win chance by its belief.";
    for (const s of steps) {
      const ev = 1 - Core.dot(cur.p, s.target);
      const played = s.mu > EPS;
      const label = s.child === -1 ? "challenge" : s.label;
      const name = label + (played ? `<span class="tag freq">plays ${pct(s.mu)}</span>`
        : (ev >= p2Value - 1e-9 ? `<span class="tag zero">equally good</span>`
                                : `<span class="tag zero">not played</span>`));
      const detail = `P1 really holds ${NAME[setup.p1]}, so P2 actually wins ${prob(1 - s.target[t])}.`;
      items.push([li(played ? "best" : "", name, prob(ev), ev, detail, s.target[t]), () => choose(s, 1)]);
    }
  }
  $("meta").innerHTML = `${phaseText(info)}. P1 holding ${NAME[setup.p1]} wins with probability ` +
    `<b>${prob(p1Value)}</b>; by its belief P2 expects to win <b>${prob(p2Value)}</b>.` +
    (fooled ? ` P1 has made a move its real card never makes in equilibrium, so P2 no longer thinks ` +
      `that card possible. P2's strategy still limits what every card can win, so the move cannot ` +
      `gain P1 anything.` : "") + note;
  $("meta").dataset.p1 = p1Value;
  for (const [el, go] of items) { el.onclick = go; ul.appendChild(el); }
}

function renderHistory() {
  const items = line.slice(1).map((e) => `<li>${e.text}</li>`);
  $("history").innerHTML = items.length ? items.join("") : "<li>(start)</li>";
  $("undo").disabled = line.length <= 1 || line.every((e, k) => k === 0 || e.auto);
}

function render() {
  renderBoard();
  renderBelief();
  renderMoves();
  renderHistory();
}

/* ---- wiring ------------------------------------------------------------ */

fillSelects();
$("start").onclick = begin;
$("undo").onclick = undo;
$("reset").onclick = () => {
  $("game").hidden = true; $("histPanel").hidden = true;
  document.getElementById("setup").scrollIntoView({ behavior: "smooth" });
};
