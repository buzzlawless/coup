"use strict";

const NAME = { D: "Duke", S: "Assassin", C: "Captain", A: "Ambassador", T: "Contessa" };
const CODES = ["D", "S", "C", "A", "T"];
const PHASE = {
  a: "to act", c: "challenge window — only pass, since nobody bluffs",
  b: "may block", k: "block-challenge window — only pass",
  x: "choosing which card to keep", l: "must reveal a card",
};

let shard = null;      // the loaded slice of the table
let manifest = null;   // which file holds which pair of upcards
let stack = [];        // [{index, label}] -- the line so far
const $ = (id) => document.getElementById(id);

for (const id of ["c0", "c1", "d0", "d1"]) {
  $(id).innerHTML = CODES.map((c) => `<option value="${c}">${NAME[c]}</option>`).join("");
}
$("c0").value = "A"; $("c1").value = "A"; $("d0").value = "T"; $("d1").value = "T";

/* ---- data -------------------------------------------------------------- */

const pairKey = (cards) => [...cards].sort().join("");

async function loadShard(a, b) {
  // Ask the manifest which file holds this pair rather than rebuilding its
  // name here: the generator names the files, so only it should decide.
  if (!manifest) {
    const m = await fetch("data/manifest.json");
    if (!m.ok) throw new Error(`could not load the data manifest (${m.status})`);
    manifest = await m.json();
  }
  const want = pairKey([a, b]);
  const entry = manifest.find((e) => pairKey(e.dead) === want);
  if (!entry) throw new Error(`no data for upcards ${NAME[a]} + ${NAME[b]}`);
  const file = entry.file;
  const res = await fetch("data/" + file);
  if (!res.ok) throw new Error(`could not load ${file} (${res.status})`);
  const buf = await res.arrayBuffer();
  const bytes = new Uint8Array(buf);
  // A host may or may not have decompressed this for us, so look rather than
  // assume: 1f 8b is the gzip magic number.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (typeof DecompressionStream === "undefined") {
      throw new Error("this browser cannot decompress the data (needs DecompressionStream)");
    }
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

const isOver = (i) => shard.info[i] === null;
const winnerOf = (i) => shard.winners[String(i)];

/** Probability that `seat` wins at position `i`. */
function valueFor(i, seat) {
  if (isOver(i)) return winnerOf(i) === seat ? 1 : 0;
  const mover = shard.info[i][10];
  return mover === seat ? shard.p[i] : 1 - shard.p[i];
}

/* ---- rendering --------------------------------------------------------- */

function cardsAndCoins(i) {
  const [mc, oc, mn, on, , , , , , , mover] = shard.info[i];
  return mover === 0
    ? { cards: [mc, oc], coins: [mn, on], mover }
    : { cards: [oc, mc], coins: [on, mn], mover };
}

function drawnLabel(i) {
  const d = shard.info[i][9];
  return d ? [...d].map((c) => NAME[c]).join(" + ") : "";
}

function renderBoard(i) {
  const board = $("board");
  if (isOver(i)) {
    const w = winnerOf(i);
    board.innerHTML = "";
    $("meta").innerHTML = `<span class="over">Player ${w + 1} wins.</span>`;
    $("moves").innerHTML = "";
    return;
  }
  const { cards, coins, mover } = cardsAndCoins(i);
  const dead = shard.dead.map((c) => NAME[c]).join(" + ");
  board.innerHTML = [0, 1].map((s) => `
    <div class="seat ${s === mover ? "turn" : ""}">
      <div class="who">Player ${s + 1}${s === mover ? " &mdash; to move" : ""}</div>
      <div class="card">${NAME[cards[s]]}</div>
      <div class="coins">${coins[s]} coin${coins[s] === 1 ? "" : "s"}</div>
    </div>`).join("");

  const info = shard.info[i];
  const bits = [`${PHASE[info[4]]}`];
  if (info[5]) bits.push(`against <b>${info[5]}</b> by the ${info[6] === "self" ? "mover" : "opponent"}`);
  if (info[7]) bits.push(`block claiming <b>${NAME[info[7]]}</b>`);
  if (info[9]) bits.push(`drew <b>${drawnLabel(i)}</b>`);
  $("meta").innerHTML =
    `Player ${mover + 1} ${bits.join(", ")} &nbsp;·&nbsp; ` +
    `win probability <b>${(shard.p[i] * 100).toFixed(2)}%</b>` +
    ` &nbsp;·&nbsp; upcards ${dead}`;
}

function moveRow(label, ev, best, chance, extra) {
  const pct = (ev * 100).toFixed(2);
  return `<li class="move ${best ? "best" : ""} ${chance ? "chance" : ""}">
      <span class="name">${label}${extra || ""}</span>
      <span class="bar"><i style="width:${Math.max(0, Math.min(100, ev * 100))}%"></i></span>
      <span class="ev">${pct}%</span>
    </li>`;
}

function renderMoves(i) {
  const list = $("moves");
  if (isOver(i)) return;
  const moves = shard.moves[i];
  const best = Math.max(...moves.map((m) => m[1]));
  list.innerHTML = moves
    .map((m, k) => ({ m, k }))
    .sort((x, y) => y.m[1] - x.m[1])
    .map(({ m, k }) => {
      const chance = m[2] === -1;
      const tag = chance ? ` <span class="tag">${m[3].length} draws</span>` : "";
      return moveRow(shard.vocab[m[0]], m[1], m[1] >= best - 1e-9, chance, tag)
        .replace("<li ", `<li data-move="${k}" `);
    }).join("");

  list.querySelectorAll("li[data-move]").forEach((el) => {
    el.onclick = () => choose(i, Number(el.dataset.move));
  });
}

function renderOutcomes(i, k) {
  const move = shard.moves[i][k];
  const mover = shard.info[i][10];
  $("meta").innerHTML += ` &nbsp;·&nbsp; <b>pick the draw</b>`;
  // Each row carries the index it leads to, so sorting the display cannot
  // put a row's label on another row's destination.
  $("moves").innerHTML = [...move[3]]
    .sort((a, b) => b[0] - a[0])
    .map(([p, idx]) => {
      const ev = valueFor(idx, mover);
      return moveRow(`${drawnLabel(idx) || "no draw"}`, ev, false, true,
        ` <span class="tag">p = ${(p * 100).toFixed(2)}%</span>`)
        .replace("<li ", `<li data-target="${idx}" `);
    }).join("") +
    `<li class="move" data-cancel="1"><span class="name">&larr; back</span></li>`;

  $("moves").querySelectorAll("li[data-target]").forEach((el) => {
    el.onclick = () => {
      const idx = Number(el.dataset.target);
      push(idx, `${shard.vocab[move[0]]} → ${drawnLabel(idx)}`);
    };
  });
  $("moves").querySelector("li[data-cancel]").onclick = () => render();
}

function renderHistory() {
  $("histPanel").hidden = stack.length < 2;
  $("history").innerHTML = stack.slice(1)
    .map((s, n) => `<li>${n + 1}. ${s.label}</li>`).join("");
}

function render() {
  const i = stack[stack.length - 1].index;
  renderBoard(i);
  renderMoves(i);
  renderHistory();
  $("undo").disabled = stack.length < 2;
}

/* ---- interaction ------------------------------------------------------- */

function choose(i, k) {
  const move = shard.moves[i][k];
  if (move[2] === -1) { renderBoard(i); renderOutcomes(i, k); return; }
  push(move[2], shard.vocab[move[0]]);
}

function push(index, label) {
  const mover = stack.length ? shard.info[stack[stack.length - 1].index][10] : 0;
  stack.push({ index, label: `P${mover + 1}: ${label}` });
  render();
}

$("undo").onclick = () => { stack.pop(); render(); };
$("reset").onclick = () => {
  $("game").hidden = true; $("histPanel").hidden = true; $("setup").hidden = false;
};
$("swap").onclick = () => {
  const c = $("c0").value, n = $("n0").value;
  $("c0").value = $("c1").value; $("n0").value = $("n1").value;
  $("c1").value = c; $("n1").value = n;
};

$("start").onclick = async () => {
  const c0 = $("c0").value, c1 = $("c1").value, d0 = $("d0").value, d1 = $("d1").value;
  const n0 = Number($("n0").value), n1 = Number($("n1").value);
  const err = $("err");
  err.textContent = "";

  const tally = {};
  for (const c of [c0, c1, d0, d1]) tally[c] = (tally[c] || 0) + 1;
  const over = Object.entries(tally).find(([, n]) => n > 3);
  if (over) { err.textContent = `Only three ${NAME[over[0]]}s exist.`; return; }
  if (![n0, n1].every((n) => Number.isInteger(n) && n >= 0 && n <= 12)) {
    err.textContent = "Coins must be between 0 and 12 — 12 is the most anyone can hold, " +
                      "since a turn starting on 10+ must Coup.";
    return;
  }

  err.textContent = "loading…";
  try {
    shard = await loadShard(d0, d1);
  } catch (e) { err.textContent = e.message; return; }

  const index = shard.starts[`${c0}${c1}${n0}.${n1}.0`];
  if (index === undefined) { err.textContent = "That position is not in the table."; return; }
  err.textContent = "";
  stack = [{ index, label: "start" }];
  $("setup").hidden = true; $("game").hidden = false;
  render();
};
