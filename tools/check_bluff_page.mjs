/**
 * Browser checks for docs/bluff.html.
 *
 *   cd docs && python3 -m http.server 8777 &
 *   node tools/check_bluff_page.mjs
 *
 * Plays random lines on all four data files and checks, at every decision,
 * that what the page shows is an equilibrium: every option the mover plays is
 * worth exactly the position's value to it, and none is worth more; and that
 * P2 only ever rules out P1's real card after P1 has left the equilibrium.
 */
import { chromium } from "/opt/node22/lib/node_modules/playwright/index.mjs";

const BASE = process.env.BASE || "http://localhost:8777/bluff.html";
const CARDS = ["D", "S", "C", "T"];
const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e.message)));
await page.goto(BASE);

let seed = 12345;
const rand = (n) => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; };

async function restart(c0, c1, d0, d1, n0, n1, first) {
  await page.evaluate(() => {
    if (!document.getElementById("game").hidden) document.getElementById("reset").click();
  });
  for (const [id, v] of [["c0", c0], ["c1", c1], ["d0", d0], ["d1", d1], ["first", String(first)]]) {
    await page.selectOption(`#${id}`, v);
  }
  await page.fill("#n0", String(n0));
  await page.fill("#n1", String(n1));
  await page.click("#start");
  await page.waitForFunction(() => !document.getElementById("game").hidden
    || document.getElementById("err").textContent);
}

/** The position's values from the meta line and every option's [value, played]. */
const read = () => page.evaluate(() => {
  const b = [...document.querySelectorAll("#meta b")].map((x) => x.textContent);
  const moves = [...document.querySelectorAll("#moves li.move")].map((li) => ({
    ev: Number(li.querySelector(".ev").textContent),
    played: li.classList.contains("best"),
    p1: Number(li.dataset.p1),
  }));
  const over = document.querySelector("#meta .over");
  const text = document.getElementById("meta").textContent;
  return { b, moves, over: over && over.textContent, text,
           p1: Number(document.getElementById("meta").dataset.p1),
           p2turn: /Values are P2/.test(text) };
});

let fail = 0;
const check = (ok, msg) => { if (!ok) { fail++; console.log("FAIL", msg); } };

// 1. a figure the Python solver gives: P2 Captain, upcards Contessa x2, P1 first,
//    no coins.  Ex-ante P1 wins 0.556; holding an Assassin it wins outright.
await restart("S", "C", "T", "T", 0, 0, 0);
let s = await read();
check(s.b[0] === "1.000" && s.b[1] === "0.444", `known start: ${s.b}`);
await restart("C", "C", "T", "T", 0, 0, 0);
s = await read();
check(s.b[0] === "1.000" && s.b[1] === "0.444", `known start, Captain: ${s.b}`);

// 2. an impossible deal is refused
await restart("T", "T", "T", "T", 0, 0, 0);
check(/only three Contessas/.test(await page.textContent("#err")), "four Contessas accepted");

// 3. random lines on every shard
let decisions = 0, games = 0;
for (const p2 of CARDS) {
  for (let g = 0; g < 12; g++) {
    let hand;
    for (;;) {
      hand = [CARDS[rand(4)], p2, CARDS[rand(4)], CARDS[rand(4)]];
      if (CARDS.every((c) => hand.filter((x) => x === c).length <= 3)) break;
    }
    await restart(hand[0], hand[1], hand[2], hand[3], rand(8), rand(8), rand(2));
    let promised = null, departed = false;
    for (let step = 0; step < 150; step++) {
      s = await read();
      // what the last move promised P1 is what P1 is then worth
      if (promised !== null) check(Math.abs(s.p1 - promised) < 1e-6, `promised ${promised}, now ${s.p1}: ${s.text}`);
      if (s.over) break;
      // P2 can only rule out P1's real card after P1 leaves the equilibrium
      check(departed || !/no longer thinks/.test(s.text), `card ruled out on the equilibrium path: ${s.text}`);
      const value = s.p2turn ? Number(s.b[1]) : Number(s.b[0]);
      const played = s.moves.filter((m) => m.played);
      check(played.length > 0, `no move played: ${s.text}`);
      if (!s.p2turn) {
        for (const m of played) check(Math.abs(m.ev - value) < 2e-3, `P1 plays ${m.ev} at value ${value}: ${s.text}`);
      } else {
        for (const m of played) check(Math.abs(m.ev - value) < 2e-3, `P2 plays ${m.ev} at value ${value}: ${s.text}`);
      }
      for (const m of s.moves) check(m.ev <= value + 2e-3, `option ${m.ev} beats value ${value}: ${s.text}`);
      decisions++;
      // mostly follow the equilibrium, sometimes stray to test off-path play
      const pool = rand(4) === 0 ? s.moves : played;
      const pick = s.moves.indexOf(pool[rand(pool.length)]);
      promised = s.moves[pick].p1;
      if (!s.p2turn && !s.moves[pick].played) departed = true;
      await (await page.$$("#moves li.move"))[pick].click();
    }
    games++;
    if (rand(3) === 0) {
      const before = await page.$$eval("#history li", (x) => x.length);
      if (!(await page.isDisabled("#undo"))) {
        await page.click("#undo");
        const after = await page.$$eval("#history li", (x) => x.length);
        check(after < before || before === 1, "undo did nothing");
      }
    }
  }
}

check(errors.length === 0, `page errors: ${errors.join(" | ")}`);
console.log(`${games} games, ${decisions} decisions checked, ${fail} failures`);
await browser.close();
process.exit(fail ? 1 : 0);
