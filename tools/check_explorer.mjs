/**
 * Browser checks for docs/ -- the things unit tests cannot see.
 *
 *   cd docs && python3 -m http.server 8777 &
 *   node tools/check_explorer.mjs
 *
 * Two failures here have already shipped, so both are covered:
 *   - every pair of upcards must load (a name rebuilt in the page disagreed
 *     with the name the generator wrote, for six of the fifteen pairs);
 *   - every chance outcome must land where its own label says (the rows were
 *     sorted for display but looked up in the unsorted array).
 *
 * It also covers play mode: the automatic side must move on its own, its
 * options must not be clickable, and the deck must be rolled rather than
 * offered as a choice.
 */
import { chromium } from "/opt/node22/lib/node_modules/playwright/index.mjs";

const BASE = process.env.BASE || "http://localhost:8777/index.html";
const CODES = ["D", "S", "C", "A", "T"];
const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e.message)));
await page.goto(BASE);

async function clickMove(text) {
  for (const el of await page.$$("#moves li.move")) {
    const t = (await el.textContent()).replace(/\s+/g, " ");
    if (t.includes(text)) { await el.click(); await page.waitForTimeout(120); return t; }
  }
  throw new Error(`no move matching "${text}"`);
}

async function restart(c0, c1, d0, d1, autoSeat = "") {
  await page.evaluate(() => {
    const g = document.getElementById("game");
    if (!g.hidden) document.getElementById("reset").click();
  });
  for (const [id, v] of [["c0", c0], ["c1", c1], ["d0", d0], ["d1", d1], ["auto", autoSeat]]) {
    await page.selectOption("#" + id, v);
  }
  await page.click("#start");
  await page.waitForSelector("#moves li.move", { timeout: 8000 });
}

let failures = 0;

/* 1. every pair of upcards loads */
let loaded = 0;
for (const d0 of CODES) for (const d1 of CODES) {
  const tally = { [d0]: 1 };
  tally[d1] = (tally[d1] || 0) + 1;
  const pick = () => {
    for (const c of CODES) if ((tally[c] || 0) < 3) { tally[c] = (tally[c] || 0) + 1; return c; }
  };
  try { await restart(pick(), pick(), d0, d1); loaded++; }
  catch { failures++; console.log(`  FAIL upcards ${d0}+${d1} did not load`); }
}
console.log(`upcard pairs that load: ${loaded}/25`);

/* 2. every chance outcome lands where its label says */
await restart("A", "A", "C", "A");
for (const m of ["Foreign Aid", "pass", "Exchange", "pass"]) await clickMove(m);
await clickMove("Contessa + Duke");
await clickMove("exchange Duke");
await clickMove("Exchange");
await clickMove("pass");

const picker = (await page.textContent("#meta")).replace(/\s+/g, " ");
if (!/percentages are Player \d's equity after it/.test(picker)) {
  failures++; console.log("  FAIL draw picker does not say whose equity it shows");
}

const labels = await page.$$eval("#moves li[data-target]",
  (ls) => ls.map((l) => l.querySelector(".name").textContent.replace(/\s+p =.*/, "").trim()));
let matched = 0;
for (const label of labels) {
  await clickMove(label);
  const meta = (await page.textContent("#meta")).replace(/\s+/g, " ");
  if (meta.includes("drew " + label)) matched++;
  else { failures++; console.log(`  FAIL picked "${label}" but landed elsewhere`); }
  await page.click("#undo");
  await page.waitForTimeout(120);
  await clickMove("pass");
}
console.log(`chance outcomes landing correctly: ${matched}/${labels.length}`);

/* 3. play mode: the automatic side moves itself, and the deck is rolled */
await restart("A", "A", "C", "A", "1");
let offeredPicker = false, clickedTheirs = false;
for (let step = 0; step < 80; step++) {
  if (await page.$(".over")) break;
  if (await page.$("#moves li[data-target]")) offeredPicker = true;
  const mine = await page.$("#moves li.move:not(.locked)");
  if (!mine) { await page.waitForTimeout(350); continue; }
  const meta = await page.textContent("#meta");
  if (/Player 2 .*to act|Player 2 may/.test(meta) && !/choosing/.test(meta)) clickedTheirs = true;
  await mine.click();
  await page.waitForTimeout(650);
}
const plies = await page.$$eval("#history li", (ls) => ls.map((l) => l.textContent.trim()));
const theirs = plies.filter((h) => h.includes("(auto)"));
const finished = !!(await page.$(".over"));
if (offeredPicker) { failures++; console.log("  FAIL a draw was offered as a choice in play mode"); }
if (!theirs.length) { failures++; console.log("  FAIL the automatic side never moved"); }
if (!theirs.every((h) => h.startsWith("P2"))) { failures++; console.log("  FAIL a non-P2 ply was marked automatic"); }
if (!finished) { failures++; console.log("  FAIL play mode did not reach a result"); }
console.log(`play mode: ${plies.length} plies, ${theirs.length} automatic, result reached: ${finished}`);

if (errors.length) { failures++; console.log("page errors:", errors); }
console.log(failures ? `\n${failures} FAILURES` : "\nall checks passed");
await browser.close();
process.exit(failures ? 1 : 0);
