/*
 * Checks docs/bluff-core.js against ground truth it did not compute.
 *
 *   node tools/check_bluff_core.cjs
 *
 * At every P1 decision the split LP's value must equal the support value of
 * the node's guarantee set -- and those sets come from the Python solver.  On
 * top of that, the same consistency checks the Python reference passes: P2
 * never exceeds its promise, and on the path P2 always best-replies.
 */
const fs = require("fs");
const zlib = require("zlib");
const path = require("path");
const core = require("../docs/bluff-core.js");

const dir = path.join(__dirname, "..", "docs", "data", "bluff");
let rng = 12345;
const rand = () => ((rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648);
const dirichlet = () => {
  const g = [0, 1, 2, 3].map(() => -Math.log(rand() + 1e-12));
  const s = g.reduce((a, b) => a + b, 0);
  return g.map((x) => x / s);
};

let failures = 0, p1Checks = 0, p2Checks = 0;
let worstSplit = 0, worstBound = 0, worstReply = 0;
for (const card of ["D", "S", "C", "T"]) {
  const data = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(dir, card + ".json.gz"))));
  const V = (i, p) => core.minSupport(data.sets[i], p);
  for (let trial = 0; trial < 250; trial++) {
    const key = `${Math.floor(rand() * 5)}.${Math.floor(rand() * 5)}.${Math.floor(rand() * 2)}`;
    let i = data.starts[key];
    let p = dirichlet();
    let y = core.argminVertex(data.sets[i], p);
    for (let step = 0; step < 40; step++) {
      const [kind, , , moves] = data.nodes[i];
      if (kind === "t") break;
      if (kind === "f") { i = moves[0][1]; continue; }
      if (kind === "1") {
        const opts = core.options(data, i);
        const { value } = core.p1Split(opts, p);
        worstSplit = Math.max(worstSplit, Math.abs(value - V(i, p)));
        const steps = core.p1Step(data, i, p, y);
        for (const s of steps) {
          worstBound = Math.max(worstBound, Math.max(...s.target.map((x, t) => x - y[t])));
          if (s.onPath) worstReply = Math.max(worstReply, Math.abs(core.dot(s.belief, s.target) - V(s.child, s.belief)));
        }
        p1Checks++;
        const s = steps[Math.floor(rand() * steps.length)];
        i = s.child; p = s.belief; y = s.target;
      } else {
        const steps = core.p2Step(data, i, p, y);
        const mix = [0, 1, 2, 3].map((t) => steps.reduce((acc, s) => acc + s.mu * s.target[t], 0));
        worstBound = Math.max(worstBound, Math.max(...mix.map((x, t) => x - y[t])));
        const total = steps.reduce((a, s) => a + s.mu, 0);
        if (Math.abs(total - 1) > 1e-6) failures++;
        p2Checks++;
        const s = steps[Math.floor(rand() * steps.length)];
        if (s.label === "call") break;
        i = s.child; y = s.target;
      }
    }
  }
}
console.log(`P1 decisions: ${p1Checks}   P2 decisions: ${p2Checks}`);
console.log(`  split LP value vs Python's set value, worst:   ${worstSplit.toExponential(2)}`);
console.log(`  continuation above P2's promise, worst:        ${worstBound.toExponential(2)}`);
console.log(`  on-path continuation vs best reply, worst:     ${worstReply.toExponential(2)}`);
console.log(`  P2 mixes not summing to 1:                     ${failures}`);
const ok = worstSplit < 1e-6 && worstBound < 1e-6 && worstReply < 1e-6 && failures === 0;
console.log(ok ? "\nall checks passed" : "\nFAILED");
process.exit(ok ? 0 : 1);
