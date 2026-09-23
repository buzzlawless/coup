/*
 * The bluffing explorer's engine room: a small linear-programming solver and
 * the per-node strategy programs, with no DOM.  It runs in the page and under
 * node, where the tests check it against SciPy on the real problems.
 *
 * A line is followed with two pieces of state: P2's belief p about P1's card,
 * and the guarantee y P2 is committed to -- a point of the node's set with
 * p . y equal to the value.  Carrying y is what keeps both sides' mixes
 * consistent where P2 is indifferent, which is where bluffing lives.
 */
(function (root) {
  "use strict";

  const EPS = 1e-9;

  /* ---- a two-phase simplex, Bland's rule: minimise c.x, A_ub x <= b_ub,
   *      A_eq x = b_eq, x >= 0.  Returns {x, fun} or null if infeasible. ---- */
  function linprog(c, Aub, bub, Aeq, beq) {
    Aub = Aub || []; bub = bub || []; Aeq = Aeq || []; beq = beq || [];
    const n = c.length;
    const rows = [];
    Aub.forEach((a, i) => rows.push({ a: a.slice(), b: bub[i], type: "le" }));
    Aeq.forEach((a, i) => rows.push({ a: a.slice(), b: beq[i], type: "eq" }));
    for (const r of rows) {
      if (r.b < 0) {
        r.a = r.a.map((v) => -v); r.b = -r.b;
        if (r.type === "le") r.type = "ge";
      }
    }
    const m = rows.length;
    const nSlack = rows.filter((r) => r.type !== "eq").length;
    const nArt = rows.filter((r) => r.type !== "le").length;
    const N = n + nSlack + nArt;
    const T = [];
    const basis = [];
    let s = n, a = n + nSlack;
    const artCols = new Set();
    for (const r of rows) {
      const row = new Array(N + 1).fill(0);
      for (let j = 0; j < n; j++) row[j] = r.a[j];
      row[N] = r.b;
      if (r.type === "le") { row[s] = 1; basis.push(s); s++; }
      else if (r.type === "ge") { row[s] = -1; s++; row[a] = 1; basis.push(a); artCols.add(a); a++; }
      else { row[a] = 1; basis.push(a); artCols.add(a); a++; }
      T.push(row);
    }
    const allowed = new Array(N).fill(true);

    function pivot(r, col) {
      const pr = T[r], pv = pr[col];
      for (let j = 0; j <= N; j++) pr[j] /= pv;
      for (let i = 0; i < T.length; i++) {
        if (i === r) continue;
        const f = T[i][col];
        if (Math.abs(f) < 1e-15) continue;
        const Ti = T[i];
        for (let j = 0; j <= N; j++) Ti[j] -= f * pr[j];
      }
      basis[r] = col;
    }

    function run(cost) {
      // cost row: reduced costs, last entry is -objective
      const z = new Array(N + 1).fill(0);
      for (let j = 0; j < N; j++) z[j] = cost[j] || 0;
      for (let i = 0; i < T.length; i++) {
        const cb = cost[basis[i]] || 0;
        if (cb) for (let j = 0; j <= N; j++) z[j] -= cb * T[i][j];
      }
      T.push(z);
      const obj = T.length - 1;
      for (let it = 0; it < 5000; it++) {
        let enter = -1;
        for (let j = 0; j < N; j++) {
          if (allowed[j] && T[obj][j] < -EPS) { enter = j; break; }
        }
        if (enter < 0) break;
        let leave = -1, best = Infinity;
        for (let i = 0; i < obj; i++) {
          const v = T[i][enter];
          if (v > EPS) {
            const ratio = T[i][N] / v;
            if (ratio < best - 1e-12 || (Math.abs(ratio - best) <= 1e-12 && basis[i] < basis[leave])) {
              best = ratio; leave = i;
            }
          }
        }
        if (leave < 0) { T.pop(); return "unbounded"; }
        pivot(leave, enter);
      }
      const value = -T[obj][N];
      T.pop();
      return value;
    }

    if (nArt) {
      const phase1 = new Array(N).fill(0);
      for (const col of artCols) phase1[col] = 1;
      const v = run(phase1);
      if (v === "unbounded" || v > 1e-7) return null;
      // drive any artificial still in the basis out, or drop its redundant row
      for (let i = T.length - 1; i >= 0; i--) {
        if (!artCols.has(basis[i])) continue;
        let col = -1;
        for (let j = 0; j < N; j++) {
          if (!artCols.has(j) && Math.abs(T[i][j]) > 1e-9) { col = j; break; }
        }
        if (col >= 0) pivot(i, col);
        else { T.splice(i, 1); basis.splice(i, 1); }
      }
      for (const col of artCols) allowed[col] = false;
    }
    const cost = new Array(N).fill(0);
    for (let j = 0; j < n; j++) cost[j] = c[j];
    const v = run(cost);
    if (v === "unbounded") return null;
    const x = new Array(n).fill(0);
    basis.forEach((col, i) => { if (col < n) x[col] = T[i][N]; });
    return { x, fun: v };
  }

  /* ---- the game ---------------------------------------------------------- */

  const dot = (u, v) => u.reduce((s, x, i) => s + x * v[i], 0);
  const minSupport = (verts, p) => Math.min(...verts.map((v) => dot(v, p)));
  const argminVertex = (verts, p) => {
    let best = verts[0], bv = dot(verts[0], p);
    for (const v of verts) { const d = dot(v, p); if (d < bv - 1e-12) { bv = d; best = v; } }
    return best.slice();
  };

  /** [label, vertices] for every choice at a node, a call included. */
  function options(data, i) {
    const [kind, , , moves, claim] = data.nodes[i];
    const out = moves.map(([w, child]) => [data.vocab[w], data.sets[child], child]);
    if (kind === "c") {
      const e = [0, 0, 0, 0]; e[claim] = 1;
      out.unshift(["call", [e], -1]);
    }
    return out;
  }

  /** P1's equilibrium mix: q[a][t] = P(P1 holds t and plays a). */
  function p1Split(opts, p) {
    const A = opts.length, nq = 4 * A;
    const c = new Array(nq + A).fill(0);
    for (let a = 0; a < A; a++) c[nq + a] = -1;
    const Aub = [], bub = [];
    opts.forEach(([, verts], a) => {
      for (const v of verts) {
        const r = new Array(nq + A).fill(0);
        for (let t = 0; t < 4; t++) r[4 * a + t] = -v[t];
        r[nq + a] = 1;
        Aub.push(r); bub.push(0);
      }
    });
    const Aeq = [];
    for (let t = 0; t < 4; t++) {
      const r = new Array(nq + A).fill(0);
      for (let a = 0; a < A; a++) r[4 * a + t] = 1;
      Aeq.push(r);
    }
    const res = linprog(c, Aub, bub, Aeq, p.slice());
    if (!res) throw new Error("P1 split infeasible");
    const q = [];
    for (let a = 0; a < A; a++) q.push(res.x.slice(4 * a, 4 * a + 4));
    return { q, value: -res.fun };
  }

  /** The point of conv(verts) at most `bound` that is best for P2 under `belief`. */
  function target(verts, belief, bound) {
    const k = verts.length;
    const c = verts.map((v) => dot(v, belief));
    const Aub = [0, 1, 2, 3].map((t) => verts.map((v) => v[t]));
    const bub = bound.map((b) => b + EPS);
    const res = linprog(c, Aub, bub, [new Array(k).fill(1)], [1]);
    if (!res) return null;
    const x = [0, 0, 0, 0];
    res.x.forEach((l, j) => { for (let t = 0; t < 4; t++) x[t] += l * verts[j][t]; });
    return x;
  }

  /** A belief under which `point` is P2's best reply, for moves no card makes. */
  function supportingBelief(verts, point) {
    // variables p0..p3, m+, m- ; maximise m = m+ - m-
    const c = [0, 0, 0, 0, -1, 1];
    const Aub = verts.map((v) => {
      const d = v.map((x, t) => x - point[t]);
      return [-d[0], -d[1], -d[2], -d[3], 1, -1];
    });
    const res = linprog(c, Aub, new Array(verts.length).fill(0), [[1, 1, 1, 1, 0, 0]], [1]);
    return res ? res.x.slice(0, 4) : [0.25, 0.25, 0.25, 0.25];
  }

  function p1Step(data, i, p, y) {
    const opts = options(data, i);
    const { q } = p1Split(opts, p);
    return opts.map(([label, verts, child], a) => {
      const mass = q[a].reduce((s, x) => s + x, 0);
      const sigma = q[a].map((x, t) => (p[t] > EPS ? x / p[t] : 0));
      let belief = null, tgt;
      if (mass > EPS) {
        belief = q[a].map((x) => x / mass);
        tgt = target(verts, belief, y);
      } else {
        tgt = target(verts, [1, 1, 1, 1], y);
      }
      if (!tgt) tgt = argminVertex(verts, belief || p);
      if (!belief) belief = supportingBelief(verts, tgt);
      return { label, child, sigma, belief, target: tgt, onPath: mass > EPS };
    });
  }

  function p2Step(data, i, p, y) {
    const opts = options(data, i);
    const stacked = [], owner = [];
    opts.forEach(([, verts], b) => verts.forEach((v) => { stacked.push(v); owner.push(b); }));
    const k = stacked.length;
    const c = stacked.map((v) => dot(v, p));
    const Aub = [0, 1, 2, 3].map((t) => stacked.map((v) => v[t]));
    const res = linprog(c, Aub, y.map((b) => b + EPS), [new Array(k).fill(1)], [1]);
    return opts.map(([label, verts, child], b) => {
      let mu = 0, tgt = null;
      if (res) {
        const idx = owner.map((o, j) => (o === b ? j : -1)).filter((j) => j >= 0);
        mu = idx.reduce((s, j) => s + res.x[j], 0);
        if (mu > EPS) {
          tgt = [0, 0, 0, 0];
          for (const j of idx) for (let t = 0; t < 4; t++) tgt[t] += res.x[j] * stacked[j][t] / mu;
        }
      }
      if (!tgt) { mu = 0; tgt = argminVertex(verts, p); }
      return { label, child, mu, target: tgt };
    });
  }

  const api = { linprog, dot, minSupport, argminVertex, options, p1Split, target,
                supportingBelief, p1Step, p2Step };
  root.BluffCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
