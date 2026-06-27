// Client-side port of the RegimeGate simulator + the trained gate (NumPy -> TS).
// Mirrors dashboard/regime_sim.py so the gate's context features match its training.
import G from "./gate.json";

const WEEK = 7;
const clip = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const weekly = (t: number) => 1 + 0.12 * Math.sin((2 * Math.PI * t) / WEEK);

// deterministic RNG so a given seed always renders the same scenario
function mulberry32(seed: number) {
  let a = seed >>> 0 || 1;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r: () => number) {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export type Shock = "none" | "spike" | "level" | "drought";

export function simulate(
  n: number, base: number, vol: number, seed: number,
  shock: Shock, shockT: number, shockDur = 12, shockMag = 2.6, rho = 0.72,
) {
  const r = mulberry32(seed + 1);
  const y: number[] = new Array(n);
  const e: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    if (i > 0) e[i] = rho * e[i - 1] + vol * base * gauss(r);
    y[i] = base * weekly(i) + e[i];
  }
  if (shock !== "none") {
    const s = shockT, en = Math.min(n, s + shockDur);
    for (let i = s; i < n; i++) {
      if (shock === "spike" && i < en) y[i] *= shockMag;
      else if (shock === "level") y[i] *= 1 + (shockMag - 1) * 0.6;
      else if (shock === "drought" && i < en) y[i] *= 0.05;
    }
  }
  return y.map((v) => Math.max(0, v));
}

export function experts(y: number[]) {
  const n = y.length;
  const deseas = y.map((v, i) => v / weekly(i));
  const deep: number[] = new Array(n);
  const tree: number[] = new Array(n);
  let deepLevel = deseas[0];
  for (let i = 0; i < n; i++) {
    if (i > 0) deepLevel = 0.1 * deseas[i - 1] + 0.9 * deepLevel;
    deep[i] = Math.max(0, deepLevel * weekly(i));
    tree[i] = i >= 2 ? 0.6 * y[i - 1] + 0.4 * y[i - 2] : i === 1 ? y[0] : y[0];
  }
  return { tree, deep };
}

const EPS = 1e-6;
export function context(y: number[], K = 14) {
  const n = y.length;
  const deseas = y.map((v, i) => v / weekly(i));
  const F: number[][] = [];
  for (let i = 0; i < n; i++) {
    const past = deseas.slice(Math.max(0, i - K), i);
    if (past.length >= 3) {
      const m = past.reduce((a, b) => a + b, 0) / past.length;
      const s = Math.sqrt(past.reduce((a, b) => a + (b - m) ** 2, 0) / past.length);
      const recent = past.slice(-Math.min(7, past.length));
      const older = past.slice(0, Math.max(1, past.length - 7));
      const rm = recent.reduce((a, b) => a + b, 0) / recent.length;
      const om = older.reduce((a, b) => a + b, 0) / older.length;
      F.push([
        s / (Math.abs(m) + EPS),
        clip((deseas[i - 1] - m) / (s + EPS), -6, 6),
        (rm - om) / (Math.abs(m) + EPS),
        Math.log1p(s),
      ]);
    } else F.push([0, 0, 0, 0]);
  }
  return F;
}

const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)));
export function gateForward(F: number[][]) {
  const { mu, sd, W1, b1, W2, b2, temperature } = G as any;
  return F.map((f) => {
    const x = f.map((v, i) => clip((v - mu[i]) / sd[i], -6, 6));
    const h: number[] = b1.map((bj: number, j: number) => {
      let s = bj; for (let i = 0; i < 4; i++) s += x[i] * W1[i][j]; return gelu(s);
    });
    const logit = b2.map((bk: number, k: number) => {
      let s = bk; for (let j = 0; j < 16; j++) s += h[j] * W2[j][k]; return s / temperature;
    });
    const mx = Math.max(logit[0], logit[1]);
    const e0 = Math.exp(logit[0] - mx), e1 = Math.exp(logit[1] - mx);
    return e0 / (e0 + e1); // weight on Tree
  });
}

function fitStacker(tree: number[], deep: number[], y: number[]) {
  const k = Math.max(8, Math.floor(y.length * 0.45));
  // normal equations for [a,b,c] minimising || y - (a*tree + b*deep + c) ||
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const rhs = [0, 0, 0];
  for (let i = 0; i < k; i++) {
    const xi = [tree[i], deep[i], 1];
    for (let a = 0; a < 3; a++) {
      rhs[a] += xi[a] * y[i];
      for (let b = 0; b < 3; b++) A[a][b] += xi[a] * xi[b];
    }
  }
  return solve3(A, rhs);
}
function solve3(A: number[][], b: number[]) {
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c; for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const pv = M[c][c] || 1e-9;
    for (let j = c; j < 4; j++) M[c][j] /= pv;
    for (let r = 0; r < 3; r++) if (r !== c) { const f = M[r][c]; for (let j = c; j < 4; j++) M[r][j] -= f * M[c][j]; }
  }
  return [M[0][3], M[1][3], M[2][3]];
}

const rmse = (y: number[], p: number[], warm: number) => {
  let s = 0, n = 0;
  for (let i = warm; i < y.length; i++) { s += (y[i] - p[i]) ** 2; n++; }
  return Math.sqrt(s / Math.max(1, n));
};

export interface Scene {
  n: number; y: number[]; tree: number[]; deep: number[];
  wTree: number[]; regime: number[]; fixed: number[]; stack: number[];
  scores: Record<string, number>;
}

export function buildScene(opts: {
  vol: number; shock: Shock; shockT: number; shockMag?: number; seed: number;
  n?: number; base?: number; wFixed?: number; warmup?: number;
}): Scene {
  const { vol, shock, shockT, shockMag = 2.6, seed, n = 170, base = 50, wFixed = 0.6, warmup = 20 } = opts;
  const y = simulate(n, base, vol, seed, shock, shockT, 12, shockMag);
  const { tree, deep } = experts(y);
  const raw = gateForward(context(y));
  // temporal smoothing (3-window) + neutral floor (anti flip-flop), matching the notebook
  const wTree = raw.map((_, i) => {
    const lo = Math.max(0, i - 1), hi = Math.min(raw.length - 1, i + 1);
    let s = 0, c = 0; for (let j = lo; j <= hi; j++) { s += raw[j]; c++; }
    return clip(0.9 * (s / c) + 0.1 * 0.5, 0, 1);
  });
  const regime = y.map((_, i) => wTree[i] * tree[i] + (1 - wTree[i]) * deep[i]);
  const fixed = y.map((_, i) => wFixed * tree[i] + (1 - wFixed) * deep[i]);
  const [a, b, c] = fitStacker(tree, deep, y);
  const stack = y.map((_, i) => Math.max(0, a * tree[i] + b * deep[i] + c));
  const scores = {
    "Tree (LightGBM-style)": rmse(y, tree, warmup),
    "Deep (N-HiTS-style)": rmse(y, deep, warmup),
    "Fixed 60/40": rmse(y, fixed, warmup),
    Stacking: rmse(y, stack, warmup),
    RegimeGate: rmse(y, regime, warmup),
  };
  return { n, y, tree, deep, wTree, regime, fixed, stack, scores };
}
