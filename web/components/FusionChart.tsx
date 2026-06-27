"use client";
import { motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Scene, Shock } from "@/lib/regimegate";

const PATH_T = { type: "tween" as const, duration: 0.55, ease: [0.4, 0, 0.2, 1] as const };

const C = { actual: "#cbd5e1", tree: "#f59e0b", deep: "#38bdf8", regime: "#10b981", band: "#f43f5e" };
const W = 1000, X0 = 56, X1 = 980;
const T = { y0: 18, y1: 292 }; // top panel (demand)
const B = { y0: 338, y1: 520 }; // bottom panel (weights)

const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

export default function FusionChart({ scene, shock, shockT }: { scene: Scene; shock: Shock; shockT: number }) {
  const { n } = scene;
  const [day, setDay] = useState(n - 1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<"Slow" | "Normal" | "Fast">("Normal");
  const dps = { Slow: 26, Normal: 46, Fast: 82 }[speed];

  // auto-play the reveal once on first mount; thereafter snap to full on scenario change
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; setDay(0); setPlaying(true); }
    else { setDay(n - 1); setPlaying(false); }
  }, [scene, n]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000; last = now;
      setDay((d) => {
        const nd = d + dt * dps;
        if (nd >= n - 1) { setPlaying(false); return n - 1; }
        return nd;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, dps, n]);

  const sx = (i: number) => X0 + (i / (n - 1)) * (X1 - X0);
  const scaleY = (arr: number[][], y0: number, y1: number) => {
    let lo = Infinity, hi = -Infinity;
    arr.forEach((a) => a.forEach((v) => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
    return (v: number) => y1 - ((v - lo) / (hi - lo)) * (y1 - y0);
  };

  const paths = useMemo(() => {
    const yT = scaleY([scene.y, scene.tree, scene.deep, scene.regime], T.y0, T.y1);
    const yB = (v: number) => B.y1 - v * (B.y1 - B.y0);
    const line = (arr: number[], f: (v: number) => number) =>
      arr.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)} ${f(v).toFixed(1)}`).join(" ");
    const band =
      `M${X0} ${B.y1} ` + scene.wTree.map((v, i) => `L${sx(i).toFixed(1)} ${yB(v).toFixed(1)}`).join(" ") + ` L${X1} ${B.y1} Z`;
    const regimeArea =
      `M${X0} ${T.y1} ` + scene.regime.map((v, i) => `L${sx(i).toFixed(1)} ${yT(v).toFixed(1)}`).join(" ") + ` L${X1} ${T.y1} Z`;
    return {
      yT, yB,
      actual: line(scene.y, yT), tree: line(scene.tree, yT), deep: line(scene.deep, yT),
      regime: line(scene.regime, yT), regimeArea, band, wline: line(scene.wTree, yB),
    };
  }, [scene, n]);

  const revealW = sx(day) - X0;
  const di = Math.floor(day), df = day - di;
  const at = (arr: number[]) => lerp(arr[di], arr[Math.min(n - 1, di + 1)], df);
  const cx = sx(day);
  const cyR = paths.yT(at(scene.regime));
  const cyW = paths.yB(at(scene.wTree));
  const shockX0 = sx(shockT), shockX1 = sx(Math.min(n - 1, shockT + 12));
  const meanW = scene.wTree.slice(20).reduce((a, b) => a + b, 0) / (n - 20);

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <Legend />
        <div className="flex items-center gap-2">
          <div className="flex overflow-hidden rounded-lg border border-white/10">
            {(["Slow", "Normal", "Fast"] as const).map((s) => (
              <button key={s} onClick={() => setSpeed(s)}
                className={`px-2.5 py-1 text-xs font-medium transition ${speed === s ? "bg-white/15 text-white" : "text-slate-400 hover:text-slate-200"}`}>
                {s}
              </button>
            ))}
          </div>
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
            onClick={() => { if (day >= n - 1) setDay(0); setPlaying((p) => !p); }}
            className="btn bg-emerald-500 text-emerald-950 hover:bg-emerald-400 shadow-glow">
            {playing ? "⏸ Pause" : "▶ Play"}
          </motion.button>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} 560`} className="w-full">
        <defs>
          <clipPath id="reveal"><rect x={X0} y="0" width={Math.max(0, revealW)} height="560" /></clipPath>
          <linearGradient id="rg" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={C.regime} stopOpacity="0.30" />
            <stop offset="100%" stopColor={C.regime} stopOpacity="0" />
          </linearGradient>
          <linearGradient id="tg" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={C.tree} stopOpacity="0.34" />
            <stop offset="100%" stopColor={C.tree} stopOpacity="0.04" />
          </linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="3.2" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>

        {/* gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((g) => {
          const y = lerp(T.y1, T.y0, g);
          return <line key={"t" + g} x1={X0} x2={X1} y1={y} y2={y} stroke="rgba(255,255,255,.06)" />;
        })}
        {[0, 0.5, 1].map((g) => {
          const y = lerp(B.y1, B.y0, g);
          return <g key={"b" + g}><line x1={X0} x2={X1} y1={y} y2={y} stroke="rgba(255,255,255,.06)" />
            <text x={X0 - 8} y={y + 3} textAnchor="end" className="fill-slate-500" fontSize="11">{g}</text></g>;
        })}

        {/* shock region */}
        {shock !== "none" && (
          <>
            <rect x={shockX0} y={T.y0} width={Math.max(2, shockX1 - shockX0)} height={T.y1 - T.y0} fill={C.band} opacity="0.10" />
            <line x1={shockX0} x2={shockX0} y1={B.y0} y2={B.y1} stroke={C.band} strokeDasharray="3 3" opacity="0.6" />
            <text x={shockX0 + 4} y={T.y0 + 12} className="fill-rose-300" fontSize="11">shock</text>
          </>
        )}

        {/* panel titles */}
        <text x={X0} y={12} className="fill-slate-400" fontSize="12.5" fontWeight="600">Demand vs. forecasts</text>
        <text x={X0} y={332} className="fill-slate-400" fontSize="12.5" fontWeight="600">Gate&apos;s weight on the Tree expert</text>

        <g clipPath="url(#reveal)">
          <motion.path initial={false} animate={{ d: paths.regimeArea }} transition={PATH_T} fill="url(#rg)" />
          <motion.path initial={false} animate={{ d: paths.actual }} transition={PATH_T} fill="none" stroke={C.actual} strokeWidth="2.2" opacity="0.85" />
          <motion.path initial={false} animate={{ d: paths.tree }} transition={PATH_T} fill="none" stroke={C.tree} strokeWidth="1.2" strokeDasharray="4 3" opacity="0.6" />
          <motion.path initial={false} animate={{ d: paths.deep }} transition={PATH_T} fill="none" stroke={C.deep} strokeWidth="1.2" strokeDasharray="4 3" opacity="0.6" />
          <motion.path initial={false} animate={{ d: paths.regime }} transition={PATH_T} fill="none" stroke={C.regime} strokeWidth="2.8" filter="url(#glow)" />
          <motion.path initial={false} animate={{ d: paths.band }} transition={PATH_T} fill="url(#tg)" />
          <motion.path initial={false} animate={{ d: paths.wline }} transition={PATH_T} fill="none" stroke={C.tree} strokeWidth="2.4" />
        </g>

        {/* play head + markers */}
        <line x1={cx} x2={cx} y1={T.y0} y2={B.y1} stroke="rgba(255,255,255,.28)" strokeWidth="1" />
        {playing && (
          <motion.circle cx={cx} cy={cyR} fill={C.regime}
            animate={{ r: [6, 15], opacity: [0.35, 0] }} transition={{ duration: 1.6, repeat: Infinity, ease: "easeOut" }} />
        )}
        <circle cx={cx} cy={cyR} r="5.5" fill={C.regime} stroke="#06210f" strokeWidth="2" />
        <circle cx={cx} cy={cyW} r="5.5" fill={C.tree} stroke="#241302" strokeWidth="2" />
        <line x1={X0} x2={X1} y1={lerp(B.y1, B.y0, 0.5)} y2={lerp(B.y1, B.y0, 0.5)} stroke="rgba(255,255,255,.14)" strokeDasharray="4 4" />
        <text x={X1} y={B.y1 + 16} textAnchor="end" className="fill-slate-500" fontSize="11">day →</text>
      </svg>

      <input type="range" min={0} max={n - 1} step={0.5} value={day}
        onChange={(e) => { setPlaying(false); setDay(parseFloat(e.target.value)); }}
        className="mt-2 w-full" />
      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>Day {Math.round(day)}</span>
        <span>Mean Tree-weight this scenario: <span className="font-semibold text-tree">{meanW.toFixed(2)}</span></span>
      </div>
    </div>
  );
}

function Legend() {
  const items = [["Actual", C.actual], ["RegimeGate", C.regime], ["Tree expert", C.tree], ["Deep expert", C.deep]] as const;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-300">
      {items.map(([l, c]) => (
        <span key={l} className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c }} />{l}
        </span>
      ))}
    </div>
  );
}
