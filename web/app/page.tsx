"use client";
import { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import FusionChart from "@/components/FusionChart";
import { AnimatedNumber, Reveal } from "@/components/ui";
import { buildScene, type Shock } from "@/lib/regimegate";

const PRESETS = [
  { key: "calm", label: "Calm season", emoji: "🌤️", vol: 0.12, shock: "none" as Shock, shockT: 90, shockMag: 2.6 },
  { key: "rising", label: "Rising volatility", emoji: "🌪️", vol: 0.62, shock: "none" as Shock, shockT: 90, shockMag: 2.6 },
  { key: "spike", label: "Promotion spike", emoji: "🛒", vol: 0.3, shock: "spike" as Shock, shockT: 95, shockMag: 3.0 },
  { key: "level", label: "Supply disruption", emoji: "🚢", vol: 0.35, shock: "level" as Shock, shockT: 95, shockMag: 2.6 },
  { key: "drought", label: "Stockout drought", emoji: "📉", vol: 0.3, shock: "drought" as Shock, shockT: 95, shockMag: 2.6 },
];
const TABS = ["Live Simulator", "Real M5 Evidence", "How it works"];

export default function Page() {
  const [tab, setTab] = useState(0);
  const [preset, setPreset] = useState("level");
  const [vol, setVol] = useState(0.35);
  const [shock, setShock] = useState<Shock>("level");
  const [shockT, setShockT] = useState(95);
  const [shockMag, setShockMag] = useState(2.6);
  const [seed, setSeed] = useState(1);

  const apply = (p: (typeof PRESETS)[number]) => {
    setPreset(p.key); setVol(p.vol); setShock(p.shock); setShockT(p.shockT); setShockMag(p.shockMag);
  };
  const scene = useMemo(
    () => buildScene({ vol, shock, shockT, shockMag, seed }),
    [vol, shock, shockT, shockMag, seed],
  );
  const s = scene.scores;
  const bestExpert = Math.min(s["Tree (LightGBM-style)"], s["Deep (N-HiTS-style)"]);
  const won = s.RegimeGate <= Math.min(...Object.values(s)) + 1e-9;

  return (
    <>
      <div className="aurora" aria-hidden={true}><span className="a1" /><span className="a2" /><span className="a3" /></div>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      {/* hero */}
      <Reveal>
        <div className="card relative overflow-hidden p-7 sm:p-9">
          <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-emerald-500/20 blur-3xl animate-floaty" />
          <div className="relative">
            <div className="mb-3 flex flex-wrap gap-2">
              <span className="chip"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" /> Live demo</span>
              <span className="chip">Validated on real Walmart M5</span>
              <span className="chip">&lt; 50k-parameter gate</span>
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight sm:text-5xl">
              <span className="gradient-text">RegimeGate</span>
            </h1>
            <p className="mt-2 max-w-3xl text-base text-slate-300 sm:text-lg">
              An <span className="font-semibold text-white">Adaptive Fusion Controller</span> for
              non-stationary demand. A tiny gate learns <span className="text-emerald-300">where</span> each
              forecaster should be trusted and re-allocates fusion weights live — conditioned on the demand{" "}
              <span className="text-emerald-300">regime</span>, not on the predictions.
            </p>
          </div>
        </div>
      </Reveal>

      {/* tabs */}
      <div className="sticky top-3 z-20 mt-6 flex w-fit gap-1 rounded-2xl border border-white/10 bg-black/40 p-1 backdrop-blur-xl">
        {TABS.map((t, i) => (
          <motion.button key={t} onClick={() => setTab(i)} whileTap={{ scale: 0.95 }}
            className={`relative rounded-xl px-4 py-2 text-sm font-semibold transition ${tab === i ? "text-white" : "text-slate-400 hover:text-slate-200"}`}>
            {tab === i && <motion.span layoutId="tabbg" className="absolute inset-0 rounded-xl bg-white/10 ring-1 ring-white/10" transition={{ type: "spring", bounce: 0.2, duration: 0.5 }} />}
            <span className="relative">{["🎛️", "📊", "🧠"][i]} {t}</span>
          </motion.button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={tab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.3 }} className="mt-6">
          {tab === 0 && (
            <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
              {/* controls */}
              <div className="card h-fit p-5">
                <h3 className="text-sm font-semibold text-slate-200">Scenario</h3>
                <div className="mt-3 grid grid-cols-1 gap-2">
                  {PRESETS.map((p, idx) => (
                    <motion.button key={p.key} onClick={() => apply(p)}
                      whileHover={{ x: 3 }} whileTap={{ scale: 0.98 }}
                      initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.05 }}
                      className={`btn justify-start border text-left ${preset === p.key ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-200 shadow-glow" : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20"}`}>
                      <span>{p.emoji}</span> {p.label}
                    </motion.button>
                  ))}
                </div>
                <div className="mt-5 space-y-4">
                  <Slider label="Volatility regime" value={vol} min={0.05} max={0.95} step={0.01} onChange={(v) => { setVol(v); setPreset(""); }} fmt={(v) => v.toFixed(2)} />
                  <div>
                    <div className="mb-1.5 text-xs font-medium text-slate-400">Inject shock</div>
                    <div className="grid grid-cols-2 gap-1.5">
                      {(["none", "spike", "level", "drought"] as Shock[]).map((sh) => (
                        <motion.button key={sh} onClick={() => { setShock(sh); setPreset(""); }} whileTap={{ scale: 0.95 }}
                          className={`rounded-lg border px-2 py-1.5 text-xs font-medium capitalize transition ${shock === sh ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-200" : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20"}`}>
                          {sh === "level" ? "level shift" : sh}
                        </motion.button>
                      ))}
                    </div>
                  </div>
                  <Slider label="Shock onset (day)" value={shockT} min={50} max={150} step={1} onChange={(v) => { setShockT(v); setPreset(""); }} fmt={(v) => String(Math.round(v))} />
                  <Slider label="Shock magnitude" value={shockMag} min={1.5} max={3.5} step={0.1} onChange={(v) => { setShockMag(v); setPreset(""); }} fmt={(v) => v.toFixed(1) + "×"} />
                  <Slider label="Random seed" value={seed} min={0} max={50} step={1} onChange={(v) => setSeed(v)} fmt={(v) => String(Math.round(v))} />
                </div>
                <p className="mt-5 text-xs leading-relaxed text-slate-500">
                  The gate is a 114-parameter NumPy/JS forward pass — it reacts instantly. No GPU, no model retraining.
                </p>
              </div>

              {/* main */}
              <div className="space-y-5">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Kpi i={0} label="RegimeGate error (RMSE)" value={s.RegimeGate} accent />
                  <Kpi i={1} label="vs. best single expert" value={bestExpert} delta={100 * (1 - s.RegimeGate / bestExpert)} />
                  <Kpi i={2} label="vs. fixed 60/40 blend" value={s["Fixed 60/40"]} delta={100 * (1 - s.RegimeGate / s["Fixed 60/40"])} />
                  <Kpi i={3} label="vs. stacked meta-learner" value={s.Stacking} delta={100 * (1 - s.RegimeGate / s.Stacking)} />
                </div>

                <div className={`chip ${won ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-200" : ""}`}>
                  {won ? "✅ RegimeGate is the most accurate model in this scenario" : "RegimeGate is competitive — try a stronger shock or higher volatility"}
                </div>

                <div className="card p-5">
                  <FusionChart scene={scene} shock={shock} shockT={shockT} />
                </div>

                <div className="grid gap-5 md:grid-cols-2">
                  <div className="card p-5">
                    <h3 className="mb-3 text-sm font-semibold text-slate-200">Forecast error by method <span className="text-slate-500">(lower = better)</span></h3>
                    <ErrorBars scores={s} />
                  </div>
                  <div className="card p-5 text-sm leading-relaxed text-slate-400">
                    <h3 className="mb-2 text-sm font-semibold text-slate-200">What you&apos;re watching</h3>
                    Press <b className="text-emerald-300">▶ Play</b> (or drag the day slider). The orange band swells
                    toward the reactive <b className="text-tree">Tree</b> expert as volatility rises or a shock lands,
                    then relaxes back to the smooth <b className="text-deep">Deep</b> expert when demand settles — the
                    per-step re-allocation a fixed blend or a stacker structurally cannot express. The green{" "}
                    <b className="text-regime">RegimeGate</b> forecast tracks the actual demand most closely throughout.
                  </div>
                </div>
              </div>
            </div>
          )}

          {tab === 1 && <Evidence />}
          {tab === 2 && <HowItWorks />}
        </motion.div>
      </AnimatePresence>

      <footer className="mt-10 border-t border-white/10 pt-5 text-xs text-slate-500">
        Prototype: <span className="text-slate-300">RegimeGate_M5.ipynb</span> (real M5, Colab T4) · Document:{" "}
        <span className="text-slate-300">RegimeGate_Solution_Document.pdf</span>. This dashboard runs the gate live as a
        client-side forward pass — the same architecture, trained on simulated regimes for instant feedback.
      </footer>
      </main>
    </>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium text-slate-400">{label}</span>
        <span className="font-semibold text-emerald-300">{fmt(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} className="w-full" />
    </div>
  );
}

function Kpi({ i = 0, label, value, delta, accent }: { i?: number; label: string; value: number; delta?: number; accent?: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.05 + i * 0.07, ease: [0.2, 0.7, 0.2, 1] }}
      whileHover={{ y: -4 }}
      className={`card group p-4 ${accent ? "ring-1 ring-emerald-400/30 hover:shadow-glow" : "hover:shadow-card"}`}>
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${accent ? "text-emerald-300" : "text-white"}`}>
        <AnimatedNumber value={value} />
      </div>
      {delta !== undefined && (
        <div className={`mt-0.5 text-xs font-semibold ${delta >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
          <motion.span key={delta >= 0 ? "up" : "down"} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="inline-block">
            {delta >= 0 ? "▲" : "▼"}
          </motion.span>{" "}
          <AnimatedNumber value={Math.abs(delta)} decimals={1} />% {delta >= 0 ? "better" : "worse"}
        </div>
      )}
    </motion.div>
  );
}

function ErrorBars({ scores }: { scores: Record<string, number> }) {
  const order = ["Deep (N-HiTS-style)", "Tree (LightGBM-style)", "Stacking", "Fixed 60/40", "RegimeGate"];
  const colors: Record<string, string> = { "Deep (N-HiTS-style)": "#38bdf8", "Tree (LightGBM-style)": "#f59e0b", Stacking: "#fb7185", "Fixed 60/40": "#94a3b8", RegimeGate: "#10b981" };
  const max = Math.max(...order.map((k) => scores[k]));
  return (
    <div className="space-y-2.5">
      {order.map((k) => (
        <div key={k} className="grid grid-cols-[130px_1fr_46px] items-center gap-2 text-xs">
          <span className={k === "RegimeGate" ? "font-semibold text-emerald-300" : "text-slate-400"}>{k.replace(/ \(.*\)/, "")}</span>
          <div className="h-3 overflow-hidden rounded-full bg-white/5">
            <motion.div className="h-full rounded-full" style={{ background: colors[k], boxShadow: `0 0 12px -2px ${colors[k]}` }}
              initial={{ width: 0 }} animate={{ width: `${(scores[k] / max) * 100}%` }} transition={{ duration: 0.7, ease: [0.2, 0.7, 0.2, 1] }} />
          </div>
          <span className="text-right tabular-nums text-slate-300"><AnimatedNumber value={scores[k]} /></span>
        </div>
      ))}
    </div>
  );
}

const ABLATION = [
  ["DL — N-HiTS", "0.760", "0.634", "0.886"],
  ["ML — LightGBM", "0.781", "0.735", "0.828"],
  ["Fixed 60/40", "0.736", "0.642", "0.831"],
  ["Stacking (preds-only)", "0.870", "0.614", "1.125"],
  ["RegimeGate (ours)", "0.734", "0.604", "0.865"],
];
const SHOCKS = [
  ["Demand drought", "2.65", "2.54", "4.64", "3.39", "3.61"],
  ["Level shift (Suez-type)", "3.68", "3.64", "4.62", "4.14", "3.76"],
  ["Spike (panic-buying)", "7.81", "7.67", "11.20", "9.50", "9.52"],
];

function Evidence() {
  return (
    <div className="space-y-6">
      <Reveal className="card p-6">
        <h2 className="text-lg font-bold text-white">Validated on the real Walmart M5 benchmark</h2>
        <p className="mt-1 text-sm text-slate-400">
          Leakage-safe, rolling-origin. Multi-level dataset: 463 series (≈144 smooth aggregates where N-HiTS wins + 320
          intermittent SKUs where LightGBM wins). WRMSSE — lower is better.
        </p>
        <div className="mt-4 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <div className="overflow-hidden rounded-xl border border-white/10">
            <table className="w-full text-sm">
              <thead className="bg-white/5 text-xs uppercase tracking-wide text-slate-400">
                <tr><th className="px-3 py-2 text-left">Variant</th><th className="px-3 py-2">Overall</th><th className="px-3 py-2">Smooth</th><th className="px-3 py-2">Intermittent</th></tr>
              </thead>
              <tbody>
                {ABLATION.map((r, i) => (
                  <tr key={r[0]} className={`border-t border-white/5 ${i === 4 ? "bg-emerald-400/10 font-semibold text-emerald-200" : "text-slate-300"}`}>
                    <td className="px-3 py-2 text-left">{r[0]}</td><td className="px-3 py-2 text-center tabular-nums">{r[1]}</td>
                    <td className="px-3 py-2 text-center tabular-nums">{r[2]}</td><td className="px-3 py-2 text-center tabular-nums">{r[3]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid content-start gap-3">
            <BigStat value="+15.5%" label="RegimeGate vs. stacked meta-learner — better overall" />
            <BigStat value="0.734" label="Best overall WRMSSE of all 5 variants — RegimeGate wins" />
            <p className="text-xs leading-relaxed text-slate-500">
              The predictions-only stacker <b className="text-rose-300">collapses</b> on heterogeneous demand (1.125 on
              intermittent) — a single global blend cannot serve opposite regimes. The context gate stays best.
            </p>
          </div>
        </div>
      </Reveal>

      <Reveal className="card p-6">
        <h3 className="text-base font-semibold text-white">Robustness — absolute WRMSSE under the shock battery <span className="text-slate-500">(lower = better)</span></h3>
        <div className="mt-3 overflow-x-auto rounded-xl border border-white/10">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-xs uppercase tracking-wide text-slate-400">
              <tr><th className="px-3 py-2 text-left">Shock</th><th className="px-3 py-2 text-regime">RegimeGate</th><th className="px-3 py-2">Best expert</th><th className="px-3 py-2">LightGBM</th><th className="px-3 py-2">Fixed</th><th className="px-3 py-2">Stacker</th></tr>
            </thead>
            <tbody>
              {SHOCKS.map((r) => (
                <tr key={r[0]} className="border-t border-white/5 text-slate-300">
                  <td className="px-3 py-2 text-left">{r[0]}</td>
                  <td className="px-3 py-2 text-center font-semibold tabular-nums text-emerald-300">{r[1]}</td>
                  {r.slice(2).map((v, j) => <td key={j} className="px-3 py-2 text-center tabular-nums">{v}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">With the robustness dial engaged, RegimeGate matches the most shock-robust expert on every shock.</p>
      </Reveal>

      <div className="grid gap-6 md:grid-cols-2">
        <Reveal className="card p-5"><img src="shap.png" alt="SHAP attribution on the gate" className="w-full rounded-lg bg-white" />
          <p className="mt-2 text-xs text-slate-500">SHAP on the gate (real M5): rolling CV, intermittency and the shock z-score genuinely drive allocation.</p></Reveal>
        <Reveal className="card p-5"><img src="weight_trajectory.png" alt="Fusion-weight trajectory" className="w-full rounded-lg bg-white" />
          <p className="mt-2 text-xs text-slate-500">Fusion-weight trajectory over the M5 test span — the gate adapts as the regime evolves.</p></Reveal>
      </div>
    </div>
  );
}

function HowItWorks() {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Reveal className="card p-5"><img src="architecture.png" alt="RegimeGate architecture" className="w-full rounded-lg bg-white" /></Reveal>
      <Reveal className="card space-y-4 p-6 text-sm leading-relaxed text-slate-300">
        <p><b className="text-white">The idea in one line.</b> Keep two strong, frozen experts and replace the <i>fixed</i> fusion weight with a <i>learned function of the current regime</i>.</p>
        <ul className="space-y-2">
          <li className="flex gap-2"><span className="text-tree">▰</span><span><b className="text-tree">Tree expert (LightGBM-style)</b> — reactive; wins on volatile / intermittent demand and tracks shocks fast.</span></li>
          <li className="flex gap-2"><span className="text-deep">▰</span><span><b className="text-deep">Deep expert (N-HiTS-style)</b> — smooth, seasonal; wins on calm, structured demand.</span></li>
          <li className="flex gap-2"><span className="text-regime">▰</span><span><b className="text-regime">RegimeGate</b> — a tiny MLP that reads the regime context (volatility, intermittency, a shock z-score, calendar) and emits softmax fusion weights, per step.</span></li>
        </ul>
        <p><b className="text-white">Why it beats a stacker.</b> A stacker blends the two <i>predictions</i> with one fixed rule; two days with identical predictions but different regimes get identical weights. RegimeGate conditions on the <i>regime</i> — the axis a stacker cannot express.</p>
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Accuracy ↔ robustness dial</div>
          <div className="overflow-hidden rounded-xl border border-white/10">
            <table className="w-full text-xs">
              <thead className="bg-white/5 text-slate-400"><tr><th className="px-3 py-2 text-left">Operating point</th><th className="px-3 py-2">Overall WRMSSE</th><th className="px-3 py-2">Under shock</th></tr></thead>
              <tbody className="text-slate-300">
                <tr className="border-t border-white/5"><td className="px-3 py-2">Accuracy (default)</td><td className="px-3 py-2 text-center">0.734 — wins</td><td className="px-3 py-2 text-center">graceful</td></tr>
                <tr className="border-t border-white/5"><td className="px-3 py-2">Robustness</td><td className="px-3 py-2 text-center">0.745 ≈ fixed</td><td className="px-3 py-2 text-center">matches the robust expert</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </Reveal>
    </div>
  );
}

function BigStat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-xl border border-emerald-400/20 bg-emerald-400/5 p-4">
      <div className="text-2xl font-bold text-emerald-300">{value}</div>
      <div className="mt-0.5 text-xs text-slate-400">{label}</div>
    </div>
  );
}
