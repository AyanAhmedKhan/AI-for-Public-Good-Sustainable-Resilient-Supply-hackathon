# RegimeGate — Demo Video Script & Storyboard

Two deliverables live in `media/`:

1. **`RegimeGate_demo.mp4`** — a finished **90.7 s · 1920×1080 · 30 fps** explainer with on‑screen
   captions and no audio. It is submittable as‑is. It is rendered by `render_video.py` using the
   *real* trained gate (`dashboard/gate_weights.npz`) and the same NumPy simulator the live dashboard
   runs — the "watch the gate re‑allocate" sequence is the actual forward pass, not a mock‑up.
2. **This script** — a timed voiceover + storyboard. Use it **either** to narrate over the MP4
   (timecodes below line up with it exactly) **or** to record a live screen‑capture walkthrough of the
   Streamlit dashboard. I can't screen‑record your machine, so Option B is yours to capture — this
   gives you everything to do it in one take.

Voiceover is ~230 words ≈ comfortable at ~150 wpm with pauses. Tone: confident, plain, a little
fast — judges have seen a hundred decks. Cut any line that feels long; the captions already carry it.

---

## Option A — narrate over the rendered MP4 (timecodes are exact)

| Time | On screen | Voiceover |
|---|---|---|
| **0:00–0:06** | Title card: *RegimeGate — Adaptive Fusion Controller* | "This is **RegimeGate** — a tiny adaptive controller for demand that never stops changing." |
| **0:07–0:18** | Demand curve draws in; calm → level shift; faint Deep/Tree experts | "Real supply‑chain demand is **non‑stationary** — calm for weeks, then a disruption rewrites the series. And no single forecaster is best across both: a smooth **deep** model wins when it's calm, a reactive **tree** model wins when it's turbulent." |
| **0:18–0:28** | Fixed blend + stacker lines; stacker mis‑tracks; right cards 0.870 / 0.736 | "The usual fixes don't fix this. A fixed blend and a stacked meta‑learner both assume each model's competence **never changes** — so the stacker actually **collapses** when the regime shifts." |
| **0:28–0:35** | Architecture diagram; three bullets build in | "RegimeGate learns **where** each expert should be trusted. Two frozen experts, and a gate — under **fifty thousand** parameters — that reads the regime and emits fusion weights, per step. It conditions on the **regime**, not the predictions — which puts it provably **beyond a stacked meta‑learner**." |
| **0:35–0:63** | **Centerpiece.** Live forecasts (top) + fusion‑weight area (bottom) + right readout; shock hits ~0:43, gate swings to Tree 84%, then scoreboard reveals | "Here's that exact gate, **running live**. In the calm stretch it leans on the smooth **Deep** expert. Then the **shock** hits — and in about **one step** the gate swings most of its weight onto the reactive **Tree** expert, tracking the new level. As it settles, it eases back. On this run, that fused forecast **beats every expert and every baseline** — while the stacker collapses." |
| **0:63–0:74** | M5 WRMSSE bar chart grows; +15.5% callout; two‑sided specialization | "And this holds on **real Walmart M5** data, with a leakage‑safe, rolling‑origin protocol. RegimeGate has the **best overall WRMSSE** — **fifteen and a half percent** better than the stacked meta‑learner — with genuine **two‑sided specialization**." |
| **0:74–0:82** | Accuracy↔robustness dial sweeps; anti‑fragility guards list | "It's also **tunable**: one knob moves between an accuracy‑optimal and a robustness‑optimal point. At the robust setting, anti‑fragility guards keep it **never meaningfully worse** than the static blend — matching the most shock‑robust expert on **every** shock." |
| **0:82–0:90** | Closing wordmark; dashboard / notebook / solution‑doc chips | "RegimeGate — **beyond a stacked meta‑learner**, in under fifty thousand parameters. There's a live dashboard, a full Colab notebook, and the solution document. **Team Absolute.**" |

---

## Option B — live dashboard walkthrough (you screen‑record)

Same arc, but you drive the real Streamlit app so judges see it's interactive. Target ~90–110 s.

1. **Hook (0:00–0:10)** — On the **Live Simulator** tab, scenario on *Calm season*.
   > "Demand is non‑stationary — and which forecaster is best keeps changing. RegimeGate is a tiny gate that re‑allocates between two experts in real time. Watch."
2. **Inject a shock (0:10–0:35)** — Pick **Suez‑style level shift** (or promo spike), then press **▶ Play**.
   Point the cursor at the **fusion‑weight** panel as it animates.
   > "Calm regime — the gate trusts the smooth Deep expert. Now I inject a level shift… and in about one step it swings to the reactive Tree expert. Live accuracy stays ahead of the fixed blend, the stacker, and each expert."
3. **Scrub to prove it's per‑step (0:35–0:45)** — Drag the **day slider** back across the shock.
   > "It's not a fixed blend — the weights are conditioned on the observable regime, step by step."
4. **Real M5 evidence (0:45–1:05)** — Switch to the **Real M5 Evidence** tab.
   > "On real Walmart M5, leakage‑safe and rolling‑origin: best overall WRMSSE, and **+15.5%** over the stacked meta‑learner, which collapses on intermittent demand."
5. **How it works + close (1:05–1:30)** — **How it works** tab; show the architecture and the dial.
   > "Two frozen experts, a sub‑50k‑parameter gate, anti‑fragility guards so it's never worse than the static blend — and one knob to dial accuracy versus robustness. That's RegimeGate. Team Absolute."

**Capture tips:** record at 1920×1080; hide the Streamlit top bar (≡ → *Settings* → wide mode);
do a silent run‑through first so the **▶ Play** animation lands while you're mid‑sentence; OBS Studio
or the Windows **Win+G** Game Bar both record clean 1080p. Keep the mouse calm — no jitter.

---

## Re‑rendering the MP4

```bash
python media/render_video.py        # -> media/RegimeGate_demo.mp4
```

Knobs at the top of `render_video.py`: `W,H,DPI,FPS` (resolution/frame‑rate), the palette block, and
the per‑scene `intro/run/hold` durations inside each `scene_*` function. The centerpiece scenario is
the `R.build_scene(...)` call (~line 122) — `seed=15`, `shock="level"`; change the seed to roll a
different (still real) demand series.
