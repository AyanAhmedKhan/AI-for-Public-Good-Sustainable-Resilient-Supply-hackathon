# RegimeGate — Web Dashboard (Next.js + Tailwind)

A modern, animated single-page dashboard for the **RegimeGate Adaptive Fusion Controller**, built
with **Next.js 14 · TypeScript · Tailwind CSS · Framer Motion**. Dark, premium UI with a Play/scrub
time animation.

- **Live Simulator** — pick a supply-chain scenario (calm season, promo spike, Suez-style level
  shift, stockout drought), tune volatility/shock, and press **▶ Play** to watch the forecast and the
  gate's fusion weights animate day-by-day as it shifts between the reactive *Tree* expert and the
  smooth *Deep* expert. Live accuracy vs. the fixed blend, the stacker, and each expert.
- **Real M5 Evidence** — the 5-way ablation, shock-robustness table, SHAP and weight-trajectory.
- **How it works** — architecture + the accuracy↔robustness dial.

Everything runs **client-side**: the gate is a 114-parameter forward pass ported to TypeScript
(`lib/regimegate.ts`, weights in `lib/gate.json`). No backend, no GPU, no data download.

## Run locally

```bash
cd web
npm install
npm run dev          # → http://localhost:3000
```

## Deploy free on Vercel (→ your Live Demo URL)

1. Push this repo to GitHub.
2. [vercel.com](https://vercel.com) → **Add New… → Project** → import the repo.
3. **Set the Root Directory to `web`** (this app lives in a subfolder). Vercel auto-detects Next.js.
4. **Deploy** → you get a public `https://<project>.vercel.app` URL. Paste it into the hackathon
   *Live Demo / Prototype URL* field.

> Static-export friendly (`output: "export"`), so it also deploys to Netlify, Cloudflare Pages, or
> GitHub Pages (`npm run build` → serve the `out/` folder).

## Structure
```
app/        layout.tsx · page.tsx · globals.css
components/  FusionChart.tsx (animated SVG) · ui.tsx
lib/         regimegate.ts (sim + gate) · gate.json (trained weights)
public/      architecture.png · shap.png · weight_trajectory.png
```
