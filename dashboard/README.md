# RegimeGate — Live Dashboard

An interactive demo of the **RegimeGate Adaptive Fusion Controller**: inject supply-chain shocks and
watch the gate re-allocate its fusion weights in real time, alongside the real Walmart **M5**
benchmark evidence.

* **Live Simulator** — pick a scenario (calm season, promo spike, Suez-style level shift, stockout
  drought), tune volatility/shock, and **press ▶ Play** to watch the forecast and the gate's fusion
  weights *animate day-by-day* (or scrub the day slider) as the gate shifts between the reactive
  *Tree* expert and the smooth *Deep* expert — with live accuracy vs. the fixed blend, the stacker,
  and each expert. Smooth eased transitions throughout.
* **Real M5 Evidence** — the 5-way ablation, the shock-robustness table, SHAP, and the weight
  trajectory from the prototype notebook.
* **How it works** — architecture and the accuracy↔robustness dial.

The gate runs as a **114-parameter NumPy forward pass** (pre-trained weights in `gate_weights.npz`),
so the app needs **no GPU, no PyTorch, and no data download** — it is tiny and fast to deploy.

---

## Run locally

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501.

## Deploy free (recommended: Streamlit Community Cloud)

1. Push this repo to GitHub (the `dashboard/` folder is self-contained).
2. Go to **share.streamlit.io** → *New app* → pick your repo/branch.
3. Set **Main file path** to `dashboard/app.py` and deploy.
   You get a public URL like `https://<app>.streamlit.app` — paste it into the hackathon
   *Live Demo / Prototype URL* field.

### Alternative: Hugging Face Spaces
Create a **Streamlit** Space, upload the contents of `dashboard/` (set `app.py` as the entry point),
and it builds automatically.

---

## Files
```
app.py             # the Streamlit dashboard
regime_sim.py      # self-contained simulator + NumPy gate forward pass
gate_weights.npz   # pre-trained gate (114 params)
requirements.txt   # streamlit, numpy, pandas, plotly  (no torch)
assets/            # architecture, SHAP, weight-trajectory figures
.streamlit/        # theme + server config
```

> The simulator is illustrative (a transparent toy that makes the mechanism visible); the rigorous
> evidence is the real-M5 prototype (`../RegimeGate_M5.ipynb`) and the solution document
> (`../docs/RegimeGate_Solution_Document.pdf`).
