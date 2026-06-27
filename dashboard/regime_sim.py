"""
regime_sim.py — the self-contained simulation core behind the live dashboard.

Design choice (best practice): the dashboard must run on a free Streamlit host with
no GPU, no PyTorch, no Kaggle download. So the heavy M5 work stays in the notebook;
here we keep a fast, transparent simulator and run the (tiny) gate as plain NumPy.

A non-stationary demand series is generated with an AR(1) deviation whose magnitude
(the "volatility") the judge controls. Two cheap experts have genuinely
regime-dependent competence — a smooth deep-style forecaster wins in calm periods,
a reactive tree-style forecaster wins in volatile/shocked periods — exactly the
trade-off RegimeGate is built to arbitrate.
"""
from __future__ import annotations
import numpy as np

WEEK = 7
FEATURES = ["roll_cv", "shock", "trend", "roll_std_log"]
EPS = 1e-6


def weekly_factor(t):
    return 1.0 + 0.12 * np.sin(2 * np.pi * np.asarray(t) / WEEK)


# ---------------------------------------------------------------- demand
def simulate(n=160, base=50.0, vol=0.40, rho=0.72, seed=0,
             shock=None, shock_t=None, shock_dur=12, shock_mag=2.6):
    """Return (t, y, sigma) for a non-stationary demand series.
    shock in {None,'spike','level','drought'} injected from shock_t."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wk = weekly_factor(t)
    sigma = np.full(n, float(vol))
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + sigma[i] * base * rng.normal()
    y = base * wk + e
    if shock and shock_t is not None:
        s = int(shock_t); en = min(n, s + int(shock_dur))
        if shock == "spike":
            y[s:en] = y[s:en] * shock_mag
        elif shock == "level":
            y[s:] = y[s:] * (1.0 + (shock_mag - 1.0) * 0.6)   # sustained shift
        elif shock == "drought":
            y[s:en] = y[s:en] * 0.05
    return t, np.clip(y, 0, None), sigma


# ---------------------------------------------------------------- experts (past-only, 1-step)
def experts(y):
    """Two regime-specialised experts, both forecasting y[t] from y[<t].

    Deep: seasonal + smooth level (deseasonalised EWMA x weekly factor) -> exploits the
          calendar structure, so it wins in calm periods.
    Tree: a reactive RAW recent average (no seasonal adjustment) -> tracks levels and
          shocks in ~1 step but misprices day-of-week, so it wins only when the level
          swings dominate the seasonality (volatile periods and shocks).
    """
    n = len(y); wk = weekly_factor(np.arange(n))
    deseas = y / wk
    deep_level = np.zeros(n); deep_level[0] = deseas[0]; a = 0.10
    tree = np.zeros(n); tree[0] = y[0]
    for i in range(1, n):
        deep_level[i] = a * deseas[i - 1] + (1 - a) * deep_level[i - 1]
        tree[i] = 0.6 * y[i - 1] + 0.4 * y[i - 2] if i >= 2 else y[i - 1]   # raw, reactive
    deep = deep_level * wk
    return np.clip(tree, 0, None), np.clip(deep, 0, None)


# ---------------------------------------------------------------- regime context (past-only)
def context(y, K=14):
    n = len(y); wk = weekly_factor(np.arange(n)); deseas = y / wk
    F = np.zeros((n, len(FEATURES)))
    for i in range(n):
        past = deseas[max(0, i - K):i]                # strictly before t
        if len(past) >= 3:
            m = past.mean(); s = past.std()
            F[i, 0] = s / (abs(m) + EPS)                                   # rolling CV
            F[i, 1] = np.clip((deseas[i - 1] - m) / (s + EPS), -6, 6)      # shock z-score
            recent = past[-min(7, len(past)):].mean()
            older = past[:max(1, len(past) - 7)].mean()
            F[i, 2] = (recent - older) / (abs(m) + EPS)                    # trend
            F[i, 3] = np.log1p(s)                                          # dispersion
    return F


# ---------------------------------------------------------------- the gate (NumPy forward pass)
def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def gate_weights(F, W):
    """Forward pass of the trained 2-layer gate. Returns weights [:,0]=tree, [:,1]=deep."""
    x = (F - W["mu"]) / W["sd"]
    x = np.clip(x, -6, 6)
    h = _gelu(x @ W["W1"] + W["b1"])
    logits = (h @ W["W2"] + W["b2"]) / W.get("temperature", 1.0)
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def load_gate(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- baselines + scoring
def fixed_blend(tree, deep, w_tree=0.6):
    return w_tree * tree + (1 - w_tree) * deep


def fit_stacker(tree, deep, y, train_frac=0.45):
    """Predictions-only least-squares blend (the contrast model), fit on a past slice."""
    k = max(8, int(len(y) * train_frac))
    X = np.column_stack([tree[:k], deep[:k], np.ones(k)])
    coef, *_ = np.linalg.lstsq(X, y[:k], rcond=None)
    return coef


def apply_stacker(tree, deep, coef):
    X = np.column_stack([tree, deep, np.ones(len(tree))])
    return np.clip(X @ coef, 0, None)


def rmse(y, p, mask=None):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if mask is not None:
        y, p = y[mask], p[mask]
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


def build_scene(vol=0.40, shock=None, shock_t=None, n=160, base=50.0, seed=0,
                shock_dur=12, shock_mag=2.6, gate=None, w_tree_fixed=0.6,
                warmup=20):
    """One call assembles everything the dashboard needs for a scenario."""
    t, y, sigma = simulate(n=n, base=base, vol=vol, seed=seed, shock=shock,
                           shock_t=shock_t, shock_dur=shock_dur, shock_mag=shock_mag)
    tree, deep = experts(y)
    F = context(y)
    out = {"t": t, "y": y, "sigma": sigma, "tree": tree, "deep": deep, "context": F}
    if gate is not None:
        W = gate_weights(F, gate)
        # temporal smoothing (matches the notebook's anti flip-flop guard)
        wt = W[:, 0].copy()
        wt = np.convolve(wt, np.ones(3) / 3, mode="same")
        wt = np.clip(0.9 * wt + 0.1 * 0.5, 0, 1)            # neutral floor (anti-collapse)
        out["w_tree"] = wt
        out["regime"] = wt * tree + (1 - wt) * deep
    out["fixed"] = fixed_blend(tree, deep, w_tree_fixed)
    coef = fit_stacker(tree, deep, y)
    out["stack"] = apply_stacker(tree, deep, coef)
    # score only after a warm-up so the experts have history
    m = np.zeros(n, bool); m[warmup:] = True
    out["mask"] = m
    out["scores"] = {
        "Tree (LightGBM-style)": rmse(y, tree, m),
        "Deep (N-HiTS-style)": rmse(y, deep, m),
        "Fixed 60/40": rmse(y, out["fixed"], m),
        "Stacking": rmse(y, out["stack"], m),
    }
    if gate is not None:
        out["scores"]["RegimeGate"] = rmse(y, out["regime"], m)
    return out
