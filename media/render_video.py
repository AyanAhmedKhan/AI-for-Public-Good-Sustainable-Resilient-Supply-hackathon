"""
render_video.py — renders the RegimeGate ~100s explainer to an MP4.

Self-contained: uses the SAME trained gate (dashboard/gate_weights.npz) and the
same NumPy simulator (dashboard/regime_sim.py) that the live dashboard runs, so the
centerpiece "watch the gate re-allocate" sequence is the real forward pass — not a
mock-up. No narration audio; on-screen captions carry the story (see media/script.md
for the voiceover track if you record over it).

Output : media/RegimeGate_demo.mp4   (1920x1080, 30 fps, ~100s)
Run     : python media/render_video.py
"""
from __future__ import annotations
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge
from matplotlib.collections import LineCollection
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "dashboard"))
import regime_sim as R  # noqa: E402

# ----------------------------------------------------------------- config
W, H, DPI, FPS = 1920, 1080, 100, 30
OUT = os.path.join(HERE, "RegimeGate_demo.mp4")

BG      = "#0B0E14"   # stage
PANEL   = "#131826"   # cards / plot panels
PANEL2  = "#1B2233"
INK     = "#EAEEF6"   # primary text
MUTE    = "#8A93A8"   # secondary text
FAINT   = "#39435C"   # grid / hairlines
TREE    = "#F5A524"   # reactive expert (LightGBM)
DEEP    = "#4C9AFF"   # smooth expert  (N-HiTS)
GATE    = "#27D796"   # RegimeGate
FIXED   = "#7C8696"   # fixed blend
STACK   = "#FF5C72"   # stacker (collapses)
ACTUAL  = "#D6DEEE"   # observed demand
GOLD    = "#FFCD4B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "text.color": INK,
    "axes.edgecolor": FAINT,
    "axes.labelcolor": MUTE,
    "xtick.color": MUTE,
    "ytick.color": MUTE,
})

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)

# ----------------------------------------------------------------- helpers
def clamp(t):
    return 0.0 if t < 0 else (1.0 if t > 1 else t)

def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)

def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3

def stage():
    """Reset the figure and return a full-bleed 0..1 axis to draw on."""
    fig.clf()
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)
    return ax

def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

def panel(ax, x, y, w, h, fc=PANEL, ec=FAINT, lw=1.4, rad=0.018, alpha=1.0, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z,
                       transform=ax.transAxes, mutation_aspect=W / H)
    ax.add_patch(p)
    return p

def text(ax, x, y, s, size=18, color=INK, weight="normal", ha="left", va="center",
         alpha=1.0, family="DejaVu Sans", style="normal", z=5):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va,
                   alpha=alpha, family=family, style=style, zorder=z,
                   transform=ax.transAxes)

def wordmark(ax, label=""):
    """Persistent footer: brand left, scene label right."""
    text(ax, 0.045, 0.052, "RegimeGate", size=17, color=INK, weight="bold", alpha=0.9)
    text(ax, 0.165, 0.052, "· Adaptive Fusion Controller", size=15, color=MUTE, alpha=0.8)
    if label:
        text(ax, 0.955, 0.052, label, size=15, color=MUTE, ha="right", alpha=0.85)
    ax.plot([0.045, 0.955], [0.085, 0.085], color=FAINT, lw=1.0, alpha=0.6,
            transform=ax.transAxes, zorder=2)

def data_axes(rect, ylim=None, xlim=None, grid=True):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(FAINT); s.set_linewidth(1.0)
    ax.tick_params(colors=MUTE, labelsize=13, length=0)
    if grid:
        ax.grid(True, color=FAINT, lw=0.7, alpha=0.35)
    if ylim: ax.set_ylim(*ylim)
    if xlim: ax.set_xlim(*xlim)
    return ax

# ----------------------------------------------------------------- shared data (the REAL gate + sim)
GATE_W = R.load_gate(os.path.join(ROOT, "dashboard", "gate_weights.npz"))
N      = 170
SHOCK_T = 85
SCENE = R.build_scene(vol=0.40, shock="level", shock_t=SHOCK_T, n=N, seed=15,
                      gate=GATE_W, shock_dur=14, shock_mag=2.6)
t   = SCENE["t"]
y   = SCENE["y"]
tree = SCENE["tree"]; deep = SCENE["deep"]
wfit = SCENE["w_tree"]; regime = SCENE["regime"]
fixed = SCENE["fixed"]; stack = SCENE["stack"]
SCORES = SCENE["scores"]
YMAX = float(np.nanmax(y) * 1.12)
WARM = 20

# ===================================================================== SCENES
def scene_title():
    """~7s — brand reveal."""
    NF = int(3.0 * FPS)
    for f in range(NF):
        ax = stage()
        p = f / NF
        # accent rule grows
        rule = smooth((p - 0.15) / 0.6)
        ax.plot([0.20, 0.20 + 0.60 * rule], [0.615, 0.615], color=GATE, lw=3,
                transform=ax.transAxes, solid_capstyle="round", zorder=3)
        text(ax, 0.20, 0.72, "RegimeGate", size=86, weight="bold",
             alpha=smooth(p / 0.45))
        text(ax, 0.205, 0.555,
             "An Adaptive Fusion Controller for Non-Stationary Demand",
             size=30, color=INK, alpha=smooth((p - 0.25) / 0.5))
        text(ax, 0.205, 0.475,
             "A < 50k-parameter gate that learns WHERE each forecaster should be trusted.",
             size=21, color=MUTE, alpha=smooth((p - 0.45) / 0.45))
        text(ax, 0.205, 0.30,
             "AI for Public Good  ·  Sustainable & Resilient Supply Chains",
             size=18, color=MUTE, alpha=smooth((p - 0.6) / 0.4))
        text(ax, 0.205, 0.255,
             "Problem Statement 3 · Round 2 · Team Absolute — Ayan Ahmed Khan",
             size=17, color=GATE, alpha=smooth((p - 0.7) / 0.3))
        yield grab()
    hold = grab()
    for _ in range(int(3.6 * FPS)):
        yield hold

def scene_problem():
    """~14s — demand is non-stationary; experts swap competence."""
    intro = int(0.8 * FPS)
    draw = int(7.0 * FPS)
    NF = intro + draw
    ax_layout = None
    for f in range(NF):
        ax = stage()
        wordmark(ax, "01 · The problem")
        text(ax, 0.06, 0.90, "Demand is non-stationary.", size=44, weight="bold",
             alpha=smooth(f / intro))
        text(ax, 0.06, 0.835,
             "The same product is calm for weeks — then a disruption rewrites the series.",
             size=21, color=MUTE, alpha=smooth((f - 0.2 * FPS) / intro))
        dax = data_axes([0.06, 0.20, 0.88, 0.56], ylim=(0, YMAX), xlim=(0, N))
        dax.set_xlabel("day", color=MUTE, fontsize=14)
        # regime shading
        dax.axvspan(0, SHOCK_T, color=DEEP, alpha=0.05)
        dax.axvspan(SHOCK_T, N, color=TREE, alpha=0.06)
        p = ease_out((f - intro) / draw)
        c = int(WARM + (N - WARM) * p)
        c = max(WARM + 1, min(N, c))
        dax.plot(t[:c], y[:c], color=ACTUAL, lw=2.6, zorder=5)
        # faint experts trailing the reveal
        dax.plot(t[:c], deep[:c], color=DEEP, lw=1.5, alpha=0.5, zorder=3)
        dax.plot(t[:c], tree[:c], color=TREE, lw=1.5, alpha=0.5, zorder=3)
        if c > SHOCK_T:
            dax.axvline(SHOCK_T, color=GOLD, lw=1.6, ls=(0, (4, 3)), alpha=0.8)
            dax.annotate("level shift", (SHOCK_T, YMAX * 0.93), color=GOLD,
                         fontsize=15, ha="left", va="top", xytext=(SHOCK_T + 2, YMAX * 0.95))
        # legend chips (above the plot, not over it)
        text(ax, 0.06, 0.788, "—  observed demand", size=15, color=ACTUAL)
        text(ax, 0.27, 0.788, "—  Deep (N-HiTS) · wins when calm", size=15, color=DEEP)
        text(ax, 0.60, 0.788, "—  Tree (LightGBM) · wins when turbulent", size=15, color=TREE)
        # caption appears late
        ca = smooth((f - intro - 0.55 * draw) / (0.3 * draw))
        if ca > 0.01:
            panel(ax, 0.06, 0.115, 0.88, 0.065, fc=PANEL2, ec=FAINT, alpha=ca, z=6)
            text(ax, 0.075, 0.1475,
                 "No single forecaster is best across regimes — and which one wins keeps changing.",
                 size=20, color=INK, alpha=ca, z=7)
        yield grab()
    hold = grab()
    for _ in range(int(3.2 * FPS)):
        yield hold

def scene_fixes():
    """~13s — fixed blend & stacker both assume static competence."""
    intro = int(0.7 * FPS)
    body = int(6.2 * FPS)
    NF = intro + body
    coef = R.fit_stacker(tree, deep, y)
    for f in range(NF):
        ax = stage()
        wordmark(ax, "02 · Why the usual fixes fail")
        text(ax, 0.06, 0.90, "A fixed blend and a stacked meta-learner",
             size=40, weight="bold", alpha=smooth(f / intro))
        text(ax, 0.06, 0.84, "both assume each model's relative competence never changes.",
             size=23, color=MUTE, alpha=smooth((f - 0.15 * FPS) / intro))
        dax = data_axes([0.06, 0.205, 0.62, 0.55], ylim=(0, YMAX), xlim=(0, N))
        dax.set_xlabel("day", color=MUTE, fontsize=14)
        dax.axvline(SHOCK_T, color=GOLD, lw=1.5, ls=(0, (4, 3)), alpha=0.7)
        p = ease_out((f - intro) / body)
        c = max(WARM + 1, min(N, int(WARM + (N - WARM) * p)))
        dax.plot(t[:c], y[:c], color=ACTUAL, lw=2.2, alpha=0.85, zorder=5, label="demand")
        dax.plot(t[:c], fixed[:c], color=FIXED, lw=2.0, zorder=4, label="Fixed 60/40")
        dax.plot(t[:c], stack[:c], color=STACK, lw=2.4, zorder=6, label="Stacking")
        if c > SHOCK_T + 4:
            dax.annotate("stacker mis-tracks\nthe new regime",
                         (SHOCK_T + 18, stack[min(N - 1, SHOCK_T + 18)]),
                         color=STACK, fontsize=14, ha="left", va="center",
                         xytext=(SHOCK_T + 24, YMAX * 0.78),
                         arrowprops=dict(arrowstyle="->", color=STACK, lw=1.4))
        dax.legend(loc="upper left", fontsize=12, framealpha=0.0, labelcolor=INK)
        # right column: the verdict cards
        ra = smooth((f - intro - 0.35 * body) / (0.3 * body))
        if ra > 0.01:
            panel(ax, 0.71, 0.50, 0.23, 0.26, fc=PANEL, ec=STACK, lw=1.6, alpha=ra)
            text(ax, 0.725, 0.715, "Stacking", size=22, weight="bold", color=STACK, alpha=ra)
            text(ax, 0.725, 0.655, "predictions-only blend", size=14, color=MUTE, alpha=ra)
            text(ax, 0.725, 0.585, "0.870", size=40, weight="bold", color=STACK, alpha=ra)
            text(ax, 0.86, 0.585, "WRMSSE", size=14, color=MUTE, alpha=ra, va="center")
            text(ax, 0.725, 0.535, "1.125 on intermittent demand", size=14, color=STACK, alpha=ra)
            panel(ax, 0.71, 0.205, 0.23, 0.26, fc=PANEL, ec=FIXED, lw=1.4, alpha=ra)
            text(ax, 0.725, 0.42, "Fixed 60/40", size=22, weight="bold", color=FIXED, alpha=ra)
            text(ax, 0.725, 0.36, "one global weight", size=14, color=MUTE, alpha=ra)
            text(ax, 0.725, 0.29, "0.736", size=40, weight="bold", color=INK, alpha=ra)
            text(ax, 0.86, 0.29, "WRMSSE", size=14, color=MUTE, alpha=ra, va="center")
            text(ax, 0.725, 0.24, "never adapts to the regime", size=14, color=MUTE, alpha=ra)
        cap = smooth((f - intro - 0.6 * body) / (0.3 * body))
        if cap > 0.01:
            text(ax, 0.06, 0.155, "One global blend cannot serve two opposite regimes.",
                 size=20, color=INK, alpha=cap)
        yield grab()
    hold = grab()
    for _ in range(int(3.0 * FPS)):
        yield hold

def scene_idea():
    """~12s — the architecture."""
    intro = int(0.7 * FPS)
    NF = intro + int(2.5 * FPS)
    arch = None
    apath = os.path.join(ROOT, "docs", "figs", "architecture.png")
    if os.path.exists(apath):
        arch = imageio.imread(apath)
    for f in range(NF):
        ax = stage()
        wordmark(ax, "03 · The idea")
        text(ax, 0.06, 0.90, "RegimeGate learns WHERE each expert should be trusted.",
             size=38, weight="bold", alpha=smooth(f / intro))
        text(ax, 0.06, 0.84,
             "Two frozen experts. A tiny gate reads the regime and emits fusion weights — per step.",
             size=21, color=MUTE, alpha=smooth((f - 0.15 * FPS) / intro))
        a = smooth((f - 0.2 * FPS) / (1.2 * FPS))
        if arch is not None:
            iax = fig.add_axes([0.10, 0.205, 0.62, 0.56])
            iax.imshow(arch); iax.axis("off")
            iax.set_alpha(a)
            for sp in iax.spines.values():
                sp.set_visible(False)
            iax.patch.set_alpha(0)
        # right: three bullets build in
        bullets = [
            (GATE, "conditioned on the regime,", "not on the predictions"),
            (DEEP, "< 50k parameters,", "frozen experts stay untouched"),
            (TREE, "anti-fragility guards", "never worse than the static blend"),
        ]
        for i, (col, b1, b2) in enumerate(bullets):
            ba = smooth((f - intro - (0.25 + i * 0.25) * FPS) / (0.5 * FPS))
            yb = 0.62 - i * 0.135
            if ba > 0.01:
                ax.add_patch(Circle((0.755, yb + 0.018), 0.008, color=col, alpha=ba,
                                    transform=ax.transAxes, zorder=6))
                text(ax, 0.775, yb + 0.018, b1, size=20, weight="bold", color=INK, alpha=ba)
                text(ax, 0.775, yb - 0.022, b2, size=16, color=MUTE, alpha=ba)
        cap = smooth((f - intro - 1.4 * FPS) / (0.6 * FPS))
        if cap > 0.01:
            text(ax, 0.06, 0.135,
                 "That conditioning is what places it provably beyond a stacked meta-learner.",
                 size=20, color=GATE, alpha=cap)
        yield grab()
    hold = grab()
    for _ in range(int(2.8 * FPS)):
        yield hold

def _mech_left(c):
    """Draw the two left-column plots up to index c; return (tax, gax, now)."""
    now = min(N - 1, max(WARM, c - 1))
    tax = data_axes([0.06, 0.47, 0.60, 0.38], ylim=(0, YMAX), xlim=(0, N))
    tax.axvspan(SHOCK_T, N, color=TREE, alpha=0.05)
    tax.plot(t[:c], y[:c], color=ACTUAL, lw=2.4, zorder=6)
    tax.plot(t[:c], deep[:c], color=DEEP, lw=1.4, alpha=0.55, zorder=3)
    tax.plot(t[:c], tree[:c], color=TREE, lw=1.4, alpha=0.55, zorder=3)
    tax.plot(t[:c], regime[:c], color=GATE, lw=2.6, zorder=7)
    tax.axvline(now, color=INK, lw=1.2, alpha=0.45)
    if c > SHOCK_T:
        tax.axvline(SHOCK_T, color=GOLD, lw=1.5, ls=(0, (4, 3)), alpha=0.8)
    gax = data_axes([0.06, 0.165, 0.60, 0.255], ylim=(0, 1), xlim=(0, N), grid=False)
    gax.set_ylabel("fusion weight", color=MUTE, fontsize=13)
    xs = t[:c]; wt = wfit[:c]
    gax.fill_between(xs, 0, wt, color=TREE, alpha=0.55, zorder=2)
    gax.fill_between(xs, wt, 1, color=DEEP, alpha=0.45, zorder=2)
    gax.plot(xs, wt, color=INK, lw=1.6, zorder=4)
    gax.axhline(0.5, color=INK, lw=0.8, ls=":", alpha=0.4)
    if c > SHOCK_T:
        gax.axvline(SHOCK_T, color=GOLD, lw=1.5, ls=(0, (4, 3)), alpha=0.8)
    gax.scatter([now], [wfit[now]], s=55, color=INK, zorder=6, edgecolor=BG, linewidth=1.5)
    return tax, gax, now

def _mech_legend(ax):
    text(ax, 0.06, 0.875, "— demand", size=14, color=ACTUAL)
    text(ax, 0.17, 0.875, "— Deep", size=14, color=DEEP)
    text(ax, 0.26, 0.875, "— Tree", size=14, color=TREE)
    text(ax, 0.35, 0.875, "— RegimeGate fusion", size=14, color=GATE, weight="bold")
    text(ax, 0.075, 0.355, "Tree share", size=12, color=TREE, weight="bold")
    text(ax, 0.075, 0.205, "Deep share", size=12, color=DEEP, weight="bold")

def _mech_readout(ax, now, alpha=1.0):
    """Right-hand live gate-weight readout (kept clear of the plots)."""
    wt = float(wfit[now])
    lead_tree = wt >= 0.5
    lcol = TREE if lead_tree else DEEP
    lead = "Tree" if lead_tree else "Deep"
    sub = "reactive expert" if lead_tree else "smooth expert"
    panel(ax, 0.69, 0.165, 0.25, 0.685, fc=PANEL, ec=FAINT, lw=1.4, alpha=min(1, alpha + 0.15))
    text(ax, 0.705, 0.805, "LIVE FUSION MIX", size=15, color=MUTE, alpha=alpha)
    text(ax, 0.705, 0.76, f"day {now}/{N - 1}", size=14, color=MUTE, alpha=alpha)
    # big indicator
    panel(ax, 0.705, 0.60, 0.22, 0.115, fc=PANEL2, ec=lcol, lw=1.8, alpha=alpha)
    text(ax, 0.72, 0.685, "gate leans →", size=15, color=MUTE, alpha=alpha)
    text(ax, 0.72, 0.635, lead, size=34, color=lcol, weight="bold", alpha=alpha)
    text(ax, 0.905, 0.635, f"{max(wt, 1 - wt):.0%}", size=30, color=INK,
         weight="bold", ha="right", alpha=alpha)
    text(ax, 0.72, 0.585, sub, size=13, color=MUTE, alpha=alpha, va="top")
    # split bar
    bx, bw, by, bh = 0.705, 0.22, 0.45, 0.05
    ax.add_patch(FancyBboxPatch((bx, by), bw * wt, bh, boxstyle="round,pad=0,rounding_size=0.004",
                 fc=TREE, ec="none", alpha=alpha, transform=ax.transAxes,
                 mutation_aspect=W / H, zorder=5))
    ax.add_patch(FancyBboxPatch((bx + bw * wt, by), bw * (1 - wt), bh,
                 boxstyle="round,pad=0,rounding_size=0.004", fc=DEEP, ec="none",
                 alpha=alpha, transform=ax.transAxes, mutation_aspect=W / H, zorder=5))
    text(ax, bx, by - 0.03, f"Tree {wt:.0%}", size=14, color=TREE, weight="bold", alpha=alpha)
    text(ax, bx + bw, by - 0.03, f"Deep {1 - wt:.0%}", size=14, color=DEEP,
         weight="bold", ha="right", alpha=alpha)
    # regime tag
    if now < SHOCK_T:
        tag, tc = "regime: CALM", DEEP
    elif now < SHOCK_T + 12:
        tag, tc = "regime: SHOCK", TREE
    else:
        tag, tc = "regime: SHIFTED", GATE
    text(ax, 0.705, 0.33, tag, size=17, color=tc, weight="bold", alpha=alpha)
    text(ax, 0.705, 0.285, "114-param NumPy forward pass", size=13, color=MUTE, alpha=alpha)
    text(ax, 0.705, 0.245, "— the same gate the dashboard runs.", size=13, color=MUTE, alpha=alpha)

def scene_mechanism():
    """~28s — centerpiece: the real gate re-allocating live, then the verdict."""
    intro = int(0.8 * FPS)
    run = int(19.0 * FPS)
    NF = intro + run
    for f in range(NF):
        ax = stage()
        wordmark(ax, "04 · Watch the gate re-allocate — live")
        text(ax, 0.06, 0.93, "The same trained gate, running live.",
             size=36, weight="bold", alpha=smooth(f / intro))
        p = clamp((f - intro) / run)          # LINEAR day pacing — calm gets fair time
        c = max(WARM + 1, min(N, int(WARM + (N - WARM) * p)))
        _mech_left(c)
        _mech_legend(ax)
        now = min(N - 1, c - 1)
        _mech_readout(ax, now)
        # dynamic caption keyed to the shock
        if now < SHOCK_T - 3:
            msg, mc = "Calm regime → the gate leans on the smooth Deep expert.", DEEP
        elif now < SHOCK_T + 12:
            msg, mc = "Shock! → in ~1 step the gate swings to the reactive Tree expert.", TREE
        else:
            msg, mc = "New regime settled → the gate holds its reallocated mix.", GATE
        text(ax, 0.06, 0.115, msg, size=21, color=mc, weight="bold")
        yield grab()
    # short hold on the completed run
    hold = grab()
    for _ in range(int(1.4 * FPS)):
        yield hold
    # ---- verdict: freeze the run and reveal the scoreboard ----
    order = [("RegimeGate", GATE), ("Tree (LightGBM-style)", TREE),
             ("Deep (N-HiTS-style)", DEEP), ("Fixed 60/40", FIXED), ("Stacking", STACK)]
    nm = {"RegimeGate": "RegimeGate", "Tree (LightGBM-style)": "Tree expert",
          "Deep (N-HiTS-style)": "Deep expert", "Fixed 60/40": "Fixed 60/40",
          "Stacking": "Stacking"}
    best = min(SCORES.values())
    rev = int(1.6 * FPS)
    for f in range(rev + int(4.8 * FPS)):
        ax = stage()
        wordmark(ax, "04 · Watch the gate re-allocate — live")
        _mech_left(N)
        _mech_legend(ax)
        text(ax, 0.06, 0.93, "Result on this run: the gate's fusion wins.",
             size=34, weight="bold")
        a = smooth(f / rev)
        panel(ax, 0.69, 0.165, 0.25, 0.685, fc=PANEL, ec=FAINT, alpha=min(1, a + 0.2))
        text(ax, 0.705, 0.805, "Error on this scenario (RMSE)", size=15, color=MUTE, alpha=a)
        for i, (k, col) in enumerate(order):
            ay = smooth((f - i * 0.12 * FPS) / rev)
            if ay < 0.01:
                continue
            yb = 0.725 - i * 0.108
            v = SCORES[k]
            isbest = abs(v - best) < 1e-6
            if isbest:
                panel(ax, 0.70, yb - 0.042, 0.225, 0.088, fc=PANEL2, ec=GATE, lw=1.6, alpha=ay)
            text(ax, 0.715, yb, nm[k], size=18, color=col,
                 weight="bold" if isbest else "normal", alpha=ay)
            text(ax, 0.915, yb, f"{v:.1f}", size=23 if isbest else 19, color=INK,
                 weight="bold" if isbest else "normal", ha="right", alpha=ay)
        cap = smooth((f - rev) / (0.8 * FPS))
        if cap > 0.01:
            text(ax, 0.06, 0.115,
                 "Stacking collapses on the regime change; RegimeGate stays best.",
                 size=20, color=GATE, alpha=cap)
        yield grab()

def scene_results():
    """~16s — real Walmart M5 evidence."""
    variants = [
        ("N-HiTS (DL)",      0.760, DEEP,  False),
        ("LightGBM (ML)",    0.781, TREE,  False),
        ("Fixed 60/40",      0.736, FIXED, False),
        ("Stacking",         0.870, STACK, False),
        ("RegimeGate",       0.734, GATE,  True),
    ]
    vmax = 0.92
    intro = int(0.7 * FPS)
    grow = int(2.2 * FPS)
    NF = intro + grow
    for f in range(NF):
        ax = stage()
        wordmark(ax, "05 · Real Walmart M5 evidence")
        text(ax, 0.06, 0.90, "Real Walmart M5 — leakage-safe, rolling-origin.",
             size=40, weight="bold", alpha=smooth(f / intro))
        text(ax, 0.06, 0.84, "Overall WRMSSE  (lower is better)",
             size=21, color=MUTE, alpha=smooth((f - 0.15 * FPS) / intro))
        x0, x1 = 0.30, 0.84
        for i, (name, val, col, best) in enumerate(variants):
            yb = 0.72 - i * 0.115
            gp = ease_out((f - intro - i * 0.12 * FPS) / grow)
            wbar = (x1 - x0) * (val / vmax) * gp
            text(ax, 0.28, yb, name, size=20, color=col, ha="right",
                 weight="bold" if best else "normal")
            ax.add_patch(FancyBboxPatch((x0, yb - 0.028), max(wbar, 1e-4), 0.056,
                         boxstyle="round,pad=0,rounding_size=0.006",
                         fc=col, ec="none", alpha=0.92 if best else 0.7,
                         transform=ax.transAxes, mutation_aspect=W / H, zorder=4))
            if gp > 0.6:
                text(ax, x0 + wbar + 0.012, yb, f"{val:.3f}", size=19, color=INK,
                     weight="bold" if best else "normal", alpha=smooth((gp - 0.6) / 0.4))
            if best and gp > 0.7:
                text(ax, x0 + wbar + 0.075, yb, "best overall", size=15, color=GATE,
                     alpha=smooth((gp - 0.7) / 0.3))
        yield grab()
    base = grab()
    # callouts build over a hold
    for f in range(int(7.2 * FPS)):
        ax = stage()
        wordmark(ax, "05 · Real Walmart M5 evidence")
        text(ax, 0.06, 0.90, "Real Walmart M5 — leakage-safe, rolling-origin.",
             size=40, weight="bold")
        text(ax, 0.06, 0.84, "Overall WRMSSE  (lower is better)", size=21, color=MUTE)
        x0, x1 = 0.30, 0.84
        for i, (name, val, col, best) in enumerate(variants):
            yb = 0.72 - i * 0.115
            wbar = (x1 - x0) * (val / vmax)
            text(ax, 0.28, yb, name, size=20, color=col, ha="right",
                 weight="bold" if best else "normal")
            ax.add_patch(FancyBboxPatch((x0, yb - 0.028), wbar, 0.056,
                         boxstyle="round,pad=0,rounding_size=0.006",
                         fc=col, ec="none", alpha=0.92 if best else 0.7,
                         transform=ax.transAxes, mutation_aspect=W / H, zorder=4))
            text(ax, x0 + wbar + 0.012, yb, f"{val:.3f}", size=19, color=INK,
                 weight="bold" if best else "normal")
            if best:
                text(ax, x0 + wbar + 0.075, yb, "best overall", size=15, color=GATE)
        a1 = smooth(f / (0.7 * FPS))
        if a1 > 0.01:
            panel(ax, 0.06, 0.135, 0.40, 0.085, fc=PANEL2, ec=GATE, lw=1.6, alpha=a1)
            text(ax, 0.075, 0.178, "+15.5%", size=30, weight="bold", color=GATE, alpha=a1)
            text(ax, 0.20, 0.187, "better than the", size=16, color=MUTE, alpha=a1)
            text(ax, 0.20, 0.158, "stacked meta-learner", size=16, color=INK, alpha=a1)
        a2 = smooth((f - 0.9 * FPS) / (0.7 * FPS))
        if a2 > 0.01:
            panel(ax, 0.50, 0.135, 0.44, 0.085, fc=PANEL2, ec=FAINT, lw=1.3, alpha=a2)
            text(ax, 0.515, 0.187, "Two-sided specialization", size=16, color=INK,
                 weight="bold", alpha=a2)
            text(ax, 0.515, 0.158, "Deep wins smooth 0.604  ·  Tree wins intermittent 0.828",
                 size=15, color=MUTE, alpha=a2)
        yield grab()

def scene_dial():
    """~10s — accuracy <-> robustness dial."""
    intro = int(0.7 * FPS)
    sweep = int(4.5 * FPS)
    NF = intro + sweep
    cx, cy, rad = 0.32, 0.46, 0.20
    for f in range(NF):
        ax = stage()
        wordmark(ax, "06 · The accuracy ↔ robustness dial")
        text(ax, 0.06, 0.90, "One tunable dial: accuracy ↔ robustness.",
             size=40, weight="bold", alpha=smooth(f / intro))
        text(ax, 0.06, 0.84,
             "The same gate, two operating points — set by one regime-balance knob.",
             size=21, color=MUTE, alpha=smooth((f - 0.15 * FPS) / intro))
        asp = W / H
        def gpt(ang, r):
            return (cx + r * math.cos(ang) / asp, cy + r * math.sin(ang))
        # arc track 210°..-30°
        a_lo, a_hi = math.radians(210), math.radians(-30)
        ang = np.linspace(a_lo, a_hi, 120)
        ax.plot([cx + rad * math.cos(a) / asp for a in ang],
                [cy + rad * math.sin(a) for a in ang],
                color=FAINT, lw=10, solid_capstyle="round", zorder=2)
        # colored progress
        prog = ease_out((f - intro) / sweep)
        # ping-pong: go to robustness then settle mid-high
        sval = 0.5 + 0.45 * math.sin(prog * math.pi)  # 0.5 -> 0.95 -> 0.5
        sval = max(0.05, min(0.95, sval))
        ai = a_lo + (a_hi - a_lo) * sval
        angp = np.linspace(a_lo, ai, 60)
        cols = [DEEP, GATE, TREE]
        ax.plot([cx + rad * math.cos(a) / asp for a in angp],
                [cy + rad * math.sin(a) for a in angp],
                color=GATE, lw=10, solid_capstyle="round", zorder=3)
        # needle
        nx, ny = gpt(ai, rad - 0.02)
        ax.plot([cx, nx], [cy, ny], color=INK, lw=4, solid_capstyle="round", zorder=5)
        ax.add_patch(Circle((cx, cy), 0.016, color=INK, transform=ax.transAxes, zorder=6))
        lx, ly = gpt(a_lo, rad + 0.06)
        rx, ry = gpt(a_hi, rad + 0.06)
        text(ax, lx - 0.02, ly - 0.03, "accuracy", size=18, color=DEEP, ha="center")
        text(ax, rx + 0.02, ry - 0.03, "robustness", size=18, color=TREE, ha="center")
        mode = "robustness" if sval > 0.62 else ("accuracy" if sval < 0.42 else "balanced")
        text(ax, cx, cy - 0.10, mode, size=22, color=GATE, weight="bold", ha="center")
        # right: guarantees
        items = [
            "Robustness setting matches the most",
            "shock-robust expert — on every shock.",
        ]
        guards = ["weight smoothing", "fixed-weight floor",
                  "confidence fallback", "shock-aware training"]
        a3 = smooth((f - intro - 0.3 * FPS) / FPS)
        if a3 > 0.01:
            text(ax, 0.60, 0.62, "Anti-fragility, by construction", size=24,
                 weight="bold", color=INK, alpha=a3)
            for i, g in enumerate(guards):
                ga = smooth((f - intro - (0.5 + i * 0.2) * FPS) / (0.5 * FPS))
                if ga > 0.01:
                    ax.add_patch(Circle((0.615, 0.545 - i * 0.06), 0.007, color=GATE,
                                 transform=ax.transAxes, alpha=ga, zorder=6))
                    text(ax, 0.635, 0.545 - i * 0.06, g, size=19, color=MUTE, alpha=ga)
            text(ax, 0.60, 0.27, "Never meaningfully worse than the static blend.",
                 size=19, color=GATE, alpha=smooth((f - intro - 1.6 * FPS) / FPS))
        yield grab()
    hold = grab()
    for _ in range(int(2.5 * FPS)):
        yield hold

def scene_close():
    """~8s — wordmark + where to find it."""
    intro = int(1.2 * FPS)
    NF = intro + int(4.6 * FPS)
    for f in range(NF):
        ax = stage()
        p = f / NF
        rule = smooth((f - 0.3 * FPS) / FPS)
        ax.plot([0.20, 0.20 + 0.30 * rule], [0.665, 0.665], color=GATE, lw=3,
                transform=ax.transAxes, solid_capstyle="round")
        text(ax, 0.20, 0.745, "RegimeGate", size=70, weight="bold",
             alpha=smooth(f / intro))
        text(ax, 0.205, 0.62, "Beyond a stacked meta-learner.  < 50k parameters.",
             size=26, color=INK, alpha=smooth((f - 0.3 * FPS) / intro))
        chips = [
            (DEEP, "Live dashboard", "interactive Streamlit demo"),
            (TREE, "Colab notebook", "full M5 prototype, run-all"),
            (GATE, "Solution document", "the Round-2 detailed write-up"),
        ]
        for i, (col, a, b) in enumerate(chips):
            ca = smooth((f - 0.6 * FPS - i * 0.25 * FPS) / (0.6 * FPS))
            yb = 0.46 - i * 0.075
            if ca > 0.01:
                ax.add_patch(Circle((0.215, yb + 0.012), 0.007, color=col,
                             transform=ax.transAxes, alpha=ca, zorder=6))
                text(ax, 0.235, yb + 0.012, a, size=22, weight="bold", color=INK, alpha=ca)
                text(ax, 0.43, yb + 0.012, b, size=18, color=MUTE, alpha=ca)
        text(ax, 0.205, 0.16,
             "Team Absolute — Ayan Ahmed Khan   ·   AI for Public Good · PS 3 · MIT-licensed",
             size=18, color=MUTE, alpha=smooth((f - 1.4 * FPS) / FPS))
        yield grab()
    hold = grab()
    for _ in range(int(1.6 * FPS)):
        yield hold

# ----------------------------------------------------------------- director
class Director:
    def __init__(self, writer):
        self.w = writer
        self.prev = None
        self.count = 0

    def _put(self, fr):
        self.w.append_data(fr)
        self.count += 1

    def _xfade(self, a, b, n):
        a = a.astype(np.float32); b = b.astype(np.float32)
        for i in range(1, n + 1):
            tt = smooth(i / n)
            self._put((a * (1 - tt) + b * tt).astype(np.uint8))

    def play(self, gen, fade=0.4):
        it = iter(gen)
        try:
            first = next(it)
        except StopIteration:
            return
        nf = int(fade * FPS)
        if nf > 0:
            src = self.prev if self.prev is not None else np.zeros_like(first)
            self._xfade(src, first, nf)
        self._put(first)
        last = first
        for fr in it:
            self._put(fr); last = fr
        self.prev = last

    def fade_out(self, sec=0.8):
        if self.prev is None:
            return
        self._xfade(self.prev, np.zeros_like(self.prev), int(sec * FPS))

# ----------------------------------------------------------------- run
def main():
    print(f"[render] writing {OUT}")
    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=8, pixelformat="yuv420p",
                                ffmpeg_log_level="error")
    d = Director(writer)
    scenes = [
        ("title",     scene_title,     0.0),
        ("problem",   scene_problem,   0.5),
        ("fixes",     scene_fixes,     0.5),
        ("idea",      scene_idea,      0.5),
        ("mechanism", scene_mechanism, 0.5),
        ("results",   scene_results,   0.5),
        ("dial",      scene_dial,      0.5),
        ("close",     scene_close,     0.5),
    ]
    for name, fn, fade in scenes:
        d.play(fn(), fade=fade)
        print(f"  · {name:10s} done  (total frames so far: {d.count}, "
              f"{d.count / FPS:5.1f}s)")
    d.fade_out(0.9)
    writer.close()
    print(f"[render] done — {d.count} frames = {d.count / FPS:.1f}s -> {OUT}")

if __name__ == "__main__":
    main()
