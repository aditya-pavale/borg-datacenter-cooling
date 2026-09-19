"""Generates docs/presentations/BORG_Datacenter_Cooling_V2_Pilot_Presentation.pptx
-- a faculty-facing "V2 pilot / development / research progress" deck.

Every number and chart in this deck is read from the repository's own
result files (results/pilot/*, results/model_selection_history.csv,
docs/*.md) -- nothing is invented. Where a requested visual would need
data that does not exist yet (e.g. a full 8-cell result), the slide
says so explicitly rather than fabricating a placeholder chart.

Run:
    .venv/bin/python scripts/generate_pilot_presentation.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ASSETS_DIR = Path("/tmp/claude-1000/-home-aditya-Documents-r-l-project-borg-datacenter-cooling/6892cf43-52b5-4756-a2d3-9c7e5c835322/scratchpad/ppt_assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO_ROOT / "docs" / "presentations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "BORG_Datacenter_Cooling_V2_Pilot_Presentation.pptx"

# ---------------------------------------------------------------- palette --
PRIMARY = RGBColor(0x0B, 0x3D, 0x59)     # deep ocean blue -- dark slides, headers
SECONDARY = RGBColor(0x1C, 0x72, 0x93)   # teal -- supporting fills
ACCENT = RGBColor(0xE0, 0x6C, 0x2B)      # warm copper -- pilot/attention labels
SUCCESS = RGBColor(0x2E, 0x7D, 0x32)     # green -- done / safe
DANGER = RGBColor(0xB0, 0x2A, 0x2A)      # red -- blocked / violation
INK = RGBColor(0x1A, 0x1F, 0x24)         # near-black text
MUTED = RGBColor(0x5B, 0x66, 0x6E)       # muted gray text
PANEL = RGBColor(0xF1, 0xF4, 0xF6)       # light panel fill
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xCE, 0xD6, 0xDA)

MPL_PRIMARY = "#0B3D59"
MPL_SECONDARY = "#1C7293"
MPL_ACCENT = "#E06C2B"
MPL_SUCCESS = "#2E7D32"
MPL_DANGER = "#B02A2A"
MPL_MUTED = "#8A9499"

FONT_HEAD = "Cambria"
FONT_BODY = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#8A9499",
    "axes.labelcolor": "#1A1F24",
    "text.color": "#1A1F24",
    "xtick.color": "#1A1F24",
    "ytick.color": "#1A1F24",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

PILOT_TAG = "PILOT RESULT — CELLS A–D — NOT FINAL"


# ------------------------------------------------------------- data load --
def load_json(rel):
    return json.loads((REPO_ROOT / rel).read_text())


FORECAST = load_json("results/pilot/forecasting/metrics.json")
POWER = load_json("results/pilot/power_model/metrics.json")
CLASSICAL = load_json("results/pilot/controllers/classical_controller_comparison.json")
MPC = load_json("results/pilot/controllers/mpc_result.json")
SHIELD = load_json("results/pilot/safety_ablation/shield_ablation.json")
ROBUST = load_json("results/pilot/robustness/robustness_results.json")
ALLCTRL = load_json("results/pilot/pilot_controller_comparison_all.json")
MAPPO_HIST = load_json("results/pilot/mappo/training_history_seed0.json")
THERMAL_SWEEP = json.loads((ASSETS_DIR / "thermal_sweep.json").read_text())
PID_TRAJ = json.loads((ASSETS_DIR / "pid_trajectory.json").read_text())

with open(REPO_ROOT / "results" / "model_selection_history.csv") as f:
    MODEL_HISTORY = list(csv.DictReader(f))


def agg(d, key):
    return d["aggregated"][key]["mean"]


# ============================================================ charts =====
def chart_forecast_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    metrics = ["mae", "rmse", "r2"]
    titles = ["MAE (lower better)", "RMSE (lower better)", "R² (higher better)"]
    models = ["persistence", "moving_average", "gru"]
    labels = ["Persistence", "Moving Avg", "GRU"]
    colors = [MPL_MUTED, MPL_SECONDARY, MPL_ACCENT]
    for ax, m, t in zip(axes, metrics, titles):
        vals = [FORECAST["overall"][mm][m] for mm in models]
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        ax.set_title(t, fontsize=11)
        ax.bar_label(bars, fmt="%.3f" if m == "r2" else "%.0f", fontsize=9, padding=2)
        ax.set_ylim(0, max(vals) * 1.25 if m != "r2" else max(vals) * 1.3)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("Workload Forecasting — Test Split (A–D pilot)", fontsize=12, y=1.03)
    fig.tight_layout()
    path = ASSETS_DIR / "forecast_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_forecast_by_regime():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    regimes = ["low_load", "high_load", "burst"]
    labels = ["Low load", "High load", "Burst (top 10%)"]
    models = ["persistence", "moving_average", "gru"]
    mlabels = ["Persistence", "Moving Avg", "GRU"]
    colors = [MPL_MUTED, MPL_SECONDARY, MPL_ACCENT]
    x = np.arange(len(regimes))
    w = 0.25
    for i, (m, ml, c) in enumerate(zip(models, mlabels, colors)):
        vals = [FORECAST["by_regime"][r][m]["r2"] for r in regimes]
        ax.bar(x + (i - 1) * w, vals, width=w, label=ml, color=c)
    ax.axhline(0, color="#8A9499", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("R² by workload regime")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Forecast Quality Degrades at Burst Load (A–D pilot, test split)", fontsize=11)
    fig.tight_layout()
    path = ASSETS_DIR / "forecast_by_regime.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_ppo_training():
    rows = []
    p = ASSETS_DIR / "ppo_monitor.monitor.csv"
    with open(p) as f:
        next(f)  # comment line written by Monitor
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(float(row["r"]))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(rows, color=MPL_ACCENT, linewidth=1.2)
    if len(rows) > 5:
        k = max(3, len(rows) // 15)
        smooth = np.convolve(rows, np.ones(k) / k, mode="valid")
        ax.plot(range(k - 1, len(rows)), smooth, color=MPL_PRIMARY, linewidth=2.2, label="rolling mean")
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return")
    ax.set_title("PPO Smoke-Test Training — 8,640 timesteps (A–D)", fontsize=11)
    fig.tight_layout()
    path = ASSETS_DIR / "ppo_training.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_mappo_training():
    rewards = [h["mean_reward"] for h in MAPPO_HIST]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(rewards, color=MPL_SECONDARY, linewidth=1.6, marker="o", markersize=3)
    ax.set_xlabel("update")
    ax.set_ylabel("mean rollout reward (scaled)")
    ax.set_title("MAPPO Smoke-Test Training — 30 updates (A–D)", fontsize=11)
    fig.tight_layout()
    path = ASSETS_DIR / "mappo_training.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_power_model_comparison():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = ["constant_baseline", "linear_pooled", "linear_plus_cell_fixed_effect", "random_forest"]
    labels = ["Constant", "Linear\n(pooled)", "Linear +\ncell effect", "Random\nForest"]
    r2 = [POWER["results"][n]["r2"] for n in names]
    colors = [MPL_MUTED, MPL_MUTED, MPL_SECONDARY, MPL_ACCENT]
    bars = ax.bar(labels, r2, color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.axhline(0, color="#8A9499", linewidth=0.8)
    ax.set_ylabel("Validation R²")
    ax.set_title("Empirical Power Model — Candidate Comparison (A–D pilot)", fontsize=11)
    fig.tight_layout()
    path = ASSETS_DIR / "power_model_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_power_residuals():
    tercile = POWER["residuals_by_load_tercile"]
    labels = [t["regime"].capitalize() for t in tercile]
    means = [t["mean"] for t in tercile]
    stds = [t["std"] for t in tercile]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(labels, means, yerr=stds, color=MPL_SECONDARY, width=0.5, capsize=4)
    ax.axhline(0, color="#1A1F24", linewidth=1)
    ax.set_ylabel("Residual (predicted − measured)")
    ax.set_title("Random Forest Residuals by Load Tercile (validation)", fontsize=11)
    fig.tight_layout()
    path = ASSETS_DIR / "power_residuals.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_thermal_sweep():
    scales = [r["heat_scale_kw"] for r in THERMAL_SWEEP]
    off_mean = [r["off_mean"] for r in THERMAL_SWEEP]
    full_mean = [r["full_mean"] for r in THERMAL_SWEEP]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(scales, off_mean, "o-", color=MPL_DANGER, label="cooling OFF (mean max temp)")
    ax.plot(scales, full_mean, "o-", color=MPL_SUCCESS, label="cooling FULL (mean max temp)")
    ax.axhline(27.0, color="#1A1F24", linestyle="--", linewidth=1, label="safety limit (27°C)")
    ax.axvline(0.85, color=MPL_ACCENT, linestyle=":", linewidth=1.6)
    ax.annotate("selected\nheat_scale_kw=0.85", xy=(0.85, 20.5), fontsize=8.5, color=MPL_ACCENT, ha="center")
    ax.set_xlabel("heat_scale_kw (assumed simulation constant)")
    ax.set_ylabel("max zone temperature, °C\n(251-window dense sweep, shield OFF)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Thermal Calibration Sweep — Cooling-off vs. Full-cooling (A–D)", fontsize=11)
    fig.tight_layout()
    path = ASSETS_DIR / "thermal_sweep.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_pid_trajectory():
    temps = np.array(PID_TRAJ["temps"])  # (T, n_zones)
    limit = PID_TRAJ["safety_limit_c"]
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    t = np.arange(temps.shape[0]) * 5 / 60.0  # hours (5-min steps)
    for z in range(temps.shape[1]):
        ax.plot(t, temps[:, z], linewidth=1.1, alpha=0.85, label=f"cell {chr(97+z)}")
    ax.axhline(limit, color=MPL_DANGER, linestyle="--", linewidth=1.4, label=f"safety limit ({limit}°C)")
    ax.fill_between(t, limit, temps.max(), where=(temps.max(axis=1) > limit), color=MPL_DANGER, alpha=0.08)
    ax.set_xlabel("hours into episode")
    ax.set_ylabel("simulated zone temperature, °C")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    n_viol = PID_TRAJ["n_violation_steps"]
    n_steps = PID_TRAJ["n_steps"]
    ax.set_title(f"SIMULATED Temperature — PID, shield ON, one 24h test episode "
                 f"({n_viol}/{n_steps} steps above limit)", fontsize=10.5)
    fig.tight_layout()
    path = ASSETS_DIR / "pid_trajectory.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_controller_comparison():
    names = ["fixed", "threshold", "pid", "mpc", "ppo_pilot_smoketest", "mappo_pilot_smoketest"]
    labels = ["Fixed", "Threshold", "PID", "MPC", "PPO\n(smoke)", "MAPPO\n(smoke)"]
    energy = []
    viol = []
    for n in names:
        if n == "mpc":
            energy.append(agg(MPC["result"], "energy.total_kwh"))
            viol.append(agg(MPC["result"], "thermal.violation_pct"))
        elif n in ALLCTRL["results"]:
            energy.append(ALLCTRL["results"][n]["energy.total_kwh"]["mean"])
            viol.append(ALLCTRL["results"][n]["thermal.violation_pct"]["mean"])
    colors = [MPL_SECONDARY, MPL_SECONDARY, MPL_PRIMARY, MPL_SECONDARY, MPL_ACCENT, MPL_ACCENT]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    bars = axes[0].bar(labels, energy, color=colors)
    axes[0].bar_label(bars, fmt="%.1f", fontsize=8.5, padding=2)
    axes[0].set_ylabel("energy, kWh/episode")
    axes[0].set_title("Energy per episode (lower better)", fontsize=10.5)

    bars2 = axes[1].bar(labels, viol, color=colors)
    axes[1].bar_label(bars2, fmt="%.0f%%", fontsize=8.5, padding=2)
    axes[1].set_ylabel("violation steps, %")
    axes[1].set_title("Safety-limit violation rate (lower better)", fontsize=10.5)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Controller Comparison — 20 Test Episodes (A–D pilot)", fontsize=12, y=1.04)
    fig.tight_layout()
    path = ASSETS_DIR / "controller_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_shield_ablation():
    rel = SHIELD["reliance_analysis"]
    names = list(rel.keys())
    labels = [n.capitalize() for n in names]
    on = [rel[n]["violation_pct_shield_on"] for n in names]
    off = [rel[n]["violation_pct_shield_off"] for n in names]
    x = np.arange(len(names))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - w / 2, on, width=w, label="shield ON", color=MPL_SUCCESS)
    ax.bar(x + w / 2, off, width=w, label="shield OFF", color=MPL_DANGER)
    ax.set_xticks(x, labels)
    ax.set_ylabel("violation steps, %")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Safety Shield Ablation (A–D pilot, 20 test episodes)", fontsize=11)
    for i, n in enumerate(names):
        note = "shield-dependent" if rel[n]["shield_dependent"] else "not shield-dependent*"
        ax.annotate(note, xy=(i, max(on[i], off[i]) + 3), ha="center", fontsize=8, color=MPL_MUTED)
    fig.tight_layout()
    path = ASSETS_DIR / "shield_ablation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_robustness():
    b = ROBUST["experiment_B_workload_spike"]
    scales = [1.0, 2.0, 5.0, 10.0]
    pid_viol = [b[f"scale_{s}"]["pid"]["viol_pct"] for s in scales]
    thr_viol = [b[f"scale_{s}"]["threshold"]["viol_pct"] for s in scales]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(scales, pid_viol, "o-", color=MPL_PRIMARY, label="PID")
    ax.plot(scales, thr_viol, "o-", color=MPL_SECONDARY, label="Threshold")
    ax.set_xscale("log")
    ax.set_xticks(scales, [str(s) + "x" for s in scales])
    ax.set_xlabel("synthetic workload/power scale factor")
    ax.set_ylabel("violation steps, %")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Robustness: Synthetic Workload Spike (A–D pilot)", fontsize=11)
    fig.tight_layout()
    path = ASSETS_DIR / "robustness.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


CHARTS = {}
def get_chart(name, fn):
    if name not in CHARTS:
        CHARTS[name] = fn()
    return CHARTS[name]


# ======================================================== pptx helpers ===
def new_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs, bg=WHITE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = bg
    return slide


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def textbox(slide, x, y, w, h, text, size=16, color=INK, bold=False, italic=False,
            font=FONT_BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.0,
            wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = font
    return tb


def bullets(slide, x, y, w, h, items, size=15, color=INK, font=FONT_BODY,
            space_after=8, bold_lead=False, marker="•  "):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.08
        lead = item
        rest = None
        if bold_lead and ":" in item:
            lead, rest = item.split(":", 1)
            lead = lead + ":"
        r0 = p.add_run()
        r0.text = marker + lead
        r0.font.size = Pt(size)
        r0.font.color.rgb = color
        r0.font.name = font
        r0.font.bold = bold_lead
        if rest is not None:
            r1 = p.add_run()
            r1.text = rest
            r1.font.size = Pt(size)
            r1.font.color.rgb = color
            r1.font.name = font
    return tb


def rect(slide, x, y, w, h, text="", fill=SECONDARY, text_color=WHITE, size=13,
         bold=False, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, font=FONT_BODY,
         align=PP_ALIGN.CENTER):
    shp = slide.shapes.add_shape(shape_type, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(4)
    tf.margin_right = Pt(4)
    tf.margin_top = Pt(2)
    tf.margin_bottom = Pt(2)
    if isinstance(text, str):
        text = [text]
    for i, line_txt in enumerate(text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line_txt
        run.font.size = Pt(size)
        run.font.color.rgb = text_color
        run.font.bold = bold if i == 0 else False
        run.font.name = font
    return shp


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    line = conn.line._get_or_add_ln()
    tail = line.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    line.append(tail)
    return conn


def down_arrow(slide, cx, y, length=Inches(0.22), color=MUTED):
    return arrow(slide, cx, y, cx, y + length, color=color)


def right_arrow(slide, x, cy, length=Inches(0.22), color=MUTED):
    return arrow(slide, x, cy, x + length, cy, color=color)


def title_bar(slide, title, kicker=None, dark=False):
    """Slide title, top-left, no accent-line motif (per style guidance --
    whitespace instead of a rule)."""
    color = WHITE if dark else PRIMARY
    y = Inches(0.35)
    if kicker:
        textbox(slide, Inches(0.6), Inches(0.28), Inches(11), Inches(0.3), kicker.upper(),
                size=11.5, color=ACCENT if not dark else RGBColor(0xFF, 0xC9, 0x9E),
                bold=True, font=FONT_BODY)
        y = Inches(0.58)
    textbox(slide, Inches(0.6), y, Inches(12.1), Inches(0.75), title,
            size=28, color=color, bold=True, font=FONT_HEAD)


def pilot_tag(slide, x=None, y=Inches(1.08), final=False):
    text = "FINAL RESULT — CELLS A–H" if final else PILOT_TAG
    fill = SUCCESS if final else ACCENT
    w = Inches(3.55) if not final else Inches(2.9)
    x = x if x is not None else (SLIDE_W - w - Inches(0.5))
    box = rect(slide, x, y, w, Inches(0.34), text, fill=fill, text_color=WHITE, size=10.5, bold=True)
    return box


def footer(slide, text):
    textbox(slide, Inches(0.6), SLIDE_H - Inches(0.42), Inches(9), Inches(0.3), text,
            size=9.5, color=MUTED, italic=True)


def slide_number_tag(slide, n):
    textbox(slide, SLIDE_W - Inches(0.7), SLIDE_H - Inches(0.42), Inches(0.5), Inches(0.3),
            str(n), size=9.5, color=MUTED, align=PP_ALIGN.RIGHT)


def vertical_chain(slide, x, y_top, box_w, box_h, gap, steps, box_size=12):
    """steps: list of (label, fill) or label strings. Returns y of the
    bottom of the last box."""
    y = y_top
    for i, step in enumerate(steps):
        if isinstance(step, tuple):
            label, fill = step
        else:
            label, fill = step, SECONDARY
        rect(slide, x, y, box_w, box_h, label, fill=fill, size=box_size)
        if i < len(steps) - 1:
            down_arrow(slide, x + box_w / 2, y + box_h, gap)
        y += box_h + gap
    return y - gap


def image_slide_full(slide, path, top=Inches(1.5), max_h=Inches(5.4)):
    from PIL import Image
    im = Image.open(path)
    ratio = im.width / im.height
    h = max_h
    w = Emu(int(h * ratio))
    if w > Inches(12.1):
        w = Inches(12.1)
        h = Emu(int(w / ratio))
    x = (SLIDE_W - w) / 2
    slide.shapes.add_picture(str(path), x, top, width=w, height=h)
    return x, top, w, h


def make_table(slide, x, y, w, h, headers, rows, font_size=12, header_fill=PRIMARY,
               highlight_row=None, col_widths=None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    gshape = slide.shapes.add_table(n_rows, n_cols, x, y, w, h)
    table = gshape.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Emu(int(w * cw / total))
    for j, htext in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        run = p.add_run(); run.text = htext
        run.font.size = Pt(font_size + 0.5); run.font.bold = True; run.font.color.rgb = WHITE
        run.font.name = FONT_BODY
        cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
        cell.margin_top = Pt(4); cell.margin_bottom = Pt(4)
    for i, row in enumerate(rows):
        is_hi = (highlight_row is not None and i == highlight_row)
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            run = p.add_run(); run.text = str(val)
            run.font.size = Pt(font_size)
            run.font.name = FONT_BODY
            run.font.bold = is_hi and j == 0
            run.font.color.rgb = SUCCESS if is_hi else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xEA, 0xF3, 0xEC) if is_hi else (WHITE if i % 2 == 0 else PANEL)
            cell.margin_top = Pt(3); cell.margin_bottom = Pt(3)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return gshape


# =============================================================== slides ==
SLIDE_N = [0]


def next_n():
    SLIDE_N[0] += 1
    return SLIDE_N[0]


def s01_title(prs):
    slide = blank_slide(prs, bg=PRIMARY)
    textbox(slide, Inches(0.9), Inches(2.0), Inches(11.5), Inches(0.5),
            "VERSION 2 — PILOT / DEVELOPMENT STAGE  ·  CELLS A–D",
            size=14, color=RGBColor(0xFF, 0xC9, 0x9E), bold=True, font=FONT_BODY)
    textbox(slide, Inches(0.9), Inches(2.5), Inches(11.5), Inches(1.7),
            "BORG Datacenter Cooling Optimization", size=42, color=WHITE, bold=True, font=FONT_HEAD,
            line_spacing=1.05)
    textbox(slide, Inches(0.9), Inches(4.25), Inches(11.0), Inches(1.3),
            "Official Google Borg 2019 Workload and Power Traces with Workload\n"
            "Forecasting, Thermal Modeling and Safe Multi-Agent Reinforcement Learning",
            size=18, color=RGBColor(0xCA, 0xDC, 0xFC), font=FONT_BODY, line_spacing=1.2)
    textbox(slide, Inches(0.9), Inches(6.35), Inches(11), Inches(0.4),
            "Research Progress Presentation · git: v2-official-google-data @ b83ce2a",
            size=12.5, color=RGBColor(0x9F, 0xB6, 0xC8), font=FONT_BODY)
    add_notes(slide, (
        "SAY (30-45s): This project asks whether we can build a workload-aware "
        "cooling controller for a datacenter, using REAL Google workload and power "
        "traces instead of assumptions, and whether learned multi-agent control "
        "(MAPPO) can beat classical control on the energy/safety trade-off. "
        "We are presenting PROGRESS, not a finished result: this is the pilot "
        "stage, built and validated on 4 of the 8 official Google Borg cells "
        "while the remaining 4 are waiting on a free-tier data quota. "
        "Every number in this deck is pilot evidence, not a final claim -- "
        "that distinction is repeated on every results slide."
    ))
    return slide


def s02_core_problem(prs):
    slide = blank_slide(prs)
    title_bar(slide, "The Core Engineering Problem")
    chain = ["Datacenter workload", "Server utilization", "Electrical power",
             "Heat generation", "Temperature", "Cooling requirement", "Cooling energy"]
    x = Inches(0.9)
    y = Inches(1.5)
    bw, bh, gap = Inches(3.0), Inches(0.5), Inches(0.18)
    vertical_chain(slide, x, y, bw, bh, gap, chain, box_size=13)
    # conflict panel on the right
    px, py, pw, ph = Inches(5.4), Inches(1.5), Inches(6.9), Inches(4.85)
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, px, py, pw, ph)
    panel.fill.solid(); panel.fill.fore_color.rgb = PANEL; panel.line.fill.background()
    panel.shadow.inherit = False
    textbox(slide, px + Inches(0.3), py + Inches(0.2), pw - Inches(0.6), Inches(0.4),
            "The control conflict", size=17, bold=True, color=PRIMARY, font=FONT_HEAD)
    rect(slide, px + Inches(0.3), py + Inches(0.75), Inches(2.9), Inches(0.7),
         ["Too much", "cooling"], fill=SECONDARY, size=13, bold=True)
    textbox(slide, px + Inches(3.35), py + Inches(0.85), Inches(3.0), Inches(0.5),
            "→ wastes energy", size=13, color=INK)
    rect(slide, px + Inches(0.3), py + Inches(1.65), Inches(2.9), Inches(0.7),
         ["Too little", "cooling"], fill=DANGER, size=13, bold=True)
    textbox(slide, px + Inches(3.35), py + Inches(1.75), Inches(3.3), Inches(0.5),
            "→ thermal safety risk", size=13, color=INK)
    textbox(slide, px + Inches(0.3), py + Inches(2.6), pw - Inches(0.6), Inches(0.4),
            "The research objective:", size=15, bold=True, color=PRIMARY)
    for i, (label, c) in enumerate([("ENERGY EFFICIENCY", SECONDARY),
                                     ("THERMAL SAFETY", DANGER),
                                     ("CONTROL STABILITY", SUCCESS)]):
        rect(slide, px + Inches(0.3) + i * Inches(2.2), py + Inches(3.15), Inches(2.0), Inches(0.65),
             label, fill=c, size=11.5, bold=True)
    textbox(slide, px + Inches(0.3), py + Inches(4.0), pw - Inches(0.6), Inches(0.7),
            "Balance all three at once — not just minimize energy, and not\n"
            "just eliminate risk by over-cooling.", size=13, color=MUTED, line_spacing=1.15)
    pilot_tag(slide)
    footer(slide, "Causal chain from workload to cooling energy; the trade-off this whole project studies.")
    add_notes(slide, (
        "SAY: Follow the left-hand chain top to bottom: workload creates utilization, "
        "utilization draws power, power becomes heat, heat raises temperature, and once "
        "temperature is high enough the cooling system has to work, which costs energy. "
        "The panel on the right is the actual conflict: cool too aggressively and you waste "
        "energy for no safety benefit; cool too little and you risk exceeding a safe operating "
        "temperature. The entire rest of the deck is about building a system that can make this "
        "trade-off well, using real workload and power data instead of guesses. "
        "LIKELY QUESTION: 'Isn't this just always cool to the max to be safe?' ANSWER: cooling "
        "capacity itself costs continuous energy -- our own thermal calibration (slide 21) shows "
        "full cooling all the time uses roughly 3x more energy per episode than a well-tuned PID, "
        "for no additional safety benefit under normal load."
    ))
    return slide


def s03_control_loop(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Why This Is a Feedback Control Problem")
    steps = ["State / Observations", "Controller", "Cooling Action", "Thermal Response", "Next State"]
    x, y, bw, bh, gap = Inches(4.6), Inches(1.6), Inches(4.1), Inches(0.62), Inches(0.28)
    vertical_chain(slide, x, y, bw, bh, gap, steps, box_size=14)
    # feedback loop arrow from bottom back to top
    loop_x = x + bw + Inches(0.6)
    arrow(slide, x + bw, y + bh / 2, loop_x, y + bh / 2, color=MUTED)
    last_y = y + 4 * (bh + gap) - gap
    arrow(slide, x + bw, last_y + bh / 2, loop_x, last_y + bh / 2, color=MUTED)
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, loop_x, y + bh / 2, loop_x, last_y + bh / 2)
    conn.line.color.rgb = MUTED; conn.line.width = Pt(1.5)
    textbox(slide, loop_x + Inches(0.1), (y + last_y) / 2, Inches(1.6), Inches(0.6),
            "feedback\nloop", size=11, color=MUTED, italic=True)

    bullets(slide, Inches(0.6), Inches(5.55), Inches(12.1), Inches(1.6), [
        "Dynamic: today's cooling decision changes tomorrow's starting temperature.",
        "Lagged: heat takes time to raise temperature, and cooling takes time to remove it.",
        "Workload varies over time — a controller that only reacts to the CURRENT state is always a step behind.",
        "Forecasting (next section) gives the controller information about near-future workload, ahead of time.",
    ], size=14.5)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: This is a classic feedback control loop, not a one-shot prediction problem. The "
        "controller looks at temperatures/workload NOW, picks a cooling action, that action plus "
        "the incoming workload determines the NEXT temperature, and the loop repeats. Because "
        "heat has thermal inertia (it takes time to build up and time to remove), a controller "
        "that only reacts to what has already happened will lag behind. That is exactly the "
        "motivation for adding a forecaster: if the controller knows workload is about to rise, "
        "it can start cooling proactively instead of reactively. "
        "LIKELY QUESTION: 'Why not just always look at current temperature, isn't that enough?' "
        "ANSWER: our own pilot data shows a reactive PID controller still violates the safety "
        "limit 34% of the time (slide 22) specifically because of this lag -- that's the "
        "motivating evidence for this slide, not a hypothetical."
    ))
    return slide


def s04_v1_system(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Version 1: The Original System")
    bullets(slide, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.2), [
        "Dataset: Kaggle export of a Google Borg trace (“muzairbair/borg-traces-data”), "
        "not the official Google release — no uploader documentation.",
        "Workload: 405,894 real records pooled across all 8 Borg clusters, "
        "binned to 15 minutes; zones were 3 SYNTHETIC groups from hash(machine_id) % 3.",
        "Power: NO real power measurements existed. Heat was computed from an "
        "ASSUMED linear CPU→power formula.",
        "Thermal model: 3-zone lumped-parameter simulation, numerically validated "
        "(9/9 tests) but never claimed to be physically calibrated.",
        "Controllers: Fixed, Threshold, PID, random-shooting MPC, single-agent PPO, "
        "and a from-scratch CTDE MAPPO.",
        "V1's own headline finding: classical PID beat both PPO and MAPPO on energy "
        "efficiency, under a CPU-only compute budget — reported honestly, not hidden.",
    ], size=15, space_after=12)
    rect(slide, Inches(7.0), Inches(1.5), Inches(5.75), Inches(5.2), "", fill=PANEL, text_color=INK)
    textbox(slide, Inches(7.3), Inches(1.7), Inches(5.2), Inches(0.4),
            "V1 result, verbatim (docs/final_validation_report.md)", size=13, bold=True, color=PRIMARY)
    rows = [("Fixed", "17.93", "0.0%"), ("Threshold", "7.20", "0.0%"),
            ("PID", "6.43", "0.0%"), ("MPC", "7.92", "0.0%"),
            ("PPO (best)", "6.47", "0.0%"), ("MAPPO (best)", "19.15", "0.0%")]
    make_table(slide, Inches(7.3), Inches(2.15), Inches(5.2), Inches(2.3),
               ["Controller", "Energy (kWh/ep)", "Violations"], rows, highlight_row=2,
               font_size=11.5, col_widths=[2.1, 1.7, 1.4])
    textbox(slide, Inches(7.3), Inches(5.15), Inches(5.2), Inches(1.4),
            "V1 is complete and preserved (tag v1-kaggle-baseline, branch main). "
            "It is the reason this project reports negative/mixed results honestly — "
            "it already found that PID beat RL, and did not suppress that.",
            size=12, color=MUTED, line_spacing=1.2)
    add_notes(slide, (
        "SAY: V1 is a completed, honestly-reported baseline, not a failure to be embarrassed "
        "about. Its most important property is methodological: when RL lost to classical "
        "control, that result was kept and published, not hidden or re-tuned away. That "
        "discipline is what V2 inherits and this deck follows the same rule -- if V2's PPO/MAPPO "
        "lose to PID again on the final 8-cell data, that will also be reported as-is. "
        "LIKELY QUESTION: 'Why build V2 at all if V1 already has a full result?' ANSWER: V1's "
        "workload came from an undocumented Kaggle re-export with no real power data at all; "
        "V2 replaces both with official, documented Google sources so any claim (positive or "
        "negative) rests on stronger evidence."
    ))
    return slide


def s05_why_v1_insufficient(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Why Version 1 Wasn't Enough for the Research Goal")
    rows = [
        ("Undocumented Kaggle re-export", "Provenance/join method unknown; can't defend as 'the official trace'",
         "Switch to official ClusterData2019 — DONE (pilot)"),
        ("No real power measurements", "Power model was a pure assumption, never validated",
         "Fit an empirical model against real PowerData2019 — DONE (pilot)"),
        ("Sparse workload (94% zero bins)", "Forced 15-min bins and cross-cluster pooling to get signal",
         "Cell-level aggregation is naturally dense — DONE (pilot)"),
        ("Thermal params never recalibrated on real data", "Risk of a vacuous (never-violating) safety problem",
         "Dense 251-window recalibration — DONE (pilot); full re-check planned for 8 cells"),
        ("Only 3 development seeds, reduced RL budget", "Can't rule out noise driving the PID-wins finding",
         "Full multi-seed budget — PLANNED for final 8-cell run"),
        ("Single population (one Kaggle export)", "No held-out, differently-sourced population to test on",
         "8 real Borg cells, split train/val/test — 4 done (pilot), 4 PLANNED"),
    ]
    make_table(slide, Inches(0.6), Inches(1.55), Inches(12.1), Inches(5.3),
               ["V1 limitation", "Why it matters", "V2 response"], rows, font_size=11.5,
               col_widths=[3.1, 4.2, 4.8])
    add_notes(slide, (
        "SAY: Read this as three columns per row: what was limited in V1, why that limitation "
        "actually mattered scientifically (not just 'it was old'), and exactly what V2 does about "
        "it -- with an explicit DONE/PLANNED tag, never both claimed as finished. "
        "LIKELY QUESTION: 'Has V2 already fixed everything in this table?' ANSWER: No -- read the "
        "right column carefully. Several rows say 'DONE (pilot)' meaning validated on 4 cells "
        "only, and two rows are explicitly 'PLANNED', not done."
    ))
    return slide


def s06_research_objective(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Research Objective and Questions")
    textbox(slide, Inches(0.6), Inches(1.5), Inches(12.1), Inches(1.1),
            "Develop and evaluate a workload-aware datacenter cooling control framework "
            "using official Google Borg workload and power traces, workload forecasting, "
            "thermal modeling, classical controllers, PPO, MAPPO, and explicit safety constraints.",
            size=17, color=PRIMARY, italic=True, line_spacing=1.25)
    rqs = [
        ("RQ1", "Can workload telemetry provide useful short-horizon forecasts?"),
        ("RQ2", "Can measured power data support an empirical workload→power model?"),
        ("RQ3", "Can the resulting thermal simulation support meaningful cooling-control experiments?"),
        ("RQ4", "How do classical controllers, PPO and MAPPO compare under common thermal constraints?"),
        ("RQ5", "Does forecasting and/or safety intervention materially affect control performance?"),
    ]
    y = Inches(2.85)
    for tag, q in rqs:
        rect(slide, Inches(0.6), y, Inches(1.05), Inches(0.62), tag, fill=PRIMARY, size=14, bold=True)
        textbox(slide, Inches(1.85), y + Inches(0.06), Inches(10.8), Inches(0.6), q, size=15.5)
        y += Inches(0.78)
    textbox(slide, Inches(0.6), y + Inches(0.1), Inches(11.5), Inches(0.6),
            "None of these questions have a predetermined answer — the evidence decides, "
            "including if the answer is negative.", size=13, italic=True, color=MUTED)
    add_notes(slide, (
        "SAY: These five questions frame everything that follows. Note we do NOT state an "
        "expected answer for any of them -- RQ4 in particular could come back 'classical control "
        "wins again', and V1's own result suggests that's a live possibility, not something to "
        "explain away in advance. "
        "LIKELY QUESTION: 'What's the actual novel contribution here?' ANSWER: honestly, at this "
        "pilot stage the contribution is the validated infrastructure (real-data pipeline, fitted "
        "power model, tested MAPPO) plus early evidence on RQ1-RQ3; RQ4/RQ5 need the full 8-cell "
        "run before a defensible answer exists."
    ))
    return slide



def s07_clusterdata(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Official Dataset 1: ClusterData2019")
    bullets(slide, Inches(0.6), Inches(1.65), Inches(6.9), Inches(4.9), [
        "Source: official Google “cluster-data” GitHub release "
        "(google/cluster-data, ClusterData2019.md), not a third-party re-export.",
        "Coverage: eight independent, concurrent Borg cells, for the month of May 2019.",
        "Resolution: per-task/per-machine CPU and memory usage, at 5-minute native "
        "granularity (instance_usage table).",
        "What V2 actually uses: start_time, average_usage.cpus, average_usage.memory, "
        "and machine_id — aggregated per cell, per 5-minute bucket.",
        "Access method: BigQuery public dataset, queried with column-pruned, "
        "cost-estimated SQL — the raw table is ~2.0–2.4 TB per cell, never downloaded whole.",
    ], size=15, space_after=13)
    panel = rect(slide, Inches(7.8), Inches(1.65), Inches(4.95), Inches(4.9), "", fill=PANEL)
    textbox(slide, Inches(8.1), Inches(1.9), Inches(4.4), Inches(0.4),
            "Why this matters vs. a generic/synthetic trace", size=14, bold=True, color=PRIMARY)
    bullets(slide, Inches(8.1), Inches(2.4), Inches(4.4), Inches(3.9), [
        "Documented schema and units — no guessing what a column means.",
        "Real inter-service, inter-machine correlation structure that a synthetic "
        "generator would not reproduce.",
        "Directly pairs with an official Google power dataset (next slide) — a "
        "synthetic trace would have no real power counterpart at all.",
    ], size=13.5, space_after=10)
    add_notes(slide, (
        "SAY: This is the real fix from V1's Kaggle export -- same underlying Google system, "
        "but now the official, documented release. The number that matters operationally: each "
        "cell's raw table is 2-2.4 terabytes, which is why every extraction script in this project "
        "dry-runs its query cost first and never downloads the raw table. "
        "LIKELY QUESTION: 'Is this the same data as V1?' ANSWER: same underlying system (Google "
        "Borg) and same rough scale (8 cells), but a completely different, officially documented "
        "export -- V1's Kaggle file had zero uploader documentation of its join/sampling method."
    ))
    return slide


def s08_powerdata(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Official Dataset 2: PowerData2019")
    bullets(slide, Inches(0.6), Inches(1.65), Inches(6.9), Inches(4.6), [
        "Source: official Google “cluster-data” release, PowerData2019.md — "
        "same GitHub repository as ClusterData2019.",
        "Coverage: 57 power domains (Power Distribution Units, PDUs) during May 2019.",
        "Google's own documentation: “two of these power domains are from cells with "
        "the new medium voltage power plane design; the remainder belong to the eight "
        "cells featured in the 2019 Cluster Data trace.”",
        "Fields actually used: cell, pdu, measured_power_util, production_power_util, "
        "and two data-quality flags — utilization FRACTIONS of PDU capacity, not watts.",
        "50 PDU tables were found in the public BigQuery mirror (documented discrepancy "
        "vs. the 55 implied by Google's text — disclosed in the alignment audit, not hidden).",
    ], size=14, space_after=11)
    panel = rect(slide, Inches(7.8), Inches(1.65), Inches(4.95), Inches(4.9), "", fill=PANEL)
    textbox(slide, Inches(8.1), Inches(1.9), Inches(4.4), Inches(1.2),
            "Three different things — do not conflate them", size=14, bold=True, color=PRIMARY,
            line_spacing=1.1)
    for i, (label, desc, c) in enumerate([
        ("WORKLOAD DATA", "real, official (ClusterData2019)", SECONDARY),
        ("MEASURED POWER DATA", "real, official (PowerData2019)", SECONDARY),
        ("SIMULATED THERMAL DATA", "a simulation abstraction, not measured", ACCENT),
    ]):
        y = Inches(2.95) + i * Inches(0.95)
        rect(slide, Inches(8.1), y, Inches(4.3), Inches(0.5), label, fill=c, size=12, bold=True)
        textbox(slide, Inches(8.1), y + Inches(0.53), Inches(4.3), Inches(0.35), desc,
                size=11.5, color=MUTED, italic=True)
    add_notes(slide, (
        "SAY: PowerData2019 is Google's own measured power reference -- the thing V1 never had "
        "at all. Two important honesty points here: first, the utilization values are FRACTIONS "
        "of a PDU's rated capacity, not absolute watts, because no capacity-in-kW field exists in "
        "the public schema -- so our power model's output is a normalized quantity, not literal "
        "watts. Second, we found and documented a real discrepancy: Google's text implies 55 "
        "domains should belong to the 8 cells, but the public BigQuery mirror only exposes 50 "
        "tables -- we reported that gap rather than quietly assuming it away. "
        "LIKELY QUESTION: 'Are the three-zone temperatures real Google measurements?' ANSWER: No "
        "-- see the right-hand panel. Workload and power are real official data; the thermal "
        "environment itself is a simulation built ON TOP of that real data, explained fully on "
        "slide 20."
    ))
    return slide


def s09_alignment(prs):
    slide = blank_slide(prs)
    title_bar(slide, "The Central Question: Workload ↔ Power Alignment")
    textbox(slide, Inches(0.6), Inches(1.35), Inches(12.1), Inches(0.5),
            "“How can workload and power be combined if they were never recorded "
            "at the same machine?”", size=16, italic=True, color=PRIMARY)

    # left column: workload -> cell
    lx = Inches(0.9)
    rect(slide, lx, Inches(2.15), Inches(3.6), Inches(0.55), "Workload records\n(per machine, per task)",
         fill=SECONDARY, size=12.5)
    down_arrow(slide, lx + Inches(1.8), Inches(2.70), Inches(0.28))
    rect(slide, lx, Inches(2.98), Inches(3.6), Inches(0.55), "Borg-cell aggregation",
         fill=SECONDARY, size=12.5)
    down_arrow(slide, lx + Inches(1.8), Inches(3.53), Inches(0.28))

    # right column: power -> cell
    rx = Inches(8.9)
    rect(slide, rx, Inches(2.15), Inches(3.6), Inches(0.55), "Power-domain (PDU)\nmeasurements",
         fill=SECONDARY, size=12.5)
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, rx + Inches(1.8), Inches(2.70),
                                       rx + Inches(1.8), Inches(3.53))
    conn.line.color.rgb = MUTED; conn.line.width = Pt(1.75)
    up = conn.line._get_or_add_ln()
    head = up.makeelement(qn("a:headEnd"), {"type": "triangle", "w": "med", "len": "med"})
    up.append(head)
    rect(slide, rx, Inches(2.98), Inches(3.6), Inches(0.55), "PowerData2019's own\n“cell” column",
         fill=SECONDARY, size=12.5)

    # center: common key
    cx = Inches(4.85)
    arrow(slide, lx + Inches(3.6), Inches(3.25), cx, Inches(3.25), color=MUTED)
    arrow(slide, rx, Inches(3.25), cx + Inches(3.65), Inches(3.25), color=MUTED)
    rect(slide, cx, Inches(2.9), Inches(3.65), Inches(0.65), "COMMON KEY:\nBorg CELL (a–h)",
         fill=PRIMARY, size=13, bold=True)

    # what we do NOT do vs what we DO
    y2 = Inches(4.0)
    rect(slide, Inches(0.9), y2, Inches(5.6), Inches(0.85),
         ["NOT DONE:", "machine A → arbitrary power domain B"],
         fill=DANGER, size=13, bold=True)
    rect(slide, Inches(6.85), y2, Inches(5.6), Inches(0.85),
         ["DONE:", "workload → documented Borg-cell → measured power"],
         fill=SUCCESS, size=13, bold=True)

    bullets(slide, Inches(0.9), Inches(5.15), Inches(11.5), Inches(1.3), [
        "No machine-to-PDU mapping exists anywhere in either schema — it was not invented.",
        "Google's own PowerData2019.md documents that (most) power domains belong to the "
        "eight ClusterData2019 cells — the cell is the finest key both datasets actually share.",
        "Cost: per-machine and per-PDU resolution is lost. The unit of prediction is one "
        "of 8 Borg cells, not a machine or a rack.",
    ], size=13, space_after=6)
    add_notes(slide, (
        "SAY: This is the single most scrutinized design decision in the whole pipeline, so take "
        "the time here. Workload records get aggregated up to the Borg-cell level; power "
        "measurements already carry an explicit 'cell' column in their own schema. The cell is "
        "therefore not a convenience choice -- it is the finest join key that is actually "
        "supported by both datasets' own structure plus Google's documentation sentence. "
        "LIKELY QUESTION: 'How do you know workload and power actually belong together?' ANSWER: "
        "we verified it three ways -- (1) Google's own text says most PDUs belong to the 8 "
        "ClusterData cells, (2) every PowerData2019 row already carries a 'cell' field matching "
        "those same 8 labels, (3) we cross-checked the time axes (both ~31 days, 5-minute "
        "cadence) line up. We do NOT claim a machine-level mapping, and we found and disclosed a "
        "real discrepancy (50 vs. 55 PDU tables) rather than smoothing it over."
    ))
    return slide


def s10_data_audit(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Data Audit and Validation — Why This Comes Before ML/RL")
    items = [
        ("Official source verification", "gcloud/BigQuery access, dataset+table listing confirmed live"),
        ("Schema validation", "every field used (cell, pdu, machine_id, average_usage.*) read from the live schema"),
        ("Timestamp / epoch check", "power row-count (8,928) matches a 5-min grid over ~31 days; cross-checked ranges"),
        ("Duplicate check", "zero duplicate (cell, time-bucket) rows after the join"),
        ("Cell/zone consistency", "zone code = deterministic alphabetical index of cell; asserted, not assumed"),
        ("Workload/power join coverage", "≥ 95% of workload buckets matched a power bucket per cell (assert, not silently drop)"),
        ("Chronology / leakage", "train < validation < test bucket ranges enforced; GRU windows never cross a split"),
        ("Config consistency", "single source of truth for active cell set; n_zones derived from data, not configured"),
        ("Extraction dry-run", "every non-trivial BigQuery query cost-estimated before running for real"),
    ]
    make_table(slide, Inches(0.6), Inches(1.55), Inches(12.1), Inches(5.35),
               ["Check", "What was verified"], items, font_size=13, col_widths=[3.4, 8.7])
    add_notes(slide, (
        "SAY: Before any model gets trained, the pipeline has to pass structural checks -- this "
        "table is the actual list, not a generic 'we did QA' claim. All nine of these are backed "
        "by a specific assertion or test in the codebase (tests/test_v2_pipeline.py and "
        "tests/test_v2_config_consistency.py). "
        "LIKELY QUESTION: 'What happens if one of these fails?' ANSWER: the pipeline raises an "
        "assertion error and stops rather than silently producing a wrong result -- e.g. the join "
        "coverage check literally throws if any cell falls below 95% match rate."
    ))
    return slide


def s11_data_availability(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Current Data Availability")
    cells = "ABCDEFGH"
    status = ["available"] * 4 + ["blocked"] * 4
    x0 = Inches(0.9)
    cw = Inches(1.42)
    for i, (c, st) in enumerate(zip(cells, status)):
        x = x0 + i * (cw + Inches(0.08))
        fill = SUCCESS if st == "available" else RGBColor(0xE8, 0xC7, 0x9A)
        mark = "✓" if st == "available" else "⏳"
        rect(slide, x, Inches(1.55), cw, Inches(0.85), [c, mark], fill=fill,
             text_color=WHITE if st == "available" else RGBColor(0x7A, 0x4B, 0x12), size=20, bold=True)
    textbox(slide, x0, Inches(2.5), Inches(11.6), Inches(0.35),
            "✓ = workload extracted & validated     ⏳ = blocked by current BigQuery free-tier scan quota",
            size=12.5, color=MUTED)

    rows = [
        ("A", "available", "available", "PILOT"), ("B", "available", "available", "PILOT"),
        ("C", "available", "available", "PILOT"), ("D", "available", "available", "PILOT"),
        ("E", "blocked", "available", "WAITING"), ("F", "blocked", "available", "WAITING"),
        ("G", "blocked", "available", "WAITING"), ("H", "blocked", "available", "WAITING"),
    ]
    make_table(slide, Inches(0.9), Inches(3.15), Inches(6.4), Inches(3.5),
               ["Cell", "Workload", "Power", "Status"], rows, font_size=12.5,
               highlight_row=None, col_widths=[1, 2, 2, 2])

    panel = rect(slide, Inches(7.65), Inches(3.15), Inches(5.1), Inches(3.5), "", fill=PANEL)
    textbox(slide, Inches(7.95), Inches(3.4), Inches(4.6), Inches(0.4),
            "This is a quota wait, not a data problem", size=14, bold=True, color=PRIMARY)
    bullets(slide, Inches(7.95), Inches(3.9), Inches(4.6), Inches(2.6), [
        "BigQuery Sandbox (free tier) caps monthly bytes scanned; cells A–D "
        "already used most of it.",
        "By explicit project decision: stay on the free tier and wait for the "
        "monthly reset, rather than enable billing.",
        "The extraction query for E–H is written, cost-estimated "
        "(~627 GiB combined), and tested live — it needs zero code changes "
        "once the quota resets.",
        "No synthetic or substitute data has been created for E–H at any point.",
    ], size=12.5, space_after=8)
    add_notes(slide, (
        "SAY: All 8 cells' POWER data and machine metadata are already fully extracted -- it is "
        "only the workload table for cells E-H that is blocked, and only because of a free-tier "
        "query quota, not because the data doesn't exist or is hard to get. We deliberately chose "
        "not to pay for BigQuery access; we're waiting for the free monthly allowance to reset. "
        "LIKELY QUESTION: 'Why not just pay the few dollars to unblock it?' ANSWER: that was an "
        "explicit project decision to stay fully on the free tier; the optimized query already "
        "fits comfortably inside one reset cycle, so paying isn't necessary, just slower."
    ))
    return slide


def s12_architecture(prs):
    slide = blank_slide(prs)
    title_bar(slide, "End-to-End Version 2 Architecture")
    steps_left = [
        ("ClusterData2019 (official)", SECONDARY),
        ("Cell-level workload aggregation", SECONDARY),
        ("Temporal preprocessing + split", SECONDARY),
        ("GRU forecasting", ACCENT),
    ]
    steps_mid = [
        ("PowerData2019 (official)", SECONDARY),
        ("Workload ↔ power alignment", SECONDARY),
        ("Empirical power model", ACCENT),
        ("Heat generation", SECONDARY),
    ]
    x1, x2 = Inches(0.7), Inches(4.85)
    bw, bh, gap = Inches(3.85), Inches(0.5), Inches(0.13)
    top = Inches(1.35)
    vertical_chain(slide, x1, top, bw, bh, gap, steps_left, box_size=11.5)
    vertical_chain(slide, x2, top, bw, bh, gap, steps_mid, box_size=11.5)
    # merge arrows into thermal model
    mid_y = top + 4 * (bh + gap) - gap
    thermal_y = mid_y + Inches(0.28)
    arrow(slide, x1 + bw / 2, mid_y + bh, x1 + bw / 2, thermal_y, color=MUTED)
    arrow(slide, x2 + bw / 2, mid_y + bh, x2 + bw / 2, thermal_y, color=MUTED)
    tx = Inches(0.7)
    tw = Inches(8.0)
    rect(slide, tx, thermal_y, tw, Inches(0.46), "Three-zone thermal environment", fill=PRIMARY, size=12.5, bold=True)
    ctrl_y = thermal_y + Inches(0.46) + Inches(0.2)
    down_arrow(slide, tx + tw / 2, thermal_y + Inches(0.46), Inches(0.2))
    ctrls = ["Fixed", "Threshold", "PID", "MPC", "PPO", "MAPPO"]
    ccw = Inches(1.28)
    total_cw = ccw * 6 + Inches(0.12) * 5
    cx0 = tx + (tw - total_cw) / 2
    for i, c in enumerate(ctrls):
        cx = cx0 + i * (ccw + Inches(0.12))
        fill = ACCENT if c in ("PPO", "MAPPO") else SECONDARY
        rect(slide, cx, ctrl_y, ccw, Inches(0.42), c, fill=fill, size=11, bold=True)
    shield_y = ctrl_y + Inches(0.42) + Inches(0.2)
    down_arrow(slide, tx + tw / 2, ctrl_y + Inches(0.42), Inches(0.2))
    rect(slide, tx, shield_y, tw, Inches(0.4), "Safety shield", fill=DANGER, size=12, bold=True)
    eval_y = shield_y + Inches(0.4) + Inches(0.18)
    down_arrow(slide, tx + tw / 2, shield_y + Inches(0.4), Inches(0.18))
    rect(slide, tx, eval_y, tw, Inches(0.44), "Evaluation → ablations + robustness → results",
         fill=SUCCESS, size=12, bold=True)

    # right side notes panel
    px = Inches(9.0)
    panel = rect(slide, px, Inches(1.5), Inches(3.75), Inches(5.35), "", fill=PANEL)
    textbox(slide, px + Inches(0.25), Inches(1.7), Inches(3.25), Inches(0.4),
            "Read this diagram as:", size=13, bold=True, color=PRIMARY)
    bullets(slide, px + Inches(0.25), Inches(2.15), Inches(3.3), Inches(4.5), [
        "Two independent official Google traces feed the pipeline from the top.",
        "They meet only at the thermal model, through heat generation.",
        "Every controller runs on the SAME environment — required for a fair comparison.",
        "The safety shield sits between every controller and the physics, always.",
        "Orange boxes are the machine-learned components (GRU, power model, PPO, MAPPO).",
    ], size=12, space_after=9)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: This is the whole system in one picture. Two real data sources come down two "
        "parallel tracks (workload/forecast on the left, power on the right), they physically "
        "meet only inside the thermal model as a heat input, and from there every controller -- "
        "classical or learned -- is evaluated through the identical safety shield and evaluation "
        "harness. That 'same environment for every controller' property is what makes the "
        "comparisons on later slides fair. "
        "LIKELY QUESTION: 'Which parts of this are actually built vs. planned?' ANSWER: every box "
        "in this diagram has working code today, validated on cells A-D; the orange PPO/MAPPO "
        "boxes are currently smoke-test scale, not final."
    ))
    return slide


def s13_preprocessing(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Workload Preprocessing, Step by Step")
    steps = [
        "1. Extract workload data (BigQuery, column-pruned, cost-estimated query)",
        "2. Identify relevant resource features (average_usage.cpus, average_usage.memory)",
        "3. Aggregate at Borg-cell level (sum across all active machines per cell, per bucket)",
        "4. Construct the temporal sequence (5-minute buckets, one row per cell per bucket)",
        "5. Handle missing values correctly (join-coverage assertion ≥ 95%, no silent fill)",
        "6. Normalize features (min-max fit on TRAIN split only, applied unchanged to val/test)",
        "7. Chronological train/validation/test split (60% / 20% / 20%, never shuffled)",
    ]
    bullets(slide, Inches(0.6), Inches(1.55), Inches(6.6), Inches(5.3), steps, size=14.5, space_after=14)

    px = Inches(7.5)
    panel = rect(slide, px, Inches(1.55), Inches(5.2), Inches(2.1), "", fill=PANEL)
    textbox(slide, px + Inches(0.25), Inches(1.7), Inches(4.7), Inches(0.4),
            "Why NOT a random split?", size=13.5, bold=True, color=PRIMARY)
    textbox(slide, px + Inches(0.25), Inches(2.15), Inches(4.7), Inches(1.4),
            "A random split lets the model see data from AFTER a test "
            "timestamp during training — information leakage. For a "
            "temporal forecasting problem, only a chronological split "
            "guarantees the model never learns from its own future.",
            size=12.5, line_spacing=1.2)

    y2 = Inches(4.0)
    steps2 = [("Raw workload\nrecords", SECONDARY), ("Cell-level\nfeature table", SECONDARY),
              ("Time-series\nmatrix", SECONDARY), ("Model\nsequences", ACCENT)]
    bw2 = Inches(1.15)
    for i, (label, c) in enumerate(steps2):
        x = px + i * (bw2 + Inches(0.15))
        rect(slide, x, y2, bw2, Inches(0.75), label, fill=c, size=10.5, bold=True)
        if i < len(steps2) - 1:
            right_arrow(slide, x + bw2, y2 + Inches(0.375), Inches(0.13))
    textbox(slide, px, y2 + Inches(0.95), Inches(5.2), Inches(0.4),
            "raw records → features → sequences → model input", size=11, italic=True, color=MUTED)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: Walk down the 7 steps on the left -- this is the literal order of operations in "
        "src/data/v2_build_timeseries.py and v2_split.py. The one step worth pausing on is #7: "
        "unlike a typical ML pipeline, we never shuffle before splitting, because that would let "
        "the model implicitly learn from the future. "
        "LIKELY QUESTION: 'How do you know there's no leakage?' ANSWER: it's not just a design "
        "intent -- tests/test_v2_pipeline.py programmatically checks the split ranges are disjoint "
        "and chronologically ordered, and a separate test checks the GRU's forecast windows never "
        "cross a split boundary."
    ))
    return slide


def s14_forecasting_problem(prs):
    slide = blank_slide(prs)
    title_bar(slide, "The Temporal Forecasting Problem")
    x = Inches(0.9)
    rect(slide, x, Inches(1.7), Inches(4.6), Inches(0.6),
         "Past workload window\n(48 steps = 4 hours of history)", fill=SECONDARY, size=13)
    down_arrow(slide, x + Inches(2.3), Inches(2.3), Inches(0.3))
    rect(slide, x + Inches(1.1), Inches(2.6), Inches(2.4), Inches(0.6), "GRU", fill=ACCENT, size=16, bold=True)
    down_arrow(slide, x + Inches(2.3), Inches(3.2), Inches(0.3))
    rect(slide, x, Inches(3.5), Inches(4.6), Inches(0.6),
         "Future workload\n(6 steps = 30 minutes ahead)", fill=SECONDARY, size=13)

    px = Inches(6.1)
    panel = rect(slide, px, Inches(1.7), Inches(6.6), Inches(4.9), "", fill=PANEL)
    textbox(slide, px + Inches(0.3), Inches(1.9), Inches(6.0), Inches(0.4),
            "Exact pilot configuration", size=14, bold=True, color=PRIMARY)
    rows = [("Input history (lookback)", "48 steps × 5 min = 4 hours"),
            ("Forecast horizon", "6 steps × 5 min = 30 minutes"),
            ("Temporal resolution", "5 minutes (native BigQuery extraction)"),
            ("Target / feature", "sum_cpu (summed CPU demand, per cell)"),
            ("Hidden size / layers", "32 / 1 (GRUForecaster)"),
            ("Normalization", "min-max, fit on TRAIN only")]
    make_table(slide, px + Inches(0.3), Inches(2.35), Inches(6.0), Inches(2.9),
               ["Setting", "Value"], rows, font_size=12.5, col_widths=[2.2, 3.8])
    textbox(slide, px + Inches(0.3), Inches(5.5), Inches(6.0), Inches(0.9),
            "Why forecast at all? A controller with a 30-minute early warning of "
            "rising workload can start cooling proactively instead of reacting "
            "after temperature has already climbed (slide 3).",
            size=12, italic=True, color=MUTED, line_spacing=1.15)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: The GRU's job is narrow and specific: given the last 4 hours of a cell's summed CPU "
        "demand, predict the next 30 minutes. That's it -- it does not forecast power or "
        "temperature directly, only workload; power and temperature are computed downstream from "
        "that forecast plus the fitted power model. "
        "LIKELY QUESTION: 'Why 4 hours in, 30 minutes out?' ANSWER: chosen to mirror V1's real-world "
        "time spans (V1 used 4h lookback / 30min horizon at a coarser 15-min resolution); at V2's "
        "native 5-minute resolution that becomes 48 and 6 steps respectively."
    ))
    return slide


def s15_why_gru(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Why a GRU?")
    x = Inches(0.9)
    steps = ["Input sequence\n(workload history)", "Update / reset gating",
             "Hidden state\n(compressed memory)", "Future workload prediction"]
    vertical_chain(slide, x, Inches(1.7), Inches(4.4), Inches(0.65), Inches(0.25), steps, box_size=13)
    px = Inches(6.0)
    panel = rect(slide, px, Inches(1.7), Inches(6.7), Inches(4.9), "", fill=PANEL)
    textbox(slide, px + Inches(0.3), Inches(1.9), Inches(6.1), Inches(0.4),
            "What a GRU actually gives us", size=14, bold=True, color=PRIMARY)
    bullets(slide, px + Inches(0.3), Inches(2.4), Inches(6.1), Inches(3.9), [
        "Handles sequential dependence: today's prediction can depend on a pattern "
        "seen several hours back, not just the last data point.",
        "Gating (“update/reset”) lets it keep relevant history and discard noise, "
        "instead of treating every past step equally.",
        "Fewer parameters than an LSTM (no separate cell state) — a reasonable, "
        "well-understood choice for a pilot-scale forecasting problem.",
        "NOT claimed to be universally optimal — it is compared against two "
        "simple, transparent baselines on the very next slide, and has to beat "
        "them to be worth using at all.",
    ], size=13, space_after=11)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: A GRU (Gated Recurrent Unit) is a recurrent neural network cell -- it reads a "
        "sequence one step at a time and keeps a running 'hidden state' summary, with gates that "
        "decide what to keep and what to forget. It's a well-established, moderate-complexity "
        "choice: simpler than an LSTM, more expressive than a plain RNN. "
        "LIKELY QUESTION: 'Why not a Transformer or something more modern?' ANSWER: at this "
        "sequence length (48 steps) and pilot data volume, a small GRU is appropriate and "
        "sufficient -- and importantly it DOES beat the simple baselines on the next slide, which "
        "is the actual justification, not the architecture's reputation."
    ))
    return slide


def s16_forecasting_baselines(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Forecasting Baselines — Does the GRU Add Value?")
    path = get_chart("forecast_comparison", chart_forecast_comparison)
    image_slide_full(slide, path, top=Inches(1.55), max_h=Inches(3.7))
    textbox(slide, Inches(0.6), Inches(5.45), Inches(12.1), Inches(0.75),
            "The real question isn't ‘is R² high?’ — it's whether the GRU adds "
            "predictive value over simple strategies. Here it does: lower MAE/RMSE and "
            "higher R² than both persistence and moving-average, on the untouched test split.",
            size=13.5, italic=True, color=INK, line_spacing=1.2)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: 'Persistence' just repeats the last observed value; 'moving average' averages the "
        "last few steps. Both are trivial to compute and set the bar the GRU has to clear. On "
        "this test split it does: MAE 260 vs 290/284, RMSE 367 vs 418/404, R-squared 0.824 vs "
        "0.772/0.787 -- read exact numbers from results/pilot/forecasting/metrics.json. "
        "LIKELY QUESTION: 'Is 0.824 a good R-squared?' ANSWER: for a real, noisy operational "
        "workload signal, yes -- and more importantly it beats the baselines by a clear, "
        "consistent margin on every metric, which is the more meaningful bar than an absolute "
        "threshold."
    ))
    return slide


def s17_gru_interpretation(prs):
    slide = blank_slide(prs)
    title_bar(slide, "What the Pilot GRU Result Actually Means")
    path = get_chart("forecast_by_regime", chart_forecast_by_regime)
    slide.shapes.add_picture(str(path), Inches(0.6), Inches(1.55), height=Inches(3.35))
    px = Inches(7.5)
    bullets(slide, px, Inches(1.55), Inches(5.3), Inches(3.5), [
        "The workload contains real, learnable temporal structure — it is not noise.",
        "GRU currently outperforms both pilot baselines on every metric on the "
        "held-out test split.",
        "This is substantially stronger than V1's forecasting result (R² near zero "
        "on the sparse Kaggle signal).",
        "Forecast quality degrades at burst load (top decile) — GRU R² drops to "
        "0.18 there, still ahead of persistence's -0.62.",
        "A–D is not the final 8-cell population — this is evidence of "
        "FEASIBILITY, not the final research claim.",
    ], size=13, space_after=9)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: Two things are true at once here: the forecaster genuinely works, and it works "
        "less well exactly where it matters most for safety -- sudden bursts. That's an honest "
        "limitation, not something to gloss over: proximity events near the safety limit "
        "correlate with exactly the regime (burst load) where the forecast is weakest. "
        "LIKELY QUESTION: 'So is forecasting actually useful for control, or not?' ANSWER: this "
        "slide only establishes forecast QUALITY; whether it changes CONTROL behavior is a "
        "separate question tested by the forecast on/off ablation, which is planned for the final "
        "8-cell run (V1's own forecast ablation found no measurable control benefit, so this is a "
        "real open question, not assumed to be positive)."
    ))
    return slide


def s18_power_model(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Empirical Power Model")
    x = Inches(0.7)
    steps = ["Workload (sum_cpu, sum_mem, per cell)", "Empirical model\n(fitted on TRAIN)",
             "Estimated power utilization", "Heat generation (× heat_scale_kw)"]
    vertical_chain(slide, x, Inches(1.55), Inches(3.9), Inches(0.62), Inches(0.2), steps, box_size=12)
    path = get_chart("power_model_comparison", chart_power_model_comparison)
    slide.shapes.add_picture(str(path), Inches(5.1), Inches(1.55), height=Inches(3.5))
    textbox(slide, Inches(0.7), Inches(5.3), Inches(11.9), Inches(1.7),
            "V2 anchors this model to OFFICIAL MEASURED Google power data (PowerData2019) — "
            "V1 had none and used a pure assumption. Four candidates were compared, "
            "selected on VALIDATION only: a Random Forest (sum_cpu, sum_mem, cell) wins "
            "with R² ≈ 0.780, MAE ≈ 0.014, RMSE ≈ 0.018 (fraction of PDU capacity). "
            "Residuals show a modest positive bias at high load (mean +0.015) — disclosed, "
            "not hidden.",
            size=13.5, line_spacing=1.25)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: This is a genuinely new capability V2 has that V1 never did: a REAL empirical "
        "relationship between workload and measured power. We tried four candidates in "
        "increasing complexity -- a constant baseline, a pooled linear model, a linear model with "
        "a per-cell fixed effect, and a random forest -- and picked the winner purely on "
        "validation RMSE, never touching the test split. "
        "LIKELY QUESTION: 'Why does a per-cell fixed effect help so much (0.648 to 0.714 R2)?' "
        "ANSWER: different cells likely have different numbers of PDUs / baseline power draw; "
        "the feature-importance breakdown in the appendix shows cell identity carries real "
        "predictive weight (36.7%), almost as much as CPU itself (58.4%)."
    ))
    return slide


def s19_why_power_matters(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Why Real Power Data Matters")
    y0 = Inches(1.7)
    colw = Inches(5.7)
    x1, x2 = Inches(0.7), Inches(6.9)
    rect(slide, x1, y0, colw, Inches(0.5), "VERSION 1", fill=MUTED, size=14, bold=True)
    v1steps = ["CPU / workload utilization", "ASSUMED linear formula\n(never measured)", "Simulated power"]
    vertical_chain(slide, x1, y0 + Inches(0.65), colw, Inches(0.55), Inches(0.2), v1steps, box_size=12.5)

    rect(slide, x2, y0, colw, Inches(0.5), "VERSION 2", fill=PRIMARY, size=14, bold=True)
    v2steps = [("Official Google workload", SECONDARY), ("Empirical model\n(fitted, R²≈0.780)", ACCENT),
               ("Official MEASURED power\nreference (PowerData2019)", SECONDARY)]
    vertical_chain(slide, x2, y0 + Inches(0.65), colw, Inches(0.55), Inches(0.2), v2steps, box_size=12.5)

    py = y0 + Inches(0.65) + 3 * (Inches(0.55) + Inches(0.2))
    panel = rect(slide, x1, py + Inches(0.1), colw * 2 + Inches(0.5), Inches(1.55), "", fill=PANEL)
    bullets(slide, x1 + Inches(0.3), py + Inches(0.3), colw * 2, Inches(1.3), [
        "The power model is now anchored to measured Google power data, not an assumption.",
        "This does NOT automatically make the thermal model physically measured — "
        "model quality still has to be evaluated on its own (previous slide's residuals).",
    ], size=13, space_after=8)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: Side by side, the difference is that V1's middle step was a guess and V2's is a "
        "fitted model checked against real measurements. That is the specific scientific upgrade "
        "this slide is making, no more and no less. "
        "LIKELY QUESTION: 'Does this mean the thermal simulation is now realistic?' ANSWER: no -- "
        "that's exactly the caution in the panel below. A better power model does not "
        "automatically validate the downstream thermal simulation; that's addressed separately, "
        "next."
    ))
    return slide


def s20_thermal_model(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Three-Zone Thermal Model")
    cx, cy = Inches(3.2), Inches(3.4)
    r = Inches(1.05)
    positions = {
        "Zone 1\n(cell a)": (cx - Inches(1.9), cy - Inches(0.7)),
        "Zone 2\n(cell b)": (cx + Inches(0.4), cy - Inches(1.3)),
        "Zone 3\n(cell c)": (cx + Inches(0.4), cy + Inches(0.4)),
    }
    coords = {}
    for label, (x, y) in positions.items():
        shp = rect(slide, x, y, Inches(1.7), Inches(0.85), label, fill=SECONDARY, size=12.5, bold=True,
                   shape_type=MSO_SHAPE.OVAL)
        coords[label] = (x, y)
    keys = list(positions.keys())
    for a, b in [(0, 1), (1, 2)]:
        xa, ya = positions[keys[a]]
        xb, yb = positions[keys[b]]
        arrow(slide, xa + Inches(1.7), ya + Inches(0.42), xb, yb + Inches(0.42), color=MUTED)
        arrow(slide, xb, yb + Inches(0.5), xa + Inches(1.7), ya + Inches(0.5), color=MUTED)
    textbox(slide, cx - Inches(2.0), cy + Inches(1.4), Inches(4.3), Inches(0.4),
            "(4th zone, cell d, omitted here for clarity)", size=10.5, italic=True, color=MUTED)

    px = Inches(6.9)
    panel = rect(slide, px, Inches(1.5), Inches(5.85), Inches(3.55), "", fill=PANEL)
    textbox(slide, px + Inches(0.3), Inches(1.7), Inches(5.3), Inches(0.4),
            "Per zone, every step:", size=13.5, bold=True, color=PRIMARY)
    bullets(slide, px + Inches(0.3), Inches(2.15), Inches(5.3), Inches(2.7), [
        "workload → power/heat (from the fitted model)",
        "heat + cooling action → temperature update",
        "coupling with neighboring zones (heat can move between them)",
        "cooling action reduces temperature, at an assumed max rate",
    ], size=13, space_after=8)

    why = rect(slide, Inches(0.7), Inches(5.3), Inches(12.0), Inches(0.55),
               "Why zones, why 4 (not a fixed 3)?", fill=PRIMARY, size=13, bold=True)
    textbox(slide, Inches(0.7), Inches(5.95), Inches(12.0), Inches(0.9),
            "Zones enable spatially distributed control and a genuine multi-agent formulation. "
            "V2 sets zones = Borg cells directly (4 in the pilot, 8 in the final run) instead of "
            "V1's fixed, synthetic 3-zone split — because cells are a real partition, not an "
            "invented one.",
            size=12.5, line_spacing=1.2)
    critical = rect(slide, Inches(0.7), Inches(6.9), Inches(12.0), Inches(0.5),
                     "The zones are LOGICAL SIMULATION zones — NOT claimed to represent three "
                     "(or four) known physical Google rooms or racks.", fill=DANGER, size=12.5, bold=True)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: The diagram shows the general topology -- each zone has its own heat and cooling, "
        "plus a coupling term with its neighbors so heat can spread. The genuinely new decision "
        "in V2 is using the Borg CELL as the zone unit instead of V1's synthetic hash-based split "
        "-- because we now have a real partition to use, re-inventing an arbitrary one would be a "
        "step backward. "
        "LIKELY QUESTION: 'Are these real Google temperature measurements?' ANSWER: No. The "
        "workload and power references are official Google data, but the three/four-zone thermal "
        "environment itself is a simulation abstraction built to study the control problem -- "
        "there is no real temperature data anywhere in either Google trace."
    ))
    return slide


def s21_thermal_calibration(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Thermal Calibration")
    path = get_chart("thermal_sweep", chart_thermal_sweep)
    slide.shapes.add_picture(str(path), Inches(0.6), Inches(1.6), width=Inches(6.9))
    px = Inches(7.85)
    bullets(slide, px, Inches(1.5), Inches(4.9), Inches(4.3), [
        "Selected value: heat_scale_kw = 0.85 — the smallest tested value where "
        "cooling OFF already violates the safety limit, while cooling FULL stays "
        "comfortably safe (a genuinely non-trivial control problem).",
        "A real bug was found and fixed during calibration: the first attempt "
        "measured “cooling off” with the safety shield still silently active, "
        "which secretly corrected the “off” action.",
        "After disabling the shield explicitly, the sweep was repeated on a "
        "dense, 251-window scan (vs. the original 14 points) — same conclusion, "
        "on much stronger evidence.",
        "A regression test now locks this in "
        "(test_open_loop_cooling_off_is_invariant_to_cooling_max_kw).",
    ], size=12.5, space_after=10)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: This chart is read left to right as we turn up an assumed heat-intensity parameter. "
        "The red line (cooling off) crosses the safety limit almost immediately; the green line "
        "(full cooling) stays safe until about 0.9, where it also starts failing -- meaning the "
        "cooling capacity itself becomes insufficient beyond that point. 0.85 sits right at the "
        "edge: violates when off, safe when full. "
        "LIKELY QUESTION: 'Why should I trust a self-reported bug fix?' ANSWER: because we're "
        "showing you exactly how it was caught -- an impossible symptom (the 'off' trajectory "
        "changing when a cooling-capacity parameter was varied, which should be physically "
        "impossible for action=0) -- and the fix is now a permanent automated test, not just a "
        "one-time patch."
    ))
    return slide


def s22_thermal_finding(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Important Thermal Finding: PID Still Violates 34% of Steps")
    path = get_chart("pid_trajectory", chart_pid_trajectory)
    image_slide_full(slide, path, top=Inches(1.5), max_h=Inches(3.15))
    bullets(slide, Inches(0.7), Inches(4.85), Inches(12.0), Inches(2.2), [
        "With the safety shield ON, a well-tuned PID controller still spends 33.9% of test-split "
        "steps above the 27°C safety limit (results/pilot/controllers/classical_controller_comparison.json).",
        "Why: thermal dynamics have lag, the actuator is rate-limited (max change per step), "
        "PID only responds once temperature error is already large (reactive, not proactive), "
        "and the safety shield is a ONE-STEP-AHEAD correction — it cannot undo several steps "
        "of prior lag in a single move.",
        "This result was KEPT, not tuned away. Re-calibrating the thermal parameters until PID "
        "showed zero violations would have altered the evidence to fit a preferred story.",
    ], size=13.5, space_after=10)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: This is a finding we're choosing to show you plainly rather than hide. The chart is "
        "one real simulated 24-hour test episode: four cell temperatures over time, the dashed "
        "red line is the safety limit, and the shaded band marks where at least one zone is over "
        "it. 83 of 288 steps in this specific episode are above the line. "
        "LIKELY QUESTION: 'Why didn't you simply tune the model until PID had zero violations?' "
        "ANSWER: because that would alter the evidence to obtain a desired result. The dense "
        "calibration (previous slide) showed this behavior is tied to the actual thermal/control "
        "dynamics -- reactive lag plus a myopic one-step shield -- so it was retained and "
        "documented, exactly as the instructions for this project require."
    ))
    return slide


def s23_control_problem(prs):
    slide = blank_slide(prs)
    title_bar(slide, "The Control Problem, Formally")
    x = Inches(1.2)
    steps = ["OBSERVATION\n(temps, workload, forecast, margin to limit, prev action, time-of-day)",
             "CONTROLLER", "ACTION\n(cooling level per zone, in [0, 1])",
             "ENVIRONMENT TRANSITION\n(heat + cooling → next temperature)"]
    vertical_chain(slide, x, Inches(1.55), Inches(6.8), Inches(0.72), Inches(0.22), steps, box_size=13)

    px = Inches(8.5)
    panel = rect(slide, px, Inches(1.55), Inches(4.2), Inches(5.2), "", fill=PANEL)
    textbox(slide, px + Inches(0.25), Inches(1.75), Inches(3.7), Inches(0.4),
            "Constraints", size=14, bold=True, color=PRIMARY)
    bullets(slide, px + Inches(0.25), Inches(2.2), Inches(3.7), Inches(1.9), [
        "Action bounds: cooling ∈ [0, 1] per zone",
        "Rate limit: |a_t − a_{t-1}| ≤ 0.3 per step",
        "Thermal safety limit: 27°C (soft penalty from 4°C below; safety shield near the limit)",
    ], size=12, space_after=8)
    textbox(slide, px + Inches(0.25), Inches(4.3), Inches(3.7), Inches(0.4),
            "The trade-off", size=14, bold=True, color=PRIMARY)
    bullets(slide, px + Inches(0.25), Inches(4.75), Inches(3.7), Inches(1.9), [
        "energy (minimize)",
        "thermal safety (respect the limit)",
        "smoothness (avoid chattering actions)",
    ], size=12, space_after=8)
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: Every controller in this project — classical or learned — is solving exactly this "
        "loop with exactly these constraints; that shared definition is what makes the comparisons "
        "on the next few slides apples-to-apples. "
        "LIKELY QUESTION: 'What exactly is in the observation?' ANSWER: current temperatures, "
        "current workload, the GRU's forecast, the safety margin remaining, the previous action "
        "(for smoothness), and a sine/cosine time-of-day encoding -- listed in full in the "
        "appendix."
    ))
    return slide


def s24_classical_controllers(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Classical Controllers")
    defs = [
        ("Fixed", "Constant cooling level, regardless of state.", SECONDARY),
        ("Threshold", "Switches cooling on/off when temperature crosses fixed points.", SECONDARY),
        ("PID", "Cools proportional to error, its accumulated history, and its rate of change.", PRIMARY),
        ("MPC", "Simulates several candidate action sequences ahead and picks the best.", SECONDARY),
    ]
    x0 = Inches(0.6)
    cw = Inches(3.0)
    for i, (name, desc, c) in enumerate(defs):
        x = x0 + i * (cw + Inches(0.13))
        rect(slide, x, Inches(1.55), cw, Inches(0.55), name, fill=c, size=15, bold=True)
        textbox(slide, x, Inches(2.2), cw, Inches(1.5), desc, size=12, line_spacing=1.2)
    path = get_chart("controller_comparison", chart_controller_comparison)
    slide.shapes.add_picture(str(path), Inches(0.6), Inches(3.85), height=Inches(3.15))
    pilot_tag(slide)
    add_notes(slide, (
        "SAY: Four classical controllers, ordered roughly by sophistication. PID is the strongest "
        "of the four (lowest violation rate at 33.9%, per the chart) because it responds to error, "
        "the trend of that error, and its accumulated history, not just an instantaneous "
        "threshold. MPC here is a documented random-shooting search over 64 candidate action "
        "sequences per step, not a full gradient/CEM solver -- a disclosed simplification, not "
        "hidden. "
        "LIKELY QUESTION: 'Why does MPC not obviously beat PID here?' ANSWER: MPC's heat forecast "
        "over its horizon is held constant (a documented simplification, since the fitted power "
        "model needs a memory-usage feature that isn't forecasted) -- so it isn't exploiting "
        "future information the way a full model-predictive approach ideally would."
    ))
    return slide


def s25_ppo(prs):
    slide = blank_slide(prs)
    title_bar(slide, "PPO — Proximal Policy Optimization")
    x = Inches(0.7)
    steps = ["State", "Policy", "Action", "Environment", "Reward", "Policy update (clipped)"]
    vertical_chain(slide, x, Inches(1.5), Inches(3.9), Inches(0.48), Inches(0.13), steps, box_size=12)
    px = Inches(5.1)
    panel = rect(slide, px, Inches(1.5), Inches(3.0), Inches(3.55), "", fill=PANEL)
    textbox(slide, px + Inches(0.2), Inches(1.65), Inches(2.6), Inches(1.9),
            "Core idea:\n\nUpdate the policy toward actions with "
            "higher-than-expected return (the ‘advantage’), but "
            "CLIP how far a single update can move it — keeps training stable.",
            size=11.5, line_spacing=1.25)
    path = get_chart("ppo_training", chart_ppo_training)
    slide.shapes.add_picture(str(path), Inches(8.35), Inches(1.5), width=Inches(4.35))
    warn = rect(slide, Inches(0.7), Inches(5.35), Inches(11.9), Inches(0.75),
                "SMOKE TEST ONLY — 8,640 timesteps (V1 used 100,000). This validates the code "
                "path executes correctly on real data; it is NOT a final PPO result.",
                fill=DANGER, size=13, bold=True)
    add_notes(slide, (
        "SAY: PPO is single-agent here: one policy observes and controls ALL zones jointly. The "
        "training curve on the right is genuine data from this pilot run, but at 8,640 timesteps "
        "it's a small fraction of a real budget -- read it as 'the wiring works', not 'this is how "
        "good PPO is'. "
        "LIKELY QUESTION: 'What's the clipping math?' ANSWER: the objective clips the probability "
        "ratio between new and old policy to [1-epsilon, 1+epsilon] (epsilon=0.2 here) before "
        "multiplying by the advantage -- full formulation in the appendix."
    ))
    return slide


def s26_mappo(prs):
    slide = blank_slide(prs)
    title_bar(slide, "MAPPO — Multi-Agent PPO with CTDE")
    gx, gy = Inches(4.9), Inches(1.55)
    rect(slide, gx, gy, Inches(3.5), Inches(0.55), "GLOBAL STATE (training only)", fill=PRIMARY, size=12, bold=True)
    down_arrow(slide, gx + Inches(1.75), gy + Inches(0.55), Inches(0.22))
    rect(slide, gx, gy + Inches(0.77), Inches(3.5), Inches(0.55), "CENTRALIZED CRITIC", fill=PRIMARY, size=12, bold=True)
    agents_y = gy + Inches(0.77) + Inches(0.55) + Inches(0.35)
    agent_x0 = Inches(1.3)
    aw = Inches(2.7)
    for i in range(4):
        ax = agent_x0 + i * (aw + Inches(0.35))
        cx_top = gx + Inches(1.75)
        cy_top = gy + Inches(0.77) + Inches(0.55)
        conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, cx_top, cy_top, ax + aw / 2, agents_y)
        conn.line.color.rgb = MUTED; conn.line.width = Pt(1.3)
        rect(slide, ax, agents_y, aw, Inches(0.5), f"Agent {i+1}\n(actor, shared weights)",
             fill=SECONDARY, size=10.5, bold=True)
        down_arrow(slide, ax + aw / 2, agents_y + Inches(0.5), Inches(0.2))
        rect(slide, ax, agents_y + Inches(0.7), aw, Inches(0.42), f"Action {i+1}", fill=ACCENT, size=10.5)
        down_arrow(slide, ax + aw / 2, agents_y + Inches(0.7) + Inches(0.42), Inches(0.2))
        rect(slide, ax, agents_y + Inches(1.32), aw, Inches(0.42), f"Zone {i+1} (cell {chr(97+i)})",
             fill=MUTED, size=10.5)

    y2 = agents_y + Inches(1.32) + Inches(0.42) + Inches(0.25)
    rect(slide, Inches(0.7), y2, Inches(5.9), Inches(0.9),
         ["TRAINING:", "centralized critic sees the concatenated", "global state of all 4 zones"],
         fill=PRIMARY, size=11.5)
    rect(slide, Inches(6.75), y2, Inches(5.9), Inches(0.9),
         ["EXECUTION:", "each agent acts from its OWN local", "observation only — no global info"],
         fill=SECONDARY, size=11.5)
    textbox(slide, Inches(0.7), y2 + Inches(1.0), Inches(11.9), Inches(0.4),
            "CTDE = Centralized Training, Decentralized Execution. This is genuine multi-agent MAPPO — "
            "NOT one PPO policy with four output values.", size=13, bold=True, color=PRIMARY)
    warn = rect(slide, Inches(0.7), y2 + Inches(1.5), Inches(11.9), Inches(0.55),
                "SMOKE TEST ONLY — 30 updates (V1 used 300). NOT a final result.", fill=DANGER, size=12.5, bold=True)
    add_notes(slide, (
        "SAY: Each of the 4 agents (one per cell) has its own actor network with SHARED weights, "
        "and sees only its own zone's local observation when acting -- that's decentralized "
        "execution. During TRAINING ONLY, a separate centralized critic sees the concatenation of "
        "all four zones' observations, so it can learn how the zones interact even though the "
        "actors themselves never do at execution time. "
        "LIKELY QUESTION: 'How do you know this is real MAPPO and not just relabeled PPO?' "
        "ANSWER: it's structurally verified by test -- the actor's input dimension (9, one zone) "
        "is asserted to differ from the critic's input dimension (36, all four zones concatenated) "
        "in tests/test_marl.py, and each agent's action is confirmed to depend only on its own "
        "local observation, never the others'."
    ))
    return slide


def s27_safety_shield(prs):
    slide = blank_slide(prs)
    title_bar(slide, "The Safety Shield")
    x = Inches(0.8)
    steps = ["RL / controller\nproposed action", "SAFETY CHECK\n(predicted next temp vs. limit)",
             "Safe action\n(possibly corrected)", "Thermal environment"]
    vertical_chain(slide, x, Inches(1.55), Inches(4.3), Inches(0.55), Inches(0.16), steps, box_size=12.5)

    px = Inches(5.7)
    panel = rect(slide, px, Inches(1.55), Inches(7.05), Inches(2.75), "", fill=PANEL)
    textbox(slide, px + Inches(0.3), Inches(1.75), Inches(6.4), Inches(0.4),
            "What it does", size=13.5, bold=True, color=PRIMARY)
    bullets(slide, px + Inches(0.3), Inches(2.2), Inches(6.4), Inches(2.0), [
        "One-step-ahead, closed-form check: would the proposed action leave a zone "
        "above (limit − margin)?",
        "If so, raises cooling to the minimum needed to reach that threshold "
        "(bounded by actuator limits) — not a general MPC solver.",
        "Every intervention is logged: which zone, how much correction, resulting temperature.",
    ], size=12.5, space_after=8)
    path = get_chart("shield_ablation", chart_shield_ablation)
    slide.shapes.add_picture(str(path), Inches(0.7), Inches(4.45), width=Inches(4.3))
    key = rect(slide, Inches(8.2), Inches(4.9), Inches(4.5), Inches(2.0),
               ["“Shield intervention” ≠ “intrinsically safe policy”", "",
                "Fixed/Threshold need the shield heavily.",
                "PID's shield-on/off results are IDENTICAL —",
                "not because PID is proactively safe, but",
                "because cooling is already saturated (slide 22)."],
               fill=PANEL, text_color=INK, size=11.5)
    add_notes(slide, (
        "SAY: The shield sits between every controller's decision and the physics -- it never "
        "changes the reward function, only the applied action, and every correction is logged so "
        "reliance on it can be measured, not assumed. "
        "LIKELY QUESTION: 'Does the shield eliminate all safety problems?' ANSWER: no, and that's "
        "the point of the ablation chart -- Fixed and Threshold lean on it heavily (25 and 21 "
        "percentage points worse without it), while PID shows no change either way. Read that "
        "last one carefully: it does NOT mean PID is inherently safe -- our thermal calibration "
        "shows PID's cooling is already maxed out in the regime that causes its violations, so "
        "the shield literally has nothing left to give it."
    ))
    return slide


def s28_validation(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Software / Research Validation")
    big = rect(slide, Inches(0.7), Inches(1.55), Inches(3.6), Inches(1.5), ["62 / 62", "tests passing"],
               fill=SUCCESS, size=30, bold=True)
    rows = [
        ("Join integrity", "no duplicate (cell, bucket) rows; ≥ 95% workload/power match"),
        ("Chronological split", "train < val < test bucket ranges, verified disjoint"),
        ("Causality / leakage", "forecast windows never cross a split; degenerate bins zeroed"),
        ("Heat-generation correctness", "heat = predicted_power_util × heat_scale_kw, asserted exactly"),
        ("Configuration consistency", "single source of truth for active cell set (new, see below)"),
        ("Active-cell / n_zones consistency", "n_zones DERIVED from loaded data, not configured"),
        ("Pilot/final artifact separation", "results/<cell_set>/... and models/v2/<cell_set>/..."),
        ("Extraction readiness", "8-cell extraction script verified live, no code changes needed"),
    ]
    make_table(slide, Inches(4.6), Inches(1.55), Inches(8.2), Inches(3.7),
               ["Category", "What's checked"], rows, font_size=11.5, col_widths=[3.0, 5.2])
    panel = rect(slide, Inches(0.7), Inches(3.35), Inches(3.6), Inches(3.7), "", fill=PANEL)
    textbox(slide, Inches(0.9), Inches(3.55), Inches(3.2), Inches(0.7),
            "A real bug this caught:", size=13, bold=True, color=PRIMARY, line_spacing=1.1)
    textbox(slide, Inches(0.9), Inches(4.25), Inches(3.2), Inches(2.6),
            "configs/v2_config.yaml independently duplicated the active-cell-set "
            "setting and manually set n_zones=4. Fixed: one source of truth, "
            "n_zones derived from data, pilot/final paths isolated — all locked "
            "in by new regression tests.",
            size=11.5, line_spacing=1.25, color=INK)
    add_notes(slide, (
        "SAY: This slide exists to show engineering rigor, not just a pass/fail count. The "
        "specific bug called out on the left is real: before a final-pipeline readiness audit, "
        "two config files could disagree about which cells were active, and the final run could "
        "have silently overwritten pilot results at the same file path. It was found, fixed, and "
        "turned into a permanent test rather than a one-off patch. "
        "LIKELY QUESTION: 'Do these 62 tests include anything about PPO/MAPPO correctness?' "
        "ANSWER: yes, inherited from V1 -- 8 dedicated tests verify MAPPO's actor only sees local "
        "observations while the critic sees the full global state, and that the PPO importance-"
        "sampling ratio is computed correctly (a real bug found and fixed during V1)."
    ))
    return slide


def s29_evidence_table(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Current Results: What Has Actually Been Proven")
    rows = [
        ("Data alignment", "Borg-cell join validated, ≥ 95% coverage per cell", "PILOT PIPELINE"),
        ("GRU forecasting", "test R² ≈ 0.824 (vs. 0.772 / 0.787 baselines)", "PILOT"),
        ("Empirical power model", "validation R² ≈ 0.780 (Random Forest)", "PILOT"),
        ("Thermal model", "calibrated + validated, with a disclosed limitation", "PILOT"),
        ("Classical control", "Fixed/Threshold/PID evaluated, 20 test episodes", "PILOT"),
        ("MPC", "implemented, evaluated (documented simplification)", "PILOT"),
        ("PPO", "runs correctly end-to-end", "SMOKE TEST — NOT FINAL"),
        ("MAPPO (CTDE)", "runs correctly end-to-end, structurally verified", "SMOKE TEST — NOT FINAL"),
        ("Safety shield", "implemented + ON/OFF ablation with reliance analysis", "PILOT"),
        ("Engineering tests", "62 / 62 passing", "VERIFIED"),
    ]
    make_table(slide, Inches(0.6), Inches(1.55), Inches(12.1), Inches(5.4),
               ["Component", "Current evidence", "Status"], rows, font_size=12.5,
               col_widths=[2.6, 6.0, 3.5])
    add_notes(slide, (
        "SAY: This table is the honest summary of the whole deck so far -- read the STATUS "
        "column carefully, it is not uniform. Most rows say PILOT, meaning validated on 4-of-8 "
        "cells; two rows (PPO, MAPPO) are explicitly downgraded further to SMOKE TEST because "
        "their training budget is a small fraction of even V1's reduced budget. "
        "LIKELY QUESTION: 'Which of these numbers would you bet on holding up on 8 cells?' "
        "ANSWER: the data-alignment and engineering-test rows are structural, not statistical -- "
        "they should hold. The GRU/power-model R2 values are real pilot measurements but the "
        "final ones must be re-measured on the complete population, not assumed unchanged."
    ))
    return slide


def s30_done_not_done(prs):
    slide = blank_slide(prs)
    title_bar(slide, "What Is Done / What Is Not Yet Final")
    colw = Inches(5.85)
    x1, x2 = Inches(0.6), Inches(6.7)
    rect(slide, x1, Inches(1.55), colw, Inches(0.5), "COMPLETED", fill=SUCCESS, size=15, bold=True)
    bullets(slide, x1, Inches(2.2), colw, Inches(5.0), [
        "Official data audit + alignment audit", "Cells A–D extraction",
        "All 8 cells' power + machine data extraction", "Preprocessing pipeline",
        "GRU forecaster", "Empirical power model", "Thermal calibration (pilot)",
        "Classical controllers (Fixed/Threshold/PID)", "MPC", "Safety-shield infrastructure + ablation",
        "Robustness infrastructure", "62/62 engineering tests", "Git/GitHub version control (V1 preserved)",
    ], size=12.5, space_after=6)
    rect(slide, x2, Inches(1.55), colw, Inches(0.5), "NOT YET FINAL", fill=ACCENT, size=15, bold=True)
    bullets(slide, x2, Inches(2.2), colw, Inches(5.0), [
        "Cells E–H workload extraction", "Complete A–H dataset",
        "Final model selection (on val, all 8 cells)", "Full PPO training budget",
        "Full MAPPO training budget", "Final multi-seed evaluation (≥ 5 seeds)",
        "Full ablations (forecast on/off, shield on/off) on final data",
        "Final robustness study", "Final untouched test-set evaluation",
        "Final independent audit", "Final research report (DOCX) and final PPTX",
    ], size=12.5, space_after=6)
    add_notes(slide, (
        "SAY: Two honest columns, no overlap. Everything on the left is genuinely finished "
        "engineering work you can inspect in the repository today; everything on the right "
        "requires the complete 8-cell dataset and has not been attempted on 4-of-8 cells as a "
        "substitute. "
        "LIKELY QUESTION: 'What's the single biggest remaining risk?' ANSWER: that PPO/MAPPO, "
        "even with a full training budget, don't beat classical control -- V1 already found "
        "exactly that outcome once, so it's a live possibility we're not assuming away."
    ))
    return slide


def s31_why_not_final(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Why We Are Not Claiming Final Results Yet")
    textbox(slide, Inches(0.6), Inches(1.6), Inches(12.1), Inches(0.6),
            "A–D is 4 of 8 cells — half the intended final population.",
            size=19, bold=True, color=PRIMARY)
    bullets(slide, Inches(0.6), Inches(2.5), Inches(12.0), Inches(2.2), [
        "Repeatedly tuning thermal parameters, rewards, and hyperparameters against A–D and "
        "then reporting that as the final research conclusion would risk OVERFITTING to the "
        "pilot population.",
        "Model/controller selection decisions made only on A–D could be BIASED in ways that "
        "don't hold on E–H.",
        "Presenting pilot smoke-test PPO/MAPPO numbers as a real comparison would be MISLEADING "
        "— they used a fraction of a real training budget.",
    ], size=15, space_after=14)
    y = Inches(5.1)
    rect(slide, Inches(2.5), y, Inches(3.2), Inches(0.65), "A–D", fill=ACCENT, size=16, bold=True)
    right_arrow(slide, Inches(5.7), y + Inches(0.325), Inches(0.5))
    rect(slide, Inches(6.35), y, Inches(3.2), Inches(0.65), "DEVELOPMENT", fill=ACCENT, size=15, bold=True)
    y2 = y + Inches(0.95)
    rect(slide, Inches(2.5), y2, Inches(3.2), Inches(0.65), "A–H", fill=SUCCESS, size=16, bold=True)
    right_arrow(slide, Inches(5.7), y2 + Inches(0.325), Inches(0.5))
    rect(slide, Inches(6.35), y2, Inches(3.2), Inches(0.65), "FINAL EVALUATION", fill=SUCCESS, size=15, bold=True)
    add_notes(slide, (
        "SAY: This is a methodology safeguard, not a hedge. If we let ourselves tune everything "
        "against the same 4 cells we then report results on, any positive finding becomes suspect "
        "-- indistinguishable from fitting to that specific pilot population. Keeping A-D strictly "
        "as development data and reserving A-H for one final, frozen evaluation is what keeps the "
        "eventual final claim defensible. "
        "LIKELY QUESTION: 'Isn't this overly cautious for a progress presentation?' ANSWER: this "
        "same discipline is exactly what let V1 report an honest negative result (PID beating RL) "
        "instead of an inflated one -- it's a deliberate scientific-integrity choice, not delay "
        "for its own sake."
    ))
    return slide


def s32_final_plan(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Final Experiment Plan (Once Cells E–H Are Available)")
    steps = [
        "E–H extraction (script ready, verified live)",
        "A–H dataset validation (same checks as slide 10, all 8 cells)",
        "Final chronological train / validation / test split",
        "Training/validation model selection (forecaster, power model, thermal recalibration)",
        "Freeze the final configuration",
        "Untouched final test evaluation (once)",
        "Fixed / Threshold / PID / MPC",
        "PPO / MAPPO, ≥ 5 seeds where computationally feasible",
        "Forecast ablation · shield ablation · reward sensitivity · thermal sensitivity",
        "Workload robustness · forecast-noise robustness",
        "Independent final audit",
    ]
    x = Inches(2.9)
    bw, bh, gap = Inches(7.5), Inches(0.42), Inches(0.09)
    vertical_chain(slide, x, Inches(1.42), bw, bh, gap, steps, box_size=11.5)
    add_notes(slide, (
        "SAY: This is the exact sequence, in order, that fires once the BigQuery quota resets -- "
        "not a vague future plan. Two rules govern it: model selection only ever touches "
        "train/validation, and the final test split is evaluated exactly once, after everything "
        "else is frozen. "
        "LIKELY QUESTION: 'How long will this take once E-H unblocks?' ANSWER: the mechanical "
        "steps (extraction, preprocessing, refitting the forecaster/power model) are fast (all "
        "pilot equivalents ran in minutes); the RL training at a REAL budget (5 seeds x PPO and "
        "MAPPO) is the dominant cost, matching V1's own multi-hour CPU training time."
    ))
    return slide


def s33_conclusion(prs):
    slide = blank_slide(prs, bg=PRIMARY)
    textbox(slide, Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.7),
            "Current Research Status", size=32, bold=True, color=WHITE, font=FONT_HEAD)
    textbox(slide, Inches(0.8), Inches(1.75), Inches(11.5), Inches(2.5),
            "The Version 2 pipeline has been implemented and technically validated on a "
            "real-data pilot using Google Borg cells A–D. The pilot demonstrates feasible "
            "workload forecasting, empirical power modeling, thermal-control infrastructure, "
            "and a tested PPO/MAPPO pipeline. However, the complete eight-cell experiment is "
            "not yet finished, so the current numerical results are development evidence "
            "rather than final research conclusions.",
            size=18, color=RGBColor(0xE6, 0xEE, 0xF4), line_spacing=1.35)
    rect(slide, Inches(0.8), Inches(4.6), Inches(11.5), Inches(1.15),
         ["IMMEDIATE NEXT STEP", "Complete E–H extraction and re-run the entire final experiment "
          "on all eight cells."],
         fill=ACCENT, size=16, bold=True)
    textbox(slide, Inches(0.8), Inches(6.2), Inches(11.5), Inches(0.5),
            "v2-official-google-data @ b83ce2a  ·  62/62 tests passing", size=12.5,
            color=RGBColor(0x9F, 0xB6, 0xC8))
    add_notes(slide, (
        "SAY: read the paragraph on this slide close to verbatim -- it is the deliberately "
        "precise summary of where the project actually stands, phrased to avoid overclaiming. "
        "LIKELY QUESTION: 'So what's the headline result of this project?' ANSWER: there isn't "
        "one yet, by design -- the headline result is deferred until the full 8-cell run "
        "completes; what exists today is validated infrastructure and early feasibility evidence."
    ))
    return slide


def s34_references(prs):
    slide = blank_slide(prs)
    title_bar(slide, "References")
    bullets(slide, Inches(0.6), Inches(1.6), Inches(12.1), Inches(4.3), [
        "Google. ClusterData2019 documentation. google/cluster-data, GitHub. "
        "https://github.com/google/cluster-data/blob/master/ClusterData2019.md",
        "Google. PowerData2019 documentation. google/cluster-data, GitHub. "
        "https://github.com/google/cluster-data/blob/master/PowerData2019.md",
        "Google. cluster-data repository (BigQuery public datasets, schema). "
        "https://github.com/google/cluster-data",
        "Kaggle dataset “muzairbair/borg-traces-data” — the Version 1 data source "
        "(superseded in V2 by the official traces above).",
        "Project internal documentation: docs/official_data_alignment_audit.md, "
        "docs/version2_research_design.md, docs/final_pipeline_readiness_audit.md, "
        "docs/final_validation_report.md (V1), results/model_selection_history.csv.",
    ], size=14.5, space_after=14)
    panel = rect(slide, Inches(0.6), Inches(5.7), Inches(12.1), Inches(1.1), "", fill=PANEL)
    textbox(slide, Inches(0.85), Inches(5.85), Inches(11.6), Inches(0.8),
            "A structured literature review (target: ~20 papers on datacenter cooling control, "
            "MARL, and safe RL) is PLANNED for the final research phase and has not yet been "
            "conducted — no papers are cited here that were not actually reviewed.",
            size=12.5, italic=True, color=MUTED, line_spacing=1.2)
    add_notes(slide, (
        "SAY: Every source listed here is one this project actually used or produced -- nothing "
        "is a placeholder citation. "
        "LIKELY QUESTION: 'Where's the related-work / literature review?' ANSWER: honestly, not "
        "done yet at this pilot stage -- it's explicitly planned for the final phase (see the "
        "master project plan), and we chose not to fabricate a reading list to fill this slide."
    ))
    return slide


def s_appendix_divider(prs):
    slide = blank_slide(prs, bg=PRIMARY)
    textbox(slide, Inches(0.9), Inches(3.2), Inches(11), Inches(1.0),
            "Appendix", size=40, bold=True, color=WHITE, font=FONT_HEAD)
    textbox(slide, Inches(0.9), Inches(4.15), Inches(11), Inches(0.6),
            "Architecture, formulation, and configuration detail for follow-up questions",
            size=15, color=RGBColor(0xCA, 0xDC, 0xFC))
    add_notes(slide, "SAY: Hold here unless a specific technical question calls for one of these slides.")
    return slide


def sA_gru_arch(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Appendix A — GRU Architecture", kicker="detail")
    rows = [
        ("Cell type", "GRU (Gated Recurrent Unit), 1 layer"), ("Hidden size", "32"),
        ("Input", "1-D sequence, 48 steps (sum_cpu, per cell, normalized on TRAIN)"),
        ("Output", "6-step forecast (30 minutes ahead)"),
        ("Loss", "MSE, Adam optimizer, lr = 0.001"),
        ("Early stopping", "patience = 5 epochs, on validation loss"),
        ("Training result", "converged (early-stopped before 30 epochs)"),
    ]
    make_table(slide, Inches(0.6), Inches(1.7), Inches(7.0), Inches(4.2),
               ["Setting", "Value"], rows, font_size=13, col_widths=[2.2, 4.8])
    textbox(slide, Inches(8.0), Inches(1.7), Inches(4.7), Inches(4.3),
            "GRU update equations (per step t):\n\n"
            "z_t = σ(W_z x_t + U_z h_{t-1})   (update gate)\n"
            "r_t = σ(W_r x_t + U_r h_{t-1})   (reset gate)\n"
            "h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}))\n"
            "h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t\n\n"
            "x_t: workload at step t. h_t: hidden state. σ: sigmoid. "
            "⊙: elementwise product.",
            size=12.5, font=FONT_BODY, line_spacing=1.35)
    return slide


def sB_power_features(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Appendix B — Power-Model Features", kicker="detail")
    rows = [("sum_cpu", "0.584", "sum of average_usage.cpus, active machines, per cell/bucket"),
            ("cell_code", "0.367", "categorical cell identity (captures per-cell baseline draw)"),
            ("sum_mem", "0.049", "sum of average_usage.memory, active machines, per cell/bucket")]
    make_table(slide, Inches(0.6), Inches(1.7), Inches(12.1), Inches(2.0),
               ["Feature", "Random Forest importance", "Meaning"], rows, font_size=13,
               col_widths=[2.0, 2.6, 7.5])
    path = get_chart("power_residuals", chart_power_residuals)
    slide.shapes.add_picture(str(path), Inches(0.6), Inches(4.0), height=Inches(3.1))
    textbox(slide, Inches(7.2), Inches(4.2), Inches(5.5), Inches(2.7),
            "Target: mean_measured_power_util (fraction of PDU rated capacity, "
            "NOT watts — no PDU capacity-in-kW field exists in the public "
            "PowerData2019 schema).\n\nResiduals show a modest positive bias at "
            "high load (underprediction) — disclosed as a limitation, not "
            "corrected by re-fitting against the same data it would be tested on.",
            size=12, line_spacing=1.3, color=MUTED)
    return slide


def sC_thermal_equations(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Appendix C — Thermal Model Equations", kicker="detail")
    textbox(slide, Inches(0.6), Inches(1.7), Inches(12.1), Inches(1.6),
            "Per zone i, discretized with explicit (forward) Euler at dt = 5 minutes:\n\n"
            "T_i[t+1] = T_i[t] + (dt_hours / C_i) × ( Q_i[t] − a_i[t]·P_cool·η  "
            "+ Σ_j K_ij (T_j[t] − T_i[t]) + K_amb (T_amb − T_i[t]) )",
            size=15.5, font=FONT_BODY, line_spacing=1.4)
    rows = [
        ("Q_i", "heat input to zone i", "FITTED (empirical power model × heat_scale_kw)"),
        ("C_i", "thermal mass", "ASSUMED simulation constant (0.5 kWh/°C)"),
        ("K_ij", "inter-zone coupling", "ASSUMED (0.05 kW/°C, line-graph topology)"),
        ("K_amb", "ambient coupling", "ASSUMED (0.02 kW/°C)"),
        ("a_i, P_cool, η", "cooling action, max cooling power, efficiency", "ASSUMED (a∈[0,1], 0.5 kW, 1.0)"),
        ("T_amb", "ambient temperature", "ASSUMED (22°C)"),
    ]
    make_table(slide, Inches(0.6), Inches(3.5), Inches(12.1), Inches(3.4),
               ["Term", "Meaning", "Status"], rows, font_size=12.5, col_widths=[2.0, 4.0, 6.1])
    return slide


def sD_rl_formulation(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Appendix D — PPO / MAPPO Objective", kicker="detail")
    textbox(slide, Inches(0.6), Inches(1.7), Inches(12.1), Inches(1.3),
            "Clipped surrogate objective (shared by PPO and MAPPO's per-agent update):\n\n"
            "L(θ) = E[ min( r_t(θ)·A_t ,  clip(r_t(θ), 1−ε, 1+ε)·A_t ) ]",
            size=16, font=FONT_BODY, line_spacing=1.4)
    rows = [
        ("r_t(θ)", "π_θ(a_t|s_t) / π_θ_old(a_t|s_t) — importance-sampling ratio"),
        ("A_t", "advantage estimate (Generalized Advantage Estimation, λ = 0.95)"),
        ("ε", "clip range (0.2 here) — bounds how far one update can move the policy"),
        ("PPO critic", "single value function over the joint (21-dim) observation"),
        ("MAPPO critic", "single centralized value function over the concatenated global "
                          "(36-dim, 4 zones × 9-dim local) state — training only"),
        ("MAPPO actor", "parameter-shared, per-zone, 9-dim local observation → 1-dim action"),
    ]
    make_table(slide, Inches(0.6), Inches(3.2), Inches(12.1), Inches(3.7),
               ["Term", "Meaning"], rows, font_size=13, col_widths=[2.1, 10.0])
    return slide


# ================================================================= main ==
def build_deck():
    prs = new_presentation()
    for fn in [
        s01_title, s02_core_problem, s03_control_loop, s04_v1_system, s05_why_v1_insufficient,
        s06_research_objective, s07_clusterdata, s08_powerdata, s09_alignment, s10_data_audit,
        s11_data_availability, s12_architecture, s13_preprocessing, s14_forecasting_problem,
        s15_why_gru, s16_forecasting_baselines, s17_gru_interpretation, s18_power_model,
        s19_why_power_matters, s20_thermal_model, s21_thermal_calibration, s22_thermal_finding,
        s23_control_problem, s24_classical_controllers, s25_ppo, s26_mappo, s27_safety_shield,
        s28_validation, s29_evidence_table, s30_done_not_done, s31_why_not_final, s32_final_plan,
        s33_conclusion, s34_references,
        s_appendix_divider, sA_gru_arch, sB_power_features, sC_thermal_equations, sD_rl_formulation,
    ]:
        fn(prs)
    prs.save(str(OUT_PATH))
    print(f"Saved {len(prs.slides)} slides -> {OUT_PATH}")
    return prs


if __name__ == "__main__":
    build_deck()
