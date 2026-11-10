import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = "DejaVu Sans"
FS = {"base": 11, "title": 13, "label": 13, "banner": 13,
      "heat_small": 8.5, "heat_big": 10, "heat_thr": 0.05, "heat_flip": 0.5,
      "leg_robust": 10, "leg_calib": 9, "leg_roc_small": 7, "leg_roc_big": 10}
C = {"graded": "#1f4e9c", "hard": "#e07b39", "posthoc": "#2ca02c", "image": "#7f7f7f",
     "ours": "navy", "guide": "gray", "banner": "#c00000", "cmap": "Blues"}
LW = {"robust": 2, "calib": 2, "calib_other": 1.4, "risk": 2.6, "risk_other": 1.5,
      "roc": 1.1, "roc_macro": 3.2, "guide_h": 1.8, "heat_line": 0.8}
MS = {"robust": 6, "calib": 5}
ALPHA = {"grid": 0.35, "band": 0.15, "roc": 0.55}
ROT, HA, RM = 45, "right", "anchor"
VMAX, VMIN = 1, 0
BANNER_Y = 0.995
LEG = {"robust": dict(fontsize=FS["leg_robust"], loc="lower left"),
       "calib": dict(fontsize=FS["leg_calib"], loc="upper left"),
       "roc_small": dict(loc="lower right", fontsize=FS["leg_roc_small"]),
       "roc_big": dict(loc="lower right", fontsize=FS["leg_roc_big"])}

def apply_style():
    plt.rcParams.update({"font.family": FONT, "font.size": FS["base"],
                         "axes.titlesize": FS["title"], "axes.labelsize": FS["label"]})

def palette(n=None):
    cm = matplotlib.colormaps
    p = [cm["tab20"](v) for v in np.linspace(0, 1, 20)] + [cm["Set2"](v) for v in np.linspace(0, 1, 8)]
    return p if n is None else [p[i % len(p)] for i in range(n)]

def banner(fig, text, y=BANNER_Y):
    fig.text(0.5, y, text, ha="center", va="top", color=C["banner"],
             fontsize=FS["banner"], fontweight="bold")

def panel(ax, letter, text):
    ax.set_title(f"({letter}) {text}", fontweight="bold")

def grid(ax):
    ax.grid(alpha=ALPHA["grid"])

def diag(ax):
    ax.plot([0, 1], [0, 1], ":", color=C["guide"])

def hline(ax, y, label=None):
    ax.axhline(y, color=C["guide"], ls=":", lw=LW["guide_h"], label=label)

def vline(ax, x):
    ax.axvline(x, color=C["guide"], ls=":")

def heat(ax, cm, labels, title, counts=False, thr=FS["heat_thr"], fs=FS["heat_small"], ylab=True):
    rn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(rn, cmap=C["cmap"], vmin=VMIN, vmax=VMAX, aspect="auto")
    k = len(labels)
    for i in range(k):
        for j in range(k):
            if rn[i, j] >= thr or i == j or (counts and cm[i, j] > 0):
                txt = f"{cm[i, j]:.0f}" if counts else f"{rn[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                        color="white" if rn[i, j] > FS["heat_flip"] else "black")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=ROT, ha=HA, rotation_mode=RM)
    ax.set_yticklabels(labels if ylab else [])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label" if ylab else "")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", lw=LW["heat_line"])
    ax.tick_params(which="minor", length=0)
    return im

def cbar(fig, im, ax, fraction=0.04, pad=0.02):
    cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cb.set_label("Proportion of true class (row-normalized)")
    return cb

def robust(ax, x, y, lo=None, hi=None, variant="graded", label=None):
    col, mk, ls = (C["graded"], "o", "-") if variant == "graded" else (C["hard"], "s", "--")
    ax.plot(x, y, marker=mk, ls=ls, color=col, lw=LW["robust"], ms=MS["robust"], label=label)
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=col, alpha=ALPHA["band"], lw=0)

def calib(ax, x, y, color, label=None, ours=False):
    ax.plot(x, y, "o-", ms=MS["calib"], color=color,
            lw=LW["calib"] if ours else LW["calib_other"], label=label)

def risk(ax, x, y, color, label=None, ours=False):
    ax.plot(x, y, "--" if ours else "-", color=color,
            lw=LW["risk"] if ours else LW["risk_other"], label=label)

def roc(ax, fpr, tpr, color, label=None):
    ax.plot(fpr, tpr, lw=LW["roc"], alpha=ALPHA["roc"], color=color, label=label)

def roc_macro(ax, x, y, label=None):
    ax.plot(x, y, "--", lw=LW["roc_macro"], color=C["ours"], label=label)

FIG = {"heat_tall": dict(size=(12, 21), rect=(0, 0, 1, 0.985), cbar=dict(fraction=0.04)),
       "heat_wide": dict(size=(16, 7.6), rect=None, tight=True, cbar=dict(fraction=0.025)),
       "robust": dict(size=(4.2, 4.5), rect=(0, 0, 1, 0.88), leg="robust"),
       "calib": dict(size=(13, 11.5), rect=(0, 0, 1, 0.97), leg="calib"),
       "roc": dict(size=(17, 8.2), rect=(0, 0, 1, 0.95), leg="roc_small")}

def size(spec, panels=None):
    w, h = spec["size"]
    return (w * panels, h) if panels else (w, h)

def save(fig, path, rect=None, tight=False, banner_text=None):
    if banner_text:
        banner(fig, banner_text)
    if rect:
        fig.tight_layout(rect=rect)
    fig.savefig(path, bbox_inches="tight" if tight else None)
    plt.close(fig)
    return path
