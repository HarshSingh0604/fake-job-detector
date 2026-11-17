"""Metrics, threshold selection and plots. `python -m src.evaluate` re-scores the saved model on the test set."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from src.preprocess import ROOT, load_data, split

MODEL_PATH = ROOT / "models" / "fraud_detector.joblib"
FIG_DIR = ROOT / "reports" / "figures"

# Same chart style as the EDA notebook
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10, "text.color": INK,
    "axes.labelcolor": MUTED, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
})


def classification_metrics(y_true, proba, threshold=0.5) -> dict:
    """All metrics for the fraud class (label 1) at a given decision threshold."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "pr_auc": average_precision_score(y_true, proba),   # threshold-free
        "roc_auc": roc_auc_score(y_true, proba),             # threshold-free
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pick_threshold(y_true, proba, min_precision: float) -> float:
    """Highest-recall threshold that keeps precision >= min_precision (ties -> higher precision)."""
    p, r, t = precision_recall_curve(y_true, proba)
    p, r = p[:-1], r[:-1]                                    # last point has no threshold
    ok = p >= min_precision
    if not ok.any():                                         # floor unreachable -> best F1 instead
        return float(t[np.argmax(2 * p * r / (p + r + 1e-12))])
    best = ok & (r == r[ok].max())
    return float(t[np.flatnonzero(best)[np.argmax(p[best])]])


def best_f1_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    return float(t[np.argmax(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12))])


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print("saved", (FIG_DIR / name).relative_to(ROOT))


def plot_pr_curves(curves: dict, prevalence: float, name: str, point: dict | None = None):
    """Precision-recall curves; `curves` maps label -> (y_true, proba). `point` marks the chosen threshold."""
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for (label, (y, p)), color in zip(curves.items(), SERIES):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, drawstyle="steps-post", color=color, lw=2,
                label=f"{label}  (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(prevalence, color=MUTED, lw=1)
    ax.text(0.01, prevalence + 0.015, f"random guessing = {prevalence:.1%}", color=MUTED, fontsize=9)
    if point:
        ax.scatter(point["recall"], point["precision"], s=80, color=INK, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(point["label"], (point["recall"], point["precision"]), xytext=(-12, -22),
                    textcoords="offset points", ha="right", fontsize=9)
    ax.set(xlim=(0, 1.01), ylim=(0, 1.03), xlabel="Recall (share of fakes caught)",
           ylabel="Precision (share of flags that are fake)", title="Precision–recall on the test set")
    ax.legend(loc="center left")
    save(fig, name)


def plot_threshold_tradeoff(y_true, proba, chosen: float, min_precision: float, name: str):
    """Precision / recall / F1 as the threshold moves (out-of-fold training predictions)."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for vals, label, color in [(p, "Precision", SERIES[0]), (r, "Recall", SERIES[1]), (f1, "F1", SERIES[2])]:
        ax.plot(t, vals[:-1], color=color, lw=2, label=label)
    ax.axhline(min_precision, color=MUTED, lw=1)
    ax.text(0.995, min_precision + 0.01, f"precision floor {min_precision:.0%}", color=MUTED, fontsize=9, ha="right", va="bottom")
    for x, label, color in [(0.5, "default 0.5", MUTED), (chosen, f"chosen {chosen:.3f}", INK)]:
        ax.axvline(x, color=color, lw=1.2)
        ax.text(x, 1.04, label, color=color, fontsize=9, ha="center", va="bottom")
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Decision threshold (flag as fake if probability ≥ threshold)",
           ylabel="Score")
    ax.set_title("Choosing the threshold (out-of-fold predictions on training data)", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0.6, 0.02))
    save(fig, name)


def plot_confusion(y_true, proba, threshold: float, name: str):
    cm = confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1])
    share = cm / cm.sum(axis=1, keepdims=True)                     # colour = share of each true class
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(share, cmap=LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#184f95"]), vmin=0, vmax=1)
    for i, row in enumerate(["real", "fake"]):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.1%} of {row}", ha="center", va="center", fontsize=11,
                    color="white" if share[i, j] > 0.5 else INK, weight="bold")
    ax.set(xticks=[0, 1], xticklabels=["Predicted real", "Predicted fake"], yticks=[0, 1],
           yticklabels=["Actually real", "Actually fake"], title=f"Test set confusion matrix (threshold {threshold:.3f})")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, name)


def main():
    bundle = joblib.load(MODEL_PATH)
    _, X_test, _, y_test = split(load_data())                      # same seed -> same test set as training
    proba = bundle["pipeline"].predict_proba(X_test)[:, 1]
    t = bundle["threshold"]
    print(f"\nModel: {bundle['model']} ({bundle['imbalance']}) | threshold {t:.3f} | test rows {len(y_test):,}")

    results = {"threshold_0.5": classification_metrics(y_test, proba, 0.5),
               "threshold_tuned": classification_metrics(y_test, proba, t)}
    for k, m in results.items():
        print(f"{k:16s} " + "  ".join(f"{n} {v:.3f}" if isinstance(v, float) else f"{n} {v}" for n, v in m.items()))

    plot_confusion(y_test, proba, t, "08_confusion_matrix.png")
    out = ROOT / "reports" / "final_metrics.json"
    out.write_text(json.dumps({"model": bundle["model"], "imbalance": bundle["imbalance"], "threshold": t,
                               "test_rows": int(len(y_test)), **results}, indent=2))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
