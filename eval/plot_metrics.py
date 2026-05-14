"""Render the model-evaluation PNGs that are embedded in README.md.

Re-runs 5-fold GroupKFold CV (so the figures stay in sync with the shipping
predictor) and writes:
    docs/confusion.png    — heuristic vs classifier confusion matrices side-by-side
    docs/roc_pr.png       — ROC and Precision-Recall curves for the classifier
    docs/calibration.png  — reliability diagram + predicted-proba histogram

Run: python -m eval.plot_metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless — no GUI required
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold  # noqa: E402

from app.heuristic import predict_pair  # noqa: E402
from eval.train_classifier import build_matrix, load_pairs, train_one  # noqa: E402

DOCS = ROOT / "docs"
PUBLIC_JSON = ROOT / "relevant_priors_public.json"


def oof_classifier_proba(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> np.ndarray:
    proba = np.zeros(len(y), dtype=np.float32)
    gkf = GroupKFold(n_splits=folds)
    for tr_idx, te_idx in gkf.split(X, y, groups):
        model = train_one(X[tr_idx], y[tr_idx])
        proba[te_idx] = model.predict(X[te_idx])
    return proba


def heuristic_predictions(rows) -> np.ndarray:
    return np.array(
        [predict_pair(r["curr_desc"], r["prior_desc"]).predicted for r in rows],
        dtype=bool,
    )


def plot_confusion_pair(y_true: np.ndarray, y_heur: np.ndarray, y_clf: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (title, pred) in zip(axes, [("Heuristic", y_heur), ("LightGBM (5-fold OOF)", y_clf)], strict=True):
        cm = np.array(
            [
                [int(((y_true == 0) & (pred == 0)).sum()), int(((y_true == 0) & (pred == 1)).sum())],
                [int(((y_true == 1) & (pred == 0)).sum()), int(((y_true == 1) & (pred == 1)).sum())],
            ]
        )
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([0, 1], ["Pred False", "Pred True"])
        ax.set_yticks([0, 1], ["True False", "True True"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(
                    j, i,
                    f"{cell_labels[i][j]}\n{cm[i, j]:,}",
                    ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Confusion matrices on the public split (N = {len(y_true):,} pairs, 23.8 % positive)",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE: |E[y - p]| weighted by bin mass."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (proba >= bins[i]) & (proba < bins[i + 1] if i < n_bins - 1 else proba <= bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_p = proba[mask].mean()
        avg_y = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_y - avg_p)
    return float(ece)


def plot_calibration(y_true: np.ndarray, proba: np.ndarray, out: Path) -> None:
    brier = brier_score_loss(y_true, proba)
    ece = expected_calibration_error(y_true, proba, n_bins=15)
    # quantile-binned so each bin has a comparable count, more honest near edges
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=15, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].plot([0, 1], [0, 1], lw=1, ls="--", color="gray", label="Perfectly calibrated")
    axes[0].plot(mean_pred, frac_pos, "o-", lw=2, label="LightGBM (quantile bins)")
    axes[0].set_xlabel("Mean predicted probability (in bin)")
    axes[0].set_ylabel("Fraction of positives (in bin)")
    axes[0].set_title(f"Reliability diagram   Brier = {brier:.4f}   ECE = {ece:.4f}")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].hist(proba[y_true == 0], bins=50, alpha=0.6, label="negatives", color="C0")
    axes[1].hist(proba[y_true == 1], bins=50, alpha=0.6, label="positives", color="C1")
    axes[1].axvline(0.5, lw=1, ls="--", color="gray", label="Shipped threshold = 0.5")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Predicted-probability distribution by class")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper center")

    fig.suptitle(f"Classifier calibration (5-fold OOF, N = {len(y_true):,} pairs)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, proba: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)

    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    pos_rate = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].plot(fpr, tpr, lw=2, label=f"LightGBM  (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], lw=1, ls="--", color="gray", label="Random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (5-fold OOF)")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.005)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, lw=2, label=f"LightGBM  (AP = {pr_auc:.4f})")
    axes[1].axhline(pos_rate, lw=1, ls="--", color="gray", label=f"Positive rate = {pos_rate:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall curve (5-fold OOF)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.005)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower left")

    fig.suptitle(f"Classifier ranking quality on the public split (N = {len(y_true):,} pairs)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Loading {PUBLIC_JSON} ...")
    rows = load_pairs(PUBLIC_JSON)
    print(f"  rows: {len(rows)}")

    print("Building feature matrix ...")
    X, y, _ = build_matrix(rows)
    groups = np.array([f"{r['curr_desc']}||{r['prior_desc']}" for r in rows])

    print("Running 5-fold GroupKFold CV (~6s) ...")
    proba = oof_classifier_proba(X, y, groups, folds=5)
    pred_clf = (proba >= 0.5).astype(int)

    print("Running heuristic baseline ...")
    pred_heur = heuristic_predictions(rows).astype(int)

    print("Rendering plots ...")
    plot_confusion_pair(y, pred_heur, pred_clf, DOCS / "confusion.png")
    plot_roc_pr(y, proba, DOCS / "roc_pr.png")
    plot_calibration(y, proba, DOCS / "calibration.png")

    brier = brier_score_loss(y, proba)
    ece = expected_calibration_error(y, proba, n_bins=15)
    print(f"  Brier score: {brier:.4f}  (lower is better; 0 = perfect)")
    print(f"  ECE (15 bins): {ece:.4f}")
    print(f"  wrote {DOCS / 'confusion.png'}")
    print(f"  wrote {DOCS / 'roc_pr.png'}")
    print(f"  wrote {DOCS / 'calibration.png'}")


if __name__ == "__main__":
    main()
