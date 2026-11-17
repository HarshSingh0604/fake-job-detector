"""Train, compare and save the best fake-job-posting model.

3 models x 3 imbalance strategies, each tuned with stratified 5-fold CV on PR-AUC.
The best model is chosen on CV score (never test), the decision threshold on out-of-fold
training predictions, and the test set is only used to report results.
Run: python -m src.train
"""
import shutil
import tempfile
import time
from datetime import date

import joblib
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Memory
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluate import (MODEL_PATH, best_f1_threshold, classification_metrics, pick_threshold,
                          plot_pr_curves, plot_threshold_tradeoff)
from src.preprocess import ROOT, SEED, build_preprocessor, load_data, split

MIN_PRECISION = 0.80   # business rule: at most 1 in 5 flagged postings may be a false alarm
N_JOBS = 5             # 5 CV folds in parallel x 2 threads per model = 10 cores
MODELS = ["LogReg", "RandomForest", "XGBoost"]
STRATEGIES = ["none", "weights", "smote"]


def make_model(name: str, strategy: str, spw: float):
    """Estimator + small hyperparameter grid. `weights` = class_weight / scale_pos_weight."""
    w = strategy == "weights"
    if name == "LogReg":
        return (LogisticRegression(max_iter=5000, class_weight="balanced" if w else None),
                {"model__C": [0.3, 1, 3, 10, 30, 100]})
    if name == "RandomForest":
        return (RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample" if w else None,
                                       n_jobs=2, random_state=SEED),
                {"model__min_samples_leaf": [1, 3, 5], "model__max_features": ["sqrt", "log2"]})
    return (XGBClassifier(n_estimators=400, learning_rate=0.1, subsample=0.8, colsample_bytree=0.3, max_bin=64,
                          scale_pos_weight=spw if w else 1.0, n_jobs=2, random_state=SEED),
            {"model__max_depth": [4, 6], "model__min_child_weight": [1, 5]})


def make_pipeline(model, strategy: str, memory):
    """features -> columns -> (SMOTE) -> model, as one flat pipeline (imblearn rejects nested Pipelines).
    imblearn's Pipeline only resamples during fit, never at predict time."""
    steps = list(build_preprocessor().steps)
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", model))
    return (ImbPipeline if strategy == "smote" else Pipeline)(steps, memory=memory)


def to_markdown(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def main():
    start = time.time()
    X_train, X_test, y_train, y_test = split(load_data())
    spw = (y_train == 0).sum() / (y_train == 1).sum()          # ~19.5 real per fake
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cache_dir = tempfile.mkdtemp(prefix="fjd-cache-")
    memory = Memory(cache_dir, verbose=0)                       # preprocessing is fit once per fold, then reused
    print(f"Train {len(X_train):,} | Test {len(X_test):,} | scale_pos_weight {spw:.2f}\n")

    rows, fitted = [], {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    rows.append({"model": "Dummy (always real)", "imbalance": "-", "cv_pr_auc": y_train.mean(), "cv_std": 0.0,
                 **classification_metrics(y_test, dummy.predict_proba(X_test)[:, 1]), "best_params": "-"})

    try:
        for name in MODELS:
            for strategy in STRATEGIES:
                t0 = time.time()
                model, grid = make_model(name, strategy, spw)
                search = GridSearchCV(make_pipeline(model, strategy, memory), grid, scoring="average_precision",
                                      cv=cv, n_jobs=N_JOBS, error_score="raise").fit(X_train, y_train)
                i = search.best_index_
                proba = search.predict_proba(X_test)[:, 1]
                params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
                rows.append({"model": name, "imbalance": strategy,
                             "cv_pr_auc": search.cv_results_["mean_test_score"][i],
                             "cv_std": search.cv_results_["std_test_score"][i],
                             **classification_metrics(y_test, proba), "best_params": params})
                fitted[(name, strategy)] = (search.best_estimator_, proba)
                print(f"{name:12s} {strategy:7s} CV PR-AUC {rows[-1]['cv_pr_auc']:.4f} ± {rows[-1]['cv_std']:.4f}"
                      f"  {params}  ({time.time() - t0:.0f}s)", flush=True)

        # ---- comparison table (all test numbers at the default 0.5 threshold) ----
        table = pd.DataFrame(rows)
        table.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
        show = table.assign(**{
            "CV PR-AUC": [f"{m:.3f} ± {s:.3f}" for m, s in zip(table.cv_pr_auc, table.cv_std)],
            **{c: table[c].map("{:.3f}".format) for c in ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]},
        })[["model", "imbalance", "CV PR-AUC", "pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]]
        show.columns = ["Model", "Imbalance", "CV PR-AUC (mean ± std)", "Test PR-AUC", "Test ROC-AUC",
                        "Precision@0.5", "Recall@0.5", "F1@0.5", "Accuracy@0.5"]
        (ROOT / "reports" / "model_comparison.md").write_text(to_markdown(show) + "\n")
        print("\n" + show.to_string(index=False))

        # ---- pick the winner on CV score only ----
        best_row = table.iloc[1:].sort_values("cv_pr_auc", ascending=False).iloc[0]
        key = (best_row.model, best_row.imbalance)
        best, test_proba = fitted[key]
        print(f"\nBest by CV PR-AUC: {key[0]} ({key[1]})")

        # ---- threshold from out-of-fold predictions on TRAIN (test set untouched) ----
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]
        threshold = pick_threshold(y_train, oof, MIN_PRECISION)
        f1_t = best_f1_threshold(y_train, oof)
        for label, t in [("default", 0.5), (f"best F1", f1_t), (f"recall-max @ precision ≥ {MIN_PRECISION:.0%}", threshold)]:
            m = classification_metrics(y_train, oof, t)
            print(f"  OOF {label:32s} t={t:.3f}  precision {m['precision']:.3f}  recall {m['recall']:.3f}  f1 {m['f1']:.3f}")
        plot_threshold_tradeoff(y_train, oof, threshold, MIN_PRECISION, "07_threshold_tradeoff.png")

        # ---- test-set PR curves: best strategy per model family, chosen threshold marked ----
        curves = {}
        for name in MODELS:
            fam = table[table.model == name].sort_values("cv_pr_auc", ascending=False).iloc[0]
            curves[f"{name} ({fam.imbalance})"] = (y_test, fitted[(name, fam.imbalance)][1])
        m = classification_metrics(y_test, test_proba, threshold)
        plot_pr_curves(curves, y_test.mean(), "06_pr_curves.png",
                       point={"recall": m["recall"], "precision": m["precision"], "label": f"chosen threshold {threshold:.3f}"})

        # ---- save the full pipeline (raw fields in -> probability out) + threshold ----
        best.set_params(memory=None)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"pipeline": best, "threshold": threshold, "model": key[0], "imbalance": key[1],
                     "best_params": best_row.best_params, "cv_pr_auc": float(best_row.cv_pr_auc),
                     "min_precision": MIN_PRECISION, "sklearn_version": sklearn.__version__,
                     "trained_on": str(date.today())}, MODEL_PATH, compress=3)
        print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB) "
              f"| total time {(time.time() - start) / 60:.1f} min")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
