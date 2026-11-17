"""SHAP explanations for the saved (tree-based) model: global importance + one fake and one real example.

Interventional TreeSHAP with a background of real training postings, so every explanation starts from
the model's average output on real-world data (not on the SMOTE-balanced data the trees were fit on).
Run: python -m src.explain
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from src.evaluate import MODEL_PATH, MUTED, save   # importing evaluate also applies the shared chart style
from src.preprocess import CAT_FIELDS, SEED, build_features, load_data, split

REAL, FAKE, NEUTRAL = "#2a78d6", "#eb6834", "#c3c2b7"   # orange = pushes toward fake, blue = toward real
N_BACKGROUND = 200
TOP_N = 20
GROUPS = {"text": "Words (TF-IDF)", "cat": "Categories", "num": "Lengths + missing count", "bin": "Yes/no flags"}


def readable(name: str) -> str:
    """'text__data entry' -> 'word: "data entry"', 'cat__industry_Oil & Energy' -> 'industry = Oil & Energy'."""
    block, feat = name.split("__", 1)
    if block == "text":
        return f'word: "{feat}"'
    if block == "cat":
        field = next(f for f in CAT_FIELDS if feat.startswith(f + "_"))
        value = feat[len(field) + 1:]
        return f"{field} = {'(rare category)' if value == 'infrequent_sklearn' else value}"
    return feat


def with_value(name: str, x: float, raw: pd.Series) -> str:
    """Feature label plus this posting's value, for the example charts."""
    block, feat = name.split("__", 1)
    if block == "text":
        return f'"{feat}" {"in text" if x > 0 else "not in text"}'
    if block == "cat":
        return f"{readable(name)}: {'yes' if x > 0 else 'no'}"
    return f"{feat} = {raw[feat]:g}"                     # raw (unscaled) value from build_features


def group_shap(sv, blocks) -> pd.DataFrame:
    """SHAP values are additive, so summing within a feature group gives that group's total push."""
    return pd.DataFrame({label: sv[:, blocks == g].sum(1) for g, label in GROUPS.items()})


def plot_importance(sv, Xt, labels, blocks, name):
    mean_abs = np.abs(sv).mean(0)
    top = np.argsort(-mean_abs)[:TOP_N][::-1]
    sign = [np.sign(np.corrcoef(Xt[:, i], sv[:, i])[0, 1]) if Xt[:, i].std() and sv[:, i].std() else 0 for i in top]
    groups = group_shap(sv, blocks).abs().mean().sort_values()
    counts = {label: int((blocks == g).sum()) for g, label in GROUPS.items()}

    fig, (ax_g, ax) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.75})
    ax_g.barh([f"{g}  ({counts[g]:,})" for g in groups.index], groups.values, height=0.5, color=MUTED)
    ax_g.grid(axis="y", visible=False)
    ax_g.set(xlabel="Mean |sum of SHAP values in group|", title="Feature groups (total push)")
    ax.barh([labels[i] for i in top], mean_abs[top], height=0.6,
            color=[FAKE if s > 0 else REAL if s < 0 else NEUTRAL for s in sign])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean |SHAP value|  (average change in fraud score)", title=f"Top {TOP_N} single features")
    ax.legend(handles=[Patch(color=FAKE, label="higher / present → more likely fake"),
                       Patch(color=REAL, label="higher / present → more likely real")], loc="lower right")
    save(fig, name)
    return (groups[::-1].rename("mean_abs_group_shap"),
            pd.DataFrame({"feature": [labels[i] for i in top[::-1]], "mean_abs_shap": mean_abs[top[::-1]],
                          "direction": ["→ fake" if s > 0 else "→ real" if s < 0 else "-" for s in sign[::-1]]}))


def plot_beeswarm(sv, Xt, base, labels, name):
    top = np.argsort(-np.abs(sv).mean(0))[:TOP_N]                # only the top features: no "sum of others" row
    expl = shap.Explanation(values=sv[:, top], base_values=np.full(len(sv), base), data=Xt[:, top],
                            feature_names=[labels[i] for i in top])
    shap.plots.beeswarm(expl, max_display=TOP_N, show=False, plot_size=(9, 8),
                        color=LinearSegmentedColormap.from_list("bgo", [REAL, NEUTRAL, FAKE]))
    fig = plt.gcf()
    fig.axes[0].set_title("How each feature value moves the fraud score", loc="left", weight="bold")
    save(fig, name)


def plot_example(sv_row, x_row, raw, base, proba, threshold, names, blocks, title, name, k=12):
    """Top-k single contributions for one posting; group totals in the subtitle."""
    order = np.argsort(-np.abs(sv_row))[:k]
    vals, labs = sv_row[order], [with_value(names[i], x_row[i], raw) for i in order]
    words, structured = sv_row[blocks == "text"].sum(), sv_row[blocks != "text"].sum()
    y = np.arange(k)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * k + 2))
    ax.barh(y, vals, height=0.6, color=[FAKE if v > 0 else REAL for v in vals])
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    verdict = "FLAGGED as fake" if proba >= threshold else "passed as real"
    ax.set(yticks=y, yticklabels=labs, xlabel="SHAP value  (← toward real | toward fake →)")
    ax.set_title(f"{title}\nbaseline {base:.3f} → score {proba:.3f} → {verdict} (threshold {threshold:.3f})", pad=30)
    ax.text(0, 1.02, f"Top {k} of {len(sv_row):,} features shown.  Combined push: all words {words:+.3f}, "
            f"all structured features {structured:+.3f}", transform=ax.transAxes, color=MUTED, fontsize=9)
    save(fig, name)
    return pd.DataFrame({"feature": labs, "shap": np.round(vals, 4)}), words, structured


def main():
    bundle = joblib.load(MODEL_PATH)
    pipe, t = bundle["pipeline"], bundle["threshold"]
    feats, cols, model = pipe.named_steps["features"], pipe.named_steps["columns"], pipe.named_steps["model"]
    prep = lambda X: cols.transform(feats.transform(X)).toarray()
    names = cols.get_feature_names_out()
    labels = [readable(n) for n in names]
    blocks = np.array([n.split("__")[0] for n in names])

    X_train, X_test, y_train, y_test = split(load_data())
    proba = pipe.predict_proba(X_test)[:, 1]
    print(f"Model: {bundle['model']} ({bundle['imbalance']}) | mean score on test {proba.mean():.3f} "
          f"vs actual fake rate {y_test.mean():.3f}  (SMOTE-trained scores are inflated, not calibrated)")

    # Examples: the most typical correct call of each kind (score closest to the median of its group)
    tp, tn = X_test.index[(y_test == 1) & (proba >= t)], X_test.index[(y_test == 0) & (proba < t)]
    p = pd.Series(proba, index=X_test.index)
    fake_id = (p[tp] - p[tp].median()).abs().idxmin()
    real_id = (p[tn] - p[tn].median()).abs().idxmin()

    # Explanation sample: every test fake + the same number of random test reals, plus the two examples
    reals = X_test.index[y_test == 0].to_series().sample((y_test == 1).sum(), random_state=SEED)
    sample = X_test.index[y_test == 1].union(reals).union([fake_id, real_id])
    print(f"Explaining {len(sample)} test postings ({(y_test[sample] == 1).sum()} fake) "
          f"against {N_BACKGROUND} real training postings...")
    explainer = shap.TreeExplainer(model, data=prep(X_train.sample(N_BACKGROUND, random_state=SEED)),
                                   feature_perturbation="interventional")
    Xt = prep(X_test.loc[sample])
    sv = np.asarray(explainer.shap_values(Xt))           # additivity is checked by shap
    sv = sv[:, :, 1] if sv.ndim == 3 else sv             # sklearn classifiers return both classes
    base = float(np.ravel(explainer.expected_value)[-1])
    print(f"Baseline (mean score over background): {base:.3f}")

    groups, imp = plot_importance(sv, Xt, labels, blocks, "09_shap_importance.png")
    print("\nFeature groups (mean |sum of SHAP in group|):\n" + groups.round(4).to_string())
    print(f"\nGlobal importance (top {TOP_N}):\n" + imp.round(4).to_string(index=False))
    plot_beeswarm(sv, Xt, base, labels, "10_shap_beeswarm.png")

    pos = {idx: i for i, idx in enumerate(sample)}
    for idx, kind, fname in [(fake_id, "fake", "11_shap_example_fake.png"), (real_id, "real", "12_shap_example_real.png")]:
        raw = build_features(X_test.loc[[idx]]).iloc[0]
        row = X_test.loc[idx]
        print(f"\n--- Example {kind}: '{row['title']}' | logo {row['has_company_logo']} | "
              f"industry {row['industry']} | score {p[idx]:.3f}")
        tbl, words, structured = plot_example(sv[pos[idx]], Xt[pos[idx]], raw, base, p[idx], t, names, blocks,
                                              f"Actually {kind}: \"{row['title'].strip()}\"", fname)
        print(tbl.to_string(index=False))
        print(f"Combined: all words {words:+.4f} | all structured {structured:+.4f} | "
              f"baseline {base:.4f} + total {words + structured:+.4f} = score {p[idx]:.4f}")


if __name__ == "__main__":
    main()
