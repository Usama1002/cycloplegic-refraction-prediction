"""
ABLATION EXPERIMENTS for Cycloplegic Refraction Prediction
==========================================================
5 ablation studies:
  1. Device leave-one-out on S4
  2. S5 post-dilation feature importance + incremental post-dilation
  3. Engineered SE feature ablation
  4. Learning curve (performance vs training set size)
  5. Error distribution analysis (per-eye, tail risk)
"""

import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance as sklearn_perm_imp
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_and_clean_data, compute_metrics, FEATURE_GROUPS

# ============================================================
# CONSTANTS
# ============================================================
N_FOLDS = 5
ALL_SEEDS = [42, 123, 456]
TARGET_COL = "target_SE"
CLIP_MIN, CLIP_MAX = -15.0, 10.0
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
PAPER_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# ============================================================
# MODEL CONSTRUCTORS
# ============================================================
def make_ridge(seed=42):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

def make_rf(seed=42):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=seed, n_jobs=-1)),
    ])

def make_xgboost(seed=42):
    return xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=0)

def make_lgbm(seed=42):
    return lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1)

def make_tabpfn(seed=42):
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor.create_default_for_version("v2", device="auto", n_estimators=8)

CONSTRUCTORS = {
    "Ridge": make_ridge,
    "RandomForest": make_rf,
    "XGBoost": make_xgboost,
    "LightGBM": make_lgbm,
    "TabPFN": make_tabpfn,
}

# ============================================================
# SHARED EVALUATION
# ============================================================
def evaluate(model_name, constructor, X, y, groups, feature_cols, seeds=ALL_SEEDS):
    """Run grouped CV, return aggregated metrics."""
    seed_metrics = []
    for seed in seeds:
        np.random.seed(seed)
        gkf = GroupKFold(n_splits=N_FOLDS)
        preds = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, groups):
            model = constructor(seed)
            if model_name == "TabPFN":
                model.fit(pd.DataFrame(X[tr], columns=feature_cols), y[tr])
                p = model.predict(pd.DataFrame(X[te], columns=feature_cols))
            else:
                model.fit(X[tr], y[tr])
                p = model.predict(X[te])
            if model_name == "Ridge":
                p = np.clip(p, CLIP_MIN, CLIP_MAX)
            preds[te] = p
        valid = ~np.isnan(preds)
        if valid.sum() > 0:
            seed_metrics.append(compute_metrics(y[valid], preds[valid]))
    agg = {}
    for key in seed_metrics[0]:
        vals = [r[key] for r in seed_metrics]
        agg[key] = float(np.nanmean(vals))
        agg[f"{key}_std"] = float(np.nanstd(vals))
    return agg

def evaluate_with_predictions(model_name, constructor, X, y, groups, feature_cols, seed=42):
    """Single-seed CV returning per-eye predictions."""
    np.random.seed(seed)
    gkf = GroupKFold(n_splits=N_FOLDS)
    preds = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        model = constructor(seed)
        if model_name == "TabPFN":
            model.fit(pd.DataFrame(X[tr], columns=feature_cols), y[tr])
            p = model.predict(pd.DataFrame(X[te], columns=feature_cols))
        else:
            model.fit(X[tr], y[tr])
            p = model.predict(X[te])
        if model_name == "Ridge":
            p = np.clip(p, CLIP_MIN, CLIP_MAX)
        preds[te] = p
    return preds

def fmt(val, std=None):
    if std is not None and std > 1e-6:
        return f"{val:.3f}±{std:.3f}"
    return f"{val:.3f}"

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    df = load_and_clean_data()
    # Add engineered SE features
    df["eyerobo_pre_SE"] = df["eyerobo_pre_SPH"] + df["eyerobo_pre_CYL"] / 2
    df["eyerobo_post_SE"] = df["eyerobo_post_SPH"] + df["eyerobo_post_CYL"] / 2
    df["auto_pre_SE"] = df["auto_pre_SPH"] + df["auto_pre_CYL"] / 2
    df["auto_post_SE"] = df["auto_post_SPH"] + df["auto_post_CYL"] / 2
    mask = df[TARGET_COL].notna() & df["patient_id"].notna()
    df_use = df.loc[mask].copy()
    return df_use

# Feature scenario definitions (matching full_experiment.py)
def get_s4_features():
    return (FEATURE_GROUPS["eyerobo_pre"] + ["eyerobo_pre_SE"]
            + FEATURE_GROUPS["auto_pre"] + ["auto_pre_SE"]
            + FEATURE_GROUPS["iol_pre"]
            + FEATURE_GROUPS["demographics"])

def get_s5_features():
    return (FEATURE_GROUPS["eyerobo_pre"] + ["eyerobo_pre_SE"]
            + FEATURE_GROUPS["eyerobo_post"] + ["eyerobo_post_SE"]
            + FEATURE_GROUPS["auto_pre"] + ["auto_pre_SE"]
            + FEATURE_GROUPS["auto_post"] + ["auto_post_SE"]
            + FEATURE_GROUPS["iol_pre"] + FEATURE_GROUPS["iol_post"]
            + FEATURE_GROUPS["demographics"])


# ============================================================
# ABLATION 1: Device Leave-One-Out on S4
# ============================================================
def ablation1_device_leaveout(df_use):
    print("\n" + "="*70)
    print("ABLATION 1: Device Leave-One-Out on S4")
    print("="*70)

    s4_full = get_s4_features()

    # Define leave-one-out scenarios
    s4_minus_eyerobo = (FEATURE_GROUPS["auto_pre"] + ["auto_pre_SE"]
                        + FEATURE_GROUPS["iol_pre"]
                        + FEATURE_GROUPS["demographics"])
    s4_minus_autoref = (FEATURE_GROUPS["eyerobo_pre"] + ["eyerobo_pre_SE"]
                        + FEATURE_GROUPS["iol_pre"]
                        + FEATURE_GROUPS["demographics"])
    s4_minus_iol = (FEATURE_GROUPS["eyerobo_pre"] + ["eyerobo_pre_SE"]
                    + FEATURE_GROUPS["auto_pre"] + ["auto_pre_SE"]
                    + FEATURE_GROUPS["demographics"])

    scenarios = {
        "S4 (full)": s4_full,
        "S4 − Eyerobo": s4_minus_eyerobo,
        "S4 − Auto-ref": s4_minus_autoref,
        "S4 − IOL Master": s4_minus_iol,
    }

    y = df_use[TARGET_COL].values.astype(float)
    groups = df_use["patient_id"].astype(str).values
    models = ["TabPFN", "XGBoost"]
    results = {}

    for sname, fcols in scenarios.items():
        X = df_use[fcols].values.astype(float)
        results[sname] = {"n_features": len(fcols)}
        print(f"\n  {sname} ({len(fcols)} features)")
        for mname in models:
            m = evaluate(mname, CONSTRUCTORS[mname], X, y, groups, fcols)
            results[sname][mname] = m
            print(f"    {mname:12s}: MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  ≤0.50D={m['within_0.50D']:.1f}%")

    return results


# ============================================================
# ABLATION 2: S5 Post-Dilation Incremental Analysis
# ============================================================
def ablation2_postdilation_incremental(df_use):
    print("\n" + "="*70)
    print("ABLATION 2: Incremental Post-Dilation Feature Addition")
    print("="*70)

    s4 = get_s4_features()

    # Incremental additions
    s4_plus_auto_post = s4 + FEATURE_GROUPS["auto_post"] + ["auto_post_SE"]
    s4_plus_eyerobo_post = s4 + FEATURE_GROUPS["eyerobo_post"] + ["eyerobo_post_SE"]
    s4_plus_iol_post = s4 + FEATURE_GROUPS["iol_post"]
    s5_full = get_s5_features()

    scenarios = {
        "S4 (pre only)": s4,
        "S4 + Auto-ref post": s4_plus_auto_post,
        "S4 + Eyerobo post": s4_plus_eyerobo_post,
        "S4 + IOL Master post": s4_plus_iol_post,
        "S5 (all pre+post)": s5_full,
    }

    y = df_use[TARGET_COL].values.astype(float)
    groups = df_use["patient_id"].astype(str).values
    models = ["TabPFN", "XGBoost"]
    results = {}

    for sname, fcols in scenarios.items():
        X = df_use[fcols].values.astype(float)
        results[sname] = {"n_features": len(fcols)}
        print(f"\n  {sname} ({len(fcols)} features)")
        for mname in models:
            m = evaluate(mname, CONSTRUCTORS[mname], X, y, groups, fcols)
            results[sname][mname] = m
            print(f"    {mname:12s}: MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  ≤0.50D={m['within_0.50D']:.1f}%")

    # S5 permutation importance (XGBoost surrogate)
    print("\n  Computing S5 permutation importance (XGBoost)...")
    fcols = s5_full
    X = df_use[fcols].values.astype(float)
    np.random.seed(42)
    gkf = GroupKFold(n_splits=N_FOLDS)
    tr_idx, te_idx = next(iter(gkf.split(X, y, groups)))
    model = make_xgboost(42)
    model.fit(X[tr_idx], y[tr_idx])
    pi = sklearn_perm_imp(model, X[te_idx], y[te_idx], n_repeats=10,
                          random_state=42, scoring="neg_mean_absolute_error")
    imp_order = np.argsort(pi.importances_mean)[::-1]
    s5_importance = []
    for i in imp_order[:15]:
        s5_importance.append({
            "feature": fcols[i],
            "importance": float(pi.importances_mean[i])
        })
        print(f"    {fcols[i]:25s}: {pi.importances_mean[i]:.4f}")
    results["s5_permutation_importance"] = s5_importance

    return results


# ============================================================
# ABLATION 3: Engineered SE Feature Ablation
# ============================================================
def ablation3_se_features(df_use):
    print("\n" + "="*70)
    print("ABLATION 3: Engineered SE Feature Ablation")
    print("="*70)

    s4_with = get_s4_features()
    s4_without = [f for f in s4_with if not f.endswith("_SE") or f.startswith("iol")]
    # iol_pre_SE is from the device, not engineered; keep it
    # Remove only eyerobo_pre_SE and auto_pre_SE

    s5_with = get_s5_features()
    s5_without = [f for f in s5_with if not (f in ["eyerobo_pre_SE", "eyerobo_post_SE",
                                                     "auto_pre_SE", "auto_post_SE"])]

    scenarios = {
        "S4 with SE": s4_with,
        "S4 without SE": s4_without,
        "S5 with SE": s5_with,
        "S5 without SE": s5_without,
    }

    y = df_use[TARGET_COL].values.astype(float)
    groups = df_use["patient_id"].astype(str).values
    models = ["TabPFN", "XGBoost", "Ridge"]
    results = {}

    for sname, fcols in scenarios.items():
        X = df_use[fcols].values.astype(float)
        results[sname] = {"n_features": len(fcols)}
        print(f"\n  {sname} ({len(fcols)} features)")
        for mname in models:
            m = evaluate(mname, CONSTRUCTORS[mname], X, y, groups, fcols)
            results[sname][mname] = m
            print(f"    {mname:12s}: MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  ≤0.50D={m['within_0.50D']:.1f}%")

    return results


# ============================================================
# ABLATION 4: Learning Curve
# ============================================================
def ablation4_learning_curve(df_use):
    print("\n" + "="*70)
    print("ABLATION 4: Learning Curve (S4)")
    print("="*70)

    fcols = get_s4_features()
    X = df_use[fcols].values.astype(float)
    y = df_use[TARGET_COL].values.astype(float)
    groups = df_use["patient_id"].astype(str).values
    unique_patients = np.unique(groups)

    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    models = ["TabPFN", "XGBoost", "RandomForest", "Ridge"]
    results = {m: {"fractions": fractions, "mae_mean": [], "mae_std": [], "n_train": []} for m in models}

    for frac in fractions:
        print(f"\n  Fraction: {frac:.0%}")
        for mname in models:
            seed_maes = []
            for seed in ALL_SEEDS:
                np.random.seed(seed)
                gkf = GroupKFold(n_splits=N_FOLDS)
                preds = np.full(len(y), np.nan)

                for tr, te in gkf.split(X, y, groups):
                    # Subsample training patients
                    train_patients = np.unique(groups[tr])
                    n_keep = max(2, int(len(train_patients) * frac))
                    rng = np.random.RandomState(seed)
                    kept_patients = set(rng.choice(train_patients, n_keep, replace=False))
                    sub_mask = np.array([groups[i] in kept_patients for i in tr])
                    tr_sub = tr[sub_mask]

                    if len(tr_sub) < 5:
                        continue

                    model = CONSTRUCTORS[mname](seed)
                    try:
                        if mname == "TabPFN":
                            model.fit(pd.DataFrame(X[tr_sub], columns=fcols), y[tr_sub])
                            p = model.predict(pd.DataFrame(X[te], columns=fcols))
                        else:
                            model.fit(X[tr_sub], y[tr_sub])
                            p = model.predict(X[te])
                        if mname == "Ridge":
                            p = np.clip(p, CLIP_MIN, CLIP_MAX)
                        preds[te] = p
                    except:
                        pass

                valid = ~np.isnan(preds)
                if valid.sum() > 0:
                    seed_maes.append(mean_absolute_error(y[valid], preds[valid]))

            mean_mae = np.mean(seed_maes) if seed_maes else np.nan
            std_mae = np.std(seed_maes) if seed_maes else np.nan
            n_approx = int(len(y) * 0.8 * frac)  # approx train size
            results[mname]["mae_mean"].append(float(mean_mae))
            results[mname]["mae_std"].append(float(std_mae))
            results[mname]["n_train"].append(n_approx)
            print(f"    {mname:12s}: MAE={mean_mae:.3f}±{std_mae:.3f} (~{n_approx} train)")

    # Generate figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"TabPFN": "#1f77b4", "XGBoost": "#ff7f0e", "RandomForest": "#2ca02c", "Ridge": "#d62728"}
    markers = {"TabPFN": "o", "XGBoost": "s", "RandomForest": "^", "Ridge": "D"}

    for mname in models:
        r = results[mname]
        n = r["n_train"]
        m_arr = np.array(r["mae_mean"])
        s_arr = np.array(r["mae_std"])
        ax.plot(n, m_arr, marker=markers[mname], color=colors[mname], label=mname, linewidth=1.5, markersize=5)
        ax.fill_between(n, m_arr - s_arr, m_arr + s_arr, alpha=0.15, color=colors[mname])

    ax.axhline(y=0.50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(n[-1]*0.98, 0.52, "0.50 D threshold", ha="right", va="bottom", fontsize=7, color="gray")
    ax.set_xlabel("Approximate training set size (eyes)", fontsize=10)
    ax.set_ylabel("MAE (D)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0.4)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(os.path.join(d, "fig_learning_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: fig_learning_curve.png")

    return results


# ============================================================
# ABLATION 5: Error Distribution Analysis
# ============================================================
def ablation5_error_distribution(df_use):
    print("\n" + "="*70)
    print("ABLATION 5: Error Distribution Analysis")
    print("="*70)

    s4 = get_s4_features()
    s5 = get_s5_features()
    y = df_use[TARGET_COL].values.astype(float)
    groups = df_use["patient_id"].astype(str).values

    # Collect per-eye predictions
    configs = {
        "TabPFN_S4": (s4, "TabPFN"),
        "TabPFN_S5": (s5, "TabPFN"),
        "XGBoost_S4": (s4, "XGBoost"),
    }
    all_preds = {}
    for label, (fcols, mname) in configs.items():
        X = df_use[fcols].values.astype(float)
        preds = evaluate_with_predictions(mname, CONSTRUCTORS[mname], X, y, groups, fcols)
        all_preds[label] = preds
        valid = ~np.isnan(preds)
        errors = np.abs(y[valid] - preds[valid])
        p90 = np.percentile(errors, 90)
        p95 = np.percentile(errors, 95)
        p99 = np.percentile(errors, 99)
        print(f"  {label}: P90={p90:.3f}D  P95={p95:.3f}D  P99={p99:.3f}D")

    # --- Figure A: Cumulative error distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Empirical CDF
    ax = axes[0]
    styles = {
        "TabPFN_S4": ("TabPFN (S4)", "#1f77b4", "-"),
        "XGBoost_S4": ("XGBoost (S4)", "#ff7f0e", "--"),
        "TabPFN_S5": ("TabPFN (S5)", "#2ca02c", "-"),
    }
    for label, (name, color, ls) in styles.items():
        preds = all_preds[label]
        valid = ~np.isnan(preds)
        errors = np.sort(np.abs(y[valid] - preds[valid]))
        cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
        ax.plot(errors, cdf, label=name, color=color, linestyle=ls, linewidth=1.5)

    for thresh in [0.25, 0.50, 1.00]:
        ax.axvline(x=thresh, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Absolute prediction error (D)", fontsize=10)
    ax.set_ylabel("Cumulative percentage of eyes (%)", fontsize=10)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=9)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    # Panel B: Box plot by refractive subgroup (TabPFN S4)
    ax = axes[1]
    preds_s4 = all_preds["TabPFN_S4"]
    valid = ~np.isnan(preds_s4)
    abs_errors = np.abs(y[valid] - preds_s4[valid])
    se_vals = y[valid]

    bins = {
        "Emmet./\nhyperop.\n(SE≥0)": se_vals >= 0,
        "Mild\nmyopia\n(0 to −3)": (se_vals < 0) & (se_vals >= -3),
        "Moderate\nmyopia\n(−3 to −6)": (se_vals < -3) & (se_vals >= -6),
        "High\nmyopia\n(< −6)": se_vals < -6,
    }

    box_data = []
    labels = []
    ns = []
    for lbl, mask in bins.items():
        box_data.append(abs_errors[mask])
        labels.append(lbl)
        ns.append(int(mask.sum()))

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    colors_box = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, n in enumerate(ns):
        ax.text(i + 1, -0.15, f"n={n}", ha="center", va="top", fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.axhline(y=0.50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Absolute prediction error (D)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    plt.tight_layout()
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(os.path.join(d, "fig_error_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig_error_distribution.png")

    # Compute percentile stats for results
    stats = {}
    for label in configs:
        preds = all_preds[label]
        valid = ~np.isnan(preds)
        errors = np.abs(y[valid] - preds[valid])
        stats[label] = {
            "P50": float(np.percentile(errors, 50)),
            "P90": float(np.percentile(errors, 90)),
            "P95": float(np.percentile(errors, 95)),
            "P99": float(np.percentile(errors, 99)),
            "max": float(np.max(errors)),
        }

    return stats


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    df_use = load_data()
    print(f"Dataset: {len(df_use)} eyes, {df_use['patient_id'].nunique()} patients")

    all_results = {}

    all_results["ablation1_device_leaveout"] = ablation1_device_leaveout(df_use)
    all_results["ablation2_postdilation"] = ablation2_postdilation_incremental(df_use)
    all_results["ablation3_se_features"] = ablation3_se_features(df_use)
    all_results["ablation4_learning_curve"] = ablation4_learning_curve(df_use)
    all_results["ablation5_error_distribution"] = ablation5_error_distribution(df_use)

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL ABLATIONS COMPLETE in {elapsed:.1f}s")
    print(f"Results saved: {out_path}")
    print(f"Figures saved: {FIG_DIR}/ and {PAPER_FIG_DIR}/")
    print(f"{'='*70}")
