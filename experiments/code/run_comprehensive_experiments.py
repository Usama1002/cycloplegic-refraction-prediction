"""
COMPREHENSIVE EXPERIMENTS: Eric's 12-scenario framework
========================================================
12 scenarios x 8 models x 3 seeds x 5-fold GroupKFold
+ Raw device agreement + Bland-Altman + Correlation + Subgroups
+ Learning curve + Error distribution + Permutation importance
+ All figures for the paper
"""

import sys, os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.inspection import permutation_importance as sklearn_perm_imp
from scipy.stats import wilcoxon, pearsonr, spearmanr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================
N_FOLDS = 5
ALL_SEEDS = [42, 123, 456]
TARGET_COL = "target_SE"
CLIP_MIN, CLIP_MAX = -15.0, 10.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PKL = os.path.join(BASE_DIR, "../../data/dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "..")
FIG_DIR = os.path.join(BASE_DIR, "../../paper/figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.labelsize": 14,
    "axes.titlesize": 14, "legend.fontsize": 11, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "figure.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 1.2,
    "lines.linewidth": 2.0, "lines.markersize": 7,
})
C = {"TabPFN": "#0072B2", "XGBoost": "#E69F00", "RF": "#009E73",
     "LightGBM": "#CC79A7", "CatBoost": "#D55E00", "Ridge": "#999999",
     "SVR": "#56B4E9", "MLP": "#F0E442"}

# ============================================================
# 12 SCENARIOS (Eric's framework)
# ============================================================
FEATURE_GROUPS = {
    "auto_pre": ["auto_pre_SPH", "auto_pre_CYL", "auto_pre_AX"],
    "auto_post": ["auto_post_SPH", "auto_post_CYL", "auto_post_AX"],
    "eyerobo_pre": ["eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_pre_AX"],
    "eyerobo_post": ["eyerobo_post_SPH", "eyerobo_post_CYL", "eyerobo_post_AX"],
    "iol_biometry": [
        "iol_pre_AL", "iol_pre_LT", "iol_pre_WTW", "iol_pre_CCT",
        "iol_pre_K1", "iol_pre_K2", "iol_pre_dK", "iol_pre_ACD", "iol_pre_SE",
    ],
    "demographics": ["age", "IOP", "gender_enc", "eye_enc", "treatment_enc"],
}

def _expand(groups):
    feats = []
    for g in groups:
        feats.extend(FEATURE_GROUPS[g])
    return feats

SCENARIOS = {
    "S01_AR_Pre":         _expand(["auto_pre", "demographics"]),
    "S02_AR_Post":        _expand(["auto_post", "demographics"]),
    "S03_AR_PrePost":     _expand(["auto_pre", "auto_post", "demographics"]),
    "S04_ER_Pre":         _expand(["eyerobo_pre", "demographics"]),
    "S05_ER_Post":        _expand(["eyerobo_post", "demographics"]),
    "S06_ER_PrePost":     _expand(["eyerobo_pre", "eyerobo_post", "demographics"]),
    "S07_AR_Pre_IOL":     _expand(["auto_pre", "iol_biometry", "demographics"]),
    "S08_AR_Post_IOL":    _expand(["auto_post", "iol_biometry", "demographics"]),
    "S09_AR_PrePost_IOL": _expand(["auto_pre", "auto_post", "iol_biometry", "demographics"]),
    "S10_ER_Pre_IOL":     _expand(["eyerobo_pre", "iol_biometry", "demographics"]),
    "S11_ER_Post_IOL":    _expand(["eyerobo_post", "iol_biometry", "demographics"]),
    "S12_ER_PrePost_IOL": _expand(["eyerobo_pre", "eyerobo_post", "iol_biometry", "demographics"]),
}

# ============================================================
# 8 MODELS
# ============================================================
def make_ridge(s=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])

def make_rf(s=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=100, max_depth=10,
                            min_samples_leaf=5, random_state=s, n_jobs=-1))])

def make_xgb(s=42):
    return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=s, n_jobs=-1, verbosity=0)

def make_lgbm(s=42):
    return lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=s, n_jobs=-1, verbose=-1)

def make_catboost(s=42):
    return cb.CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=s, verbose=0, allow_writing_files=False)

def make_tabpfn(s=42):
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor.create_default_for_version("v2", device="auto", n_estimators=8)

def make_svr(s=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=10.0, epsilon=0.1))])

def make_mlp(s=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("m", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                            random_state=s, early_stopping=True, validation_fraction=0.15))])

MODELS = {
    "Ridge": make_ridge, "RF": make_rf, "XGBoost": make_xgb,
    "LightGBM": make_lgbm, "CatBoost": make_catboost, "TabPFN": make_tabpfn,
    "SVR": make_svr, "MLP": make_mlp,
}

# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred):
    ae = np.abs(y_true - y_pred)
    diff = y_true - y_pred
    return {
        "MAE": float(np.mean(ae)), "RMSE": float(np.sqrt(np.mean(diff**2))),
        "R2": float(r2_score(y_true, y_pred)), "MedAE": float(np.median(ae)),
        "within_0.25D": float(np.mean(ae <= 0.25) * 100),
        "within_0.50D": float(np.mean(ae <= 0.50) * 100),
        "within_1.00D": float(np.mean(ae <= 1.00) * 100),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "LoA_lower": float(np.mean(diff) - 1.96 * np.std(diff, ddof=1)),
        "LoA_upper": float(np.mean(diff) + 1.96 * np.std(diff, ddof=1)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "spearman_rho": float(spearmanr(y_true, y_pred)[0]),
    }

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model_name, constructor, X, y, groups, fcols, seeds=ALL_SEEDS):
    seed_metrics, all_fold_maes = [], []
    oof_preds = np.full(len(y), np.nan)  # from first seed

    for si, seed in enumerate(seeds):
        np.random.seed(seed)
        gkf = GroupKFold(n_splits=N_FOLDS)
        preds = np.full(len(y), np.nan)
        fold_maes = []

        for tr, te in gkf.split(X, y, groups):
            model = constructor(seed)
            try:
                if model_name == "TabPFN":
                    model.fit(pd.DataFrame(X[tr], columns=fcols), y[tr])
                    p = model.predict(pd.DataFrame(X[te], columns=fcols))
                else:
                    model.fit(X[tr], y[tr])
                    p = model.predict(X[te])
                if model_name == "Ridge":
                    p = np.clip(p, CLIP_MIN, CLIP_MAX)
                preds[te] = p
                fold_maes.append(float(mean_absolute_error(y[te], p)))
            except Exception as e:
                fold_maes.append(np.nan)

        valid = ~np.isnan(preds)
        if valid.sum() > 0:
            seed_metrics.append(compute_metrics(y[valid], preds[valid]))
        all_fold_maes.append(np.array(fold_maes))
        if si == 0:
            oof_preds = preds.copy()

    # Aggregate
    agg = {}
    if seed_metrics:
        for key in seed_metrics[0]:
            vals = [r[key] for r in seed_metrics]
            agg[key] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
    return agg, all_fold_maes, oof_preds


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("COMPREHENSIVE EXPERIMENTS: 12 scenarios x 8 models")
    print("=" * 70)

    # Load cleaned data
    df = pd.read_csv(DATA_PKL)
    y = df[TARGET_COL].values.astype(float)
    groups = df["patient_id"].astype(str).values
    print(f"Dataset: {len(df)} eyes, {df['patient_id'].nunique()} patients")

    all_results = {}
    fold_maes_store = {}
    oof_store = {}

    # ========================================================
    # 1. RUN ALL 12 SCENARIOS x 8 MODELS
    # ========================================================
    for sname, fcols in SCENARIOS.items():
        X = df[fcols].values.astype(float)
        nf = len(fcols)
        print(f"\n{'='*60}")
        print(f"{sname} ({nf} features, {len(y)} eyes)")
        print(f"{'='*60}")
        all_results[sname] = {}
        fold_maes_store[sname] = {}

        for mname, ctor in MODELS.items():
            agg, fmaes, oof_p = evaluate(mname, ctor, X, y, groups, fcols)
            all_results[sname][mname] = agg
            fold_maes_store[sname][mname] = fmaes
            oof_store[f"{mname}_{sname}"] = oof_p
            mae = agg.get("MAE", np.nan)
            r2 = agg.get("R2", np.nan)
            w50 = agg.get("within_0.50D", np.nan)
            print(f"  {mname:10s}: MAE={mae:.3f}  R2={r2:.3f}  <=0.50D={w50:.1f}%")

    # ========================================================
    # 2. WILCOXON SIGNIFICANCE TESTS
    # ========================================================
    print(f"\n{'='*60}\nSIGNIFICANCE TESTS\n{'='*60}")
    sig_tests = {}
    for sname in SCENARIOS:
        sig_tests[sname] = {}
        if "TabPFN" not in fold_maes_store.get(sname, {}):
            continue
        tab_fmaes = np.concatenate(fold_maes_store[sname]["TabPFN"])
        for mname in MODELS:
            if mname == "TabPFN" or mname not in fold_maes_store[sname]:
                continue
            other_fmaes = np.concatenate(fold_maes_store[sname][mname])
            valid = ~(np.isnan(tab_fmaes) | np.isnan(other_fmaes))
            try:
                stat, pv = wilcoxon(tab_fmaes[valid], other_fmaes[valid], alternative="two-sided")
                sig_tests[sname][f"TabPFN_vs_{mname}"] = {
                    "statistic": float(stat), "p_value": float(pv),
                    "n_pairs": int(valid.sum()),
                    "significant_0.05": pv < 0.05,
                }
            except:
                pass

    # ========================================================
    # 3. RAW DEVICE AGREEMENT (before ML)
    # ========================================================
    print(f"\n{'='*60}\nRAW DEVICE AGREEMENT\n{'='*60}")
    raw_agreement = {}
    for device, sph_col, cyl_col, label in [
        ("eyerobo_pre", "eyerobo_pre_SPH", "eyerobo_pre_CYL", "Eyerobo VS Pre"),
        ("auto_pre", "auto_pre_SPH", "auto_pre_CYL", "Auto-refractor Pre"),
        ("eyerobo_post", "eyerobo_post_SPH", "eyerobo_post_CYL", "Eyerobo VS Post"),
        ("auto_post", "auto_post_SPH", "auto_post_CYL", "Auto-refractor Post"),
    ]:
        valid = df[sph_col].notna() & df[cyl_col].notna() & df[TARGET_COL].notna()
        device_se = df.loc[valid, sph_col] + df.loc[valid, cyl_col] / 2
        cyclo_se = df.loc[valid, TARGET_COL]
        m = compute_metrics(cyclo_se.values, device_se.values)
        m["n_eyes"] = int(valid.sum())
        raw_agreement[device] = m
        print(f"  {label:25s}: n={m['n_eyes']}, mean_diff={m['mean_diff']:+.3f}D, "
              f"LoA=[{m['LoA_lower']:+.3f}, {m['LoA_upper']:+.3f}], r={m['pearson_r']:.3f}")

    # ========================================================
    # 4. PERMUTATION IMPORTANCE
    # ========================================================
    print(f"\n{'='*60}\nPERMUTATION IMPORTANCE\n{'='*60}")
    perm_imp = {}
    for sname in ["S01_AR_Pre", "S04_ER_Pre", "S07_AR_Pre_IOL", "S10_ER_Pre_IOL"]:
        fcols = SCENARIOS[sname]
        X = df[fcols].values.astype(float)
        np.random.seed(42)
        gkf = GroupKFold(n_splits=N_FOLDS)
        tr, te = next(iter(gkf.split(X, y, groups)))
        model = make_xgb(42)
        model.fit(X[tr], y[tr])
        pi = sklearn_perm_imp(model, X[te], y[te], n_repeats=10, random_state=42,
                              scoring="neg_mean_absolute_error")
        order = np.argsort(pi.importances_mean)[::-1]
        imp_list = []
        for i in order[:10]:
            imp_list.append({"feature": fcols[i], "importance": float(pi.importances_mean[i])})
        perm_imp[sname] = imp_list
        print(f"  {sname}: top={fcols[order[0]]} ({pi.importances_mean[order[0]]:.4f})")

    # ========================================================
    # 5. SUBGROUP ANALYSIS
    # ========================================================
    print(f"\n{'='*60}\nSUBGROUP ANALYSIS\n{'='*60}")
    subgroup_results = {}
    sg_defs = {
        "emmetropic_hyperopic": df[TARGET_COL] >= 0,
        "mild_myopia": (df[TARGET_COL] < 0) & (df[TARGET_COL] >= -3),
        "moderate_myopia": (df[TARGET_COL] < -3) & (df[TARGET_COL] >= -6),
        "high_myopia": df[TARGET_COL] < -6,
        "age_6_9": (df["age"] >= 6) & (df["age"] <= 9),
        "age_10_13": (df["age"] >= 10) & (df["age"] <= 13),
        "age_14_plus": df["age"] >= 14,
    }
    sg_scenarios = ["S01_AR_Pre", "S04_ER_Pre", "S07_AR_Pre_IOL", "S10_ER_Pre_IOL"]
    sg_models = ["TabPFN", "XGBoost"]

    for sg_name, sg_mask in sg_defs.items():
        subgroup_results[sg_name] = {"n_eyes": int(sg_mask.sum())}
        print(f"  {sg_name}: n={sg_mask.sum()}")
        for sname in sg_scenarios:
            fcols = SCENARIOS[sname]
            subgroup_results[sg_name][sname] = {}
            for mname in sg_models:
                key = f"{mname}_{sname}"
                if key in oof_store:
                    p = oof_store[key]
                    valid = ~np.isnan(p) & sg_mask.values
                    if valid.sum() > 10:
                        m = compute_metrics(y[valid], p[valid])
                        subgroup_results[sg_name][sname][mname] = m

    # ========================================================
    # 6. LEARNING CURVE
    # ========================================================
    print(f"\n{'='*60}\nLEARNING CURVE (S04_ER_Pre)\n{'='*60}")
    learning_curve = {}
    lc_fcols = SCENARIOS["S04_ER_Pre"]
    lc_X = df[lc_fcols].values.astype(float)
    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    lc_models = {"TabPFN": make_tabpfn, "XGBoost": make_xgb, "RF": make_rf, "Ridge": make_ridge}

    for mname, ctor in lc_models.items():
        learning_curve[mname] = {"fractions": fractions, "mae_mean": [], "mae_std": [], "n_train": []}
        for frac in fractions:
            seed_maes = []
            for seed in ALL_SEEDS:
                np.random.seed(seed)
                gkf = GroupKFold(n_splits=N_FOLDS)
                preds = np.full(len(y), np.nan)
                for tr, te in gkf.split(lc_X, y, groups):
                    train_patients = np.unique(groups[tr])
                    n_keep = max(2, int(len(train_patients) * frac))
                    rng = np.random.RandomState(seed)
                    kept = set(rng.choice(train_patients, n_keep, replace=False))
                    sub = tr[np.array([groups[i] in kept for i in tr])]
                    if len(sub) < 5:
                        continue
                    model = ctor(seed)
                    try:
                        if mname == "TabPFN":
                            model.fit(pd.DataFrame(lc_X[sub], columns=lc_fcols), y[sub])
                            p = model.predict(pd.DataFrame(lc_X[te], columns=lc_fcols))
                        else:
                            model.fit(lc_X[sub], y[sub])
                            p = model.predict(lc_X[te])
                        if mname == "Ridge":
                            p = np.clip(p, CLIP_MIN, CLIP_MAX)
                        preds[te] = p
                    except:
                        pass
                valid = ~np.isnan(preds)
                if valid.sum() > 0:
                    seed_maes.append(mean_absolute_error(y[valid], preds[valid]))
            n_approx = int(len(y) * 0.8 * frac)
            learning_curve[mname]["mae_mean"].append(float(np.mean(seed_maes)) if seed_maes else np.nan)
            learning_curve[mname]["mae_std"].append(float(np.std(seed_maes)) if seed_maes else np.nan)
            learning_curve[mname]["n_train"].append(n_approx)
            print(f"  {mname:10s} frac={frac:.0%}: MAE={np.mean(seed_maes):.3f}" if seed_maes else "")

    # ========================================================
    # 7. ERROR DISTRIBUTION
    # ========================================================
    print(f"\n{'='*60}\nERROR DISTRIBUTION\n{'='*60}")
    error_dist = {}
    for key_label in ["TabPFN_S01_AR_Pre", "TabPFN_S04_ER_Pre",
                       "TabPFN_S07_AR_Pre_IOL", "TabPFN_S10_ER_Pre_IOL",
                       "XGBoost_S01_AR_Pre", "XGBoost_S04_ER_Pre"]:
        if key_label in oof_store:
            p = oof_store[key_label]
            valid = ~np.isnan(p)
            errs = np.abs(y[valid] - p[valid])
            error_dist[key_label] = {
                "P50": float(np.percentile(errs, 50)),
                "P90": float(np.percentile(errs, 90)),
                "P95": float(np.percentile(errs, 95)),
                "P99": float(np.percentile(errs, 99)),
                "max": float(np.max(errs)),
            }
            print(f"  {key_label}: P90={error_dist[key_label]['P90']:.3f} P95={error_dist[key_label]['P95']:.3f}")

    # ========================================================
    # 8. GENERATE FIGURES
    # ========================================================
    print(f"\n{'='*60}\nGENERATING FIGURES\n{'='*60}")

    def panel_label(ax, label, x=0.02, y=0.97):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="top", ha="left", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    # --- Fig: Model comparison (2 panels: device-only vs device+IOL) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scenarios_left = ["S01_AR_Pre", "S02_AR_Post", "S03_AR_PrePost",
                      "S04_ER_Pre", "S05_ER_Post", "S06_ER_PrePost"]
    scenarios_right = ["S07_AR_Pre_IOL", "S08_AR_Post_IOL", "S09_AR_PrePost_IOL",
                       "S10_ER_Pre_IOL", "S11_ER_Post_IOL", "S12_ER_PrePost_IOL"]
    labels_left = ["AR\nPre", "AR\nPost", "AR\nPre+Post", "ER\nPre", "ER\nPost", "ER\nPre+Post"]
    labels_right = ["AR+IOL\nPre", "AR+IOL\nPost", "AR+IOL\nPre+Post",
                    "ER+IOL\nPre", "ER+IOL\nPost", "ER+IOL\nPre+Post"]
    core_models = ["Ridge", "RF", "XGBoost", "LightGBM", "CatBoost", "TabPFN", "SVR", "MLP"]

    for ax_idx, (sc_list, sc_labels, title) in enumerate([
        (scenarios_left, labels_left, "Device Only"),
        (scenarios_right, labels_right, "Device + IOL Master"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(sc_list))
        w = 0.09
        offsets = np.arange(len(core_models)) * w - (len(core_models) - 1) * w / 2
        for i, mn in enumerate(core_models):
            vals = [all_results.get(sc, {}).get(mn, {}).get("MAE", np.nan) for sc in sc_list]
            vals_d = [min(v, 1.5) if not np.isnan(v) else 0 for v in vals]
            bars = ax.bar(x + offsets[i], vals_d, w, label=mn if ax_idx == 0 else "",
                          color=C.get(mn, "#888"), edgecolor="white", linewidth=0.4)
            if mn == "TabPFN":
                for j, bar in enumerate(bars):
                    if bar.get_height() > 0.01:
                        ax.text(bar.get_x() + w/2, bar.get_height() + 0.01,
                                f"{vals_d[j]:.3f}", ha="center", va="bottom", fontsize=6.5,
                                color=C["TabPFN"], fontweight="bold")
        ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sc_labels, fontsize=9)
        ax.set_ylabel("MAE (D)")
        ax.set_ylim(0, 1.3)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        panel_label(ax, f"({'ab'[ax_idx]})")

    axes[0].legend(loc="upper center", ncol=4, fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_model_comparison_12s.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_model_comparison_12s.png")

    # --- Fig: Raw device agreement Bland-Altman ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for idx, (device, title) in enumerate([
        ("eyerobo_pre", "Eyerobo VS (Pre-dilation)"),
        ("auto_pre", "Auto-refractor (Pre-dilation)"),
    ]):
        ra = raw_agreement[device]
        sph_col = f"{device}_SPH"
        cyl_col = f"{device}_CYL"
        valid = df[sph_col].notna() & df[cyl_col].notna()
        dev_se = (df.loc[valid, sph_col] + df.loc[valid, cyl_col] / 2).values
        cyc_se = df.loc[valid, TARGET_COL].values
        means = (cyc_se + dev_se) / 2
        diffs = cyc_se - dev_se
        md, loa_l, loa_u = ra["mean_diff"], ra["LoA_lower"], ra["LoA_upper"]

        ax = axes[idx]
        ax.scatter(means, diffs, alpha=0.3, s=10, color="#2C3E50", edgecolors="none")
        ax.axhline(md, color="#c0392b", linewidth=2.0, label=f"Mean: {md:+.3f} D")
        ax.axhline(loa_u, color="#2980b9", linewidth=1.5, linestyle="--",
                   label=f"+1.96 SD: {loa_u:+.3f} D")
        ax.axhline(loa_l, color="#2980b9", linewidth=1.5, linestyle="--",
                   label=f"\u20131.96 SD: {loa_l:+.3f} D")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Mean of Cycloplegic SE and Device SE (D)")
        ax.set_ylabel("Cycloplegic SE \u2013 Device SE (D)")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
        ax.grid(axis="y", alpha=0.2)
        panel_label(ax, f"({'ab'[idx]})")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_bland_altman_raw.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_bland_altman_raw.png")

    # --- Fig: ML-corrected Bland-Altman (TabPFN on 4 key scenarios) ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    key_scenarios = [("S01_AR_Pre", "AR Pre"), ("S04_ER_Pre", "ER Pre"),
                     ("S07_AR_Pre_IOL", "AR Pre + IOL"), ("S10_ER_Pre_IOL", "ER Pre + IOL")]
    for idx, (sname, title) in enumerate(key_scenarios):
        ax = axes[idx // 2][idx % 2]
        key = f"TabPFN_{sname}"
        if key in oof_store:
            p = oof_store[key]
            valid = ~np.isnan(p)
            yv, pv = y[valid], p[valid]
            means = (yv + pv) / 2
            diffs = yv - pv
            md = np.mean(diffs)
            sd = np.std(diffs, ddof=1)
            loa_u, loa_l = md + 1.96*sd, md - 1.96*sd

            ax.scatter(means, diffs, alpha=0.3, s=10, color="#2C3E50", edgecolors="none")
            ax.axhline(md, color="#c0392b", linewidth=2.0, label=f"Mean: {md:+.3f} D")
            ax.axhline(loa_u, color="#2980b9", linewidth=1.5, linestyle="--",
                       label=f"+1.96SD: {loa_u:+.3f} D")
            ax.axhline(loa_l, color="#2980b9", linewidth=1.5, linestyle="--",
                       label=f"\u20131.96SD: {loa_l:+.3f} D")
            ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Mean of Cycloplegic SE and Predicted SE (D)")
        ax.set_ylabel("Cycloplegic SE \u2013 Predicted SE (D)")
        ax.set_title(f"TabPFN: {title}", fontsize=12)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
        ax.grid(axis="y", alpha=0.2)
        panel_label(ax, f"({'abcd'[idx]})")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_bland_altman_ml.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_bland_altman_ml.png")

    # --- Fig: Scatter predicted vs actual ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (sname, title) in enumerate(key_scenarios):
        ax = axes[idx // 2][idx % 2]
        key = f"TabPFN_{sname}"
        if key in oof_store:
            p = oof_store[key]
            valid = ~np.isnan(p)
            yv, pv = y[valid], p[valid]
            m = compute_metrics(yv, pv)
            ax.scatter(yv, pv, alpha=0.3, s=10, color="#2C3E50", edgecolors="none")
            lims = [min(yv.min(), pv.min()) - 0.5, max(yv.max(), pv.max()) + 0.5]
            ax.plot(lims, lims, color="#c0392b", linestyle="--", linewidth=2.0, alpha=0.8)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_aspect("equal")
            txt = f"MAE = {m['MAE']:.3f} D\nR\u00b2 = {m['R2']:.3f}\nr = {m['pearson_r']:.3f}\n\u22640.50D: {m['within_0.50D']:.1f}%"
            ax.text(0.97, 0.05, txt, transform=ax.transAxes, va="bottom", ha="right", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffcc", alpha=0.9, edgecolor="0.7"))
        ax.set_xlabel("Cycloplegic SE (D)")
        ax.set_ylabel("Predicted SE (D)")
        ax.set_title(f"TabPFN: {title}", fontsize=12)
        ax.grid(alpha=0.2)
        panel_label(ax, f"({'abcd'[idx]})")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_scatter_ml.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_scatter_ml.png")

    # --- Fig: Head-to-head clinical thresholds (ER vs AR) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [("S01_AR_Pre", "S04_ER_Pre", "Pre-dilation"),
             ("S03_AR_PrePost", "S06_ER_PrePost", "Pre+Post"),
             ("S07_AR_Pre_IOL", "S10_ER_Pre_IOL", "Pre + IOL Master")]
    thresh_keys = ["within_0.25D", "within_0.50D", "within_1.00D"]
    thresh_labels = ["\u00b10.25 D", "\u00b10.50 D", "\u00b11.00 D"]

    for ax_idx, (ar_sc, er_sc, title) in enumerate(pairs):
        ax = axes[ax_idx]
        x = np.arange(len(thresh_keys))
        w = 0.3
        for i, (sc, lbl, col) in enumerate([(ar_sc, "Auto-refractor", C["XGBoost"]),
                                              (er_sc, "Eyerobo VS", C["TabPFN"])]):
            vals = [all_results.get(sc, {}).get("TabPFN", {}).get(k, 0) for k in thresh_keys]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lbl, color=col, edgecolor="white", linewidth=0.6)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + w/2, bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(thresh_labels)
        ax.set_ylabel("Percentage of Eyes (%)")
        ax.set_ylim(0, 110)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        panel_label(ax, f"({'abc'[ax_idx]})")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_headtohead_thresholds.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_headtohead_thresholds.png")

    # --- Fig: Learning curve ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lc_colors = {"TabPFN": C["TabPFN"], "XGBoost": C["XGBoost"], "RF": C["RF"], "Ridge": C["Ridge"]}
    lc_markers = {"TabPFN": "o", "XGBoost": "s", "RF": "^", "Ridge": "D"}
    y_offsets = {"TabPFN": -12, "XGBoost": 8, "RF": -4, "Ridge": 14}
    for mn in lc_models:
        r = learning_curve[mn]
        n = np.array(r["n_train"])
        m_arr = np.array(r["mae_mean"])
        s_arr = np.array(r["mae_std"])
        ax.plot(n, m_arr, marker=lc_markers[mn], color=lc_colors[mn], label=mn, linewidth=2.5, markersize=8)
        ax.fill_between(n, m_arr - s_arr, m_arr + s_arr, alpha=0.15, color=lc_colors[mn])
        ax.annotate(f"{m_arr[-1]:.3f}", xy=(n[-1], m_arr[-1]),
                    xytext=(10, y_offsets[mn]), textcoords="offset points",
                    fontsize=10, color=lc_colors[mn], fontweight="bold")
    ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("Approximate training set size (eyes)")
    ax.set_ylabel("MAE (D)")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_learning_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_learning_curve.png")

    # --- Fig: Error distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for key, label, col, ls in [
        ("TabPFN_S04_ER_Pre", "TabPFN ER Pre", C["TabPFN"], "-"),
        ("TabPFN_S01_AR_Pre", "TabPFN AR Pre", C["XGBoost"], "--"),
        ("TabPFN_S10_ER_Pre_IOL", "TabPFN ER+IOL", "#009E73", "-"),
    ]:
        if key in oof_store:
            p = oof_store[key]
            valid = ~np.isnan(p)
            errs = np.sort(np.abs(y[valid] - p[valid]))
            cdf = np.arange(1, len(errs)+1) / len(errs) * 100
            ax.plot(errs, cdf, label=label, color=col, linestyle=ls, linewidth=2.5)
    for t in [0.25, 0.50, 1.00]:
        ax.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(t, 103, f"{t:.2f}D", fontsize=9, color="gray", ha="center")
    ax.set_xlabel("Absolute prediction error (D)")
    ax.set_ylabel("Cumulative percentage of eyes (%)")
    ax.set_xlim(0, 3.0); ax.set_ylim(0, 107)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.2)
    panel_label(ax, "(a)")

    ax = axes[1]
    key = "TabPFN_S04_ER_Pre"
    if key in oof_store:
        p = oof_store[key]
        valid = ~np.isnan(p)
        abs_errs = np.abs(y[valid] - p[valid])
        se_v = y[valid]
        bins = {"Emmet./\nHyperop.\n(SE\u22650)": se_v >= 0,
                "Mild\nMyopia\n(0 to \u22123)": (se_v < 0) & (se_v >= -3),
                "Moderate\nMyopia\n(\u22123 to \u22126)": (se_v < -3) & (se_v >= -6),
                "High\nMyopia\n(< \u22126)": se_v < -6}
        box_data, labels, ns, meds = [], [], [], []
        for lbl, mask in bins.items():
            e = abs_errs[mask]
            box_data.append(e); labels.append(lbl); ns.append(int(mask.sum())); meds.append(float(np.median(e)))
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=2.0),
                        whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
                        flierprops=dict(marker=".", markersize=4, alpha=0.4))
        for patch, col in zip(bp["boxes"], ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]):
            patch.set_facecolor(col); patch.set_alpha(0.6); patch.set_linewidth(1.5)
        for i, (n, med) in enumerate(zip(ns, meds)):
            ax.text(i+1, -0.16, f"n={n}", ha="center", va="top", fontsize=10,
                    transform=ax.get_xaxis_transform())
            ax.text(i+1+0.32, med+0.08, f"{med:.2f}", ha="left", va="bottom", fontsize=9.5, fontweight="bold")
        ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
        ax.set_ylabel("Absolute prediction error (D)")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "(b)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_error_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_error_distribution.png")

    # ========================================================
    # 9. SAVE ALL RESULTS
    # ========================================================
    elapsed = time.time() - t0
    results = {
        "all_results": all_results,
        "significance_tests": sig_tests,
        "raw_device_agreement": raw_agreement,
        "permutation_importance": perm_imp,
        "subgroup_results": subgroup_results,
        "learning_curve": learning_curve,
        "error_distribution": error_dist,
        "metadata": {
            "n_folds": N_FOLDS, "seeds": ALL_SEEDS,
            "n_eyes": len(df), "n_patients": int(df["patient_id"].nunique()),
            "target": TARGET_COL,
            "scenarios": {k: len(v) for k, v in SCENARIOS.items()},
            "models": list(MODELS.keys()),
            "total_time_seconds": round(elapsed, 1),
            "dataset": "GBD_Eyerobo_Vision_Screener_Myopia_Study_Mar_2026.xlsx",
        }
    }

    out_path = os.path.join(OUT_DIR, "comprehensive_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results: {out_path}")
    print(f"Figures: {FIG_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
