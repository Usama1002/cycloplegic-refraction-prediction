"""
POWER VECTOR EXPERIMENTS
========================
Extends the 12-scenario framework with J0/J45 targets.
Produces:
  1. Raw device agreement table (Eric's Table 1 style) for SPH, CYL, SE, J0, J45
  2. ML predictions for SE, J0, J45 across 12 scenarios (TabPFN, XGBoost, Ridge, RF)
  3. Bland-Altman + correlation for all targets
  4. Comprehensive JSON with ALL results for paper writing
"""

import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================
N_FOLDS = 5
ALL_SEEDS = [42, 123, 456]
CLIP_MIN, CLIP_MAX = -15.0, 10.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PKL = os.path.join(BASE_DIR, "../../data/dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "..")
FIG_DIR = os.path.join(BASE_DIR, "../../paper/figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEMO = ["age", "IOP", "gender_enc", "eye_enc", "treatment_enc"]
IOL_BIO = ["iol_pre_AL", "iol_pre_LT", "iol_pre_WTW", "iol_pre_CCT",
           "iol_pre_K1", "iol_pre_K2", "iol_pre_dK", "iol_pre_ACD", "iol_pre_SE"]

# ============================================================
# POWER VECTOR FORMULAS
# ============================================================
def compute_power_vectors(sph, cyl, axis_deg):
    """Compute SE, J0, J45 from SPH, CYL, axis (degrees)."""
    axis_rad = np.deg2rad(axis_deg.astype(float))
    se = sph + cyl / 2
    j0 = -(cyl / 2) * np.cos(2 * axis_rad)
    j45 = -(cyl / 2) * np.sin(2 * axis_rad)
    return se, j0, j45

# ============================================================
# 12 SCENARIOS
# ============================================================
def _expand(groups):
    FG = {
        "auto_pre": ["auto_pre_SPH", "auto_pre_CYL", "auto_pre_AX"],
        "auto_post": ["auto_post_SPH", "auto_post_CYL", "auto_post_AX"],
        "eyerobo_pre": ["eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_pre_AX"],
        "eyerobo_post": ["eyerobo_post_SPH", "eyerobo_post_CYL", "eyerobo_post_AX"],
        "iol_biometry": IOL_BIO,
        "demographics": DEMO,
    }
    feats = []
    for g in groups:
        feats.extend(FG[g])
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
# MODELS
# ============================================================
def make_tabpfn(s=42):
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor.create_default_for_version("v2", device="auto", n_estimators=8)

def make_xgb(s=42):
    return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=s, n_jobs=-1, verbosity=0)

def make_rf(s=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=100, max_depth=10,
                            min_samples_leaf=5, random_state=s, n_jobs=-1))])

def make_ridge(s=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])

MODELS = {"TabPFN": make_tabpfn, "XGBoost": make_xgb, "RF": make_rf, "Ridge": make_ridge}

# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred):
    ae = np.abs(y_true - y_pred)
    diff = y_true - y_pred
    sd = np.std(diff, ddof=1)
    md = np.mean(diff)
    r_p, _ = pearsonr(y_true, y_pred)
    r_s, _ = spearmanr(y_true, y_pred)
    return {
        "MAE": float(np.mean(ae)),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(np.median(ae)),
        "within_0.25D": float(np.mean(ae <= 0.25) * 100),
        "within_0.50D": float(np.mean(ae <= 0.50) * 100),
        "within_1.00D": float(np.mean(ae <= 1.00) * 100),
        "mean_diff": float(md),
        "std_diff": float(sd),
        "LoA_lower": float(md - 1.96 * sd),
        "LoA_upper": float(md + 1.96 * sd),
        "pearson_r": float(r_p),
        "spearman_rho": float(r_s),
    }

def compute_agreement(device_vals, cyclo_vals):
    """Raw device agreement statistics (no ML)."""
    valid = ~(np.isnan(device_vals) | np.isnan(cyclo_vals))
    if valid.sum() < 10:
        return None
    dv, cv = device_vals[valid], cyclo_vals[valid]
    diff = cv - dv  # cycloplegic minus device
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    se_diff = sd / np.sqrt(len(diff))
    try:
        t_stat, t_pval = ttest_rel(cv, dv)
    except:
        t_pval = np.nan
    try:
        r_p, _ = pearsonr(cv, dv)
    except:
        r_p = np.nan
    return {
        "n": int(valid.sum()),
        "cyclo_mean": float(np.mean(cv)), "cyclo_sd": float(np.std(cv, ddof=1)),
        "cyclo_min": float(np.min(cv)), "cyclo_max": float(np.max(cv)),
        "device_mean": float(np.mean(dv)), "device_sd": float(np.std(dv, ddof=1)),
        "device_min": float(np.min(dv)), "device_max": float(np.max(dv)),
        "mean_diff": float(md), "sd_diff": float(sd),
        "ci95_lower": float(md - 1.96 * se_diff),
        "ci95_upper": float(md + 1.96 * se_diff),
        "LoA_lower": float(md - 1.96 * sd),
        "LoA_upper": float(md + 1.96 * sd),
        "pct_within_LoA": float(np.mean((diff >= md - 1.96*sd) & (diff <= md + 1.96*sd)) * 100),
        "p_value": float(t_pval),
        "pearson_r": float(r_p),
    }

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model_name, constructor, X, y, groups, fcols, seeds=ALL_SEEDS):
    seed_metrics = []
    all_fold_maes = []
    oof_preds = np.full(len(y), np.nan)

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
            except:
                fold_maes.append(np.nan)

        valid = ~np.isnan(preds)
        if valid.sum() > 0:
            seed_metrics.append(compute_metrics(y[valid], preds[valid]))
        all_fold_maes.append(np.array(fold_maes))
        if si == 0:
            oof_preds = preds.copy()

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
    print("POWER VECTOR EXPERIMENTS: SE + J0 + J45")
    print("=" * 70)

    df = pd.read_csv(DATA_PKL)
    groups = df["patient_id"].astype(str).values
    print(f"Dataset: {len(df)} eyes, {df['patient_id'].nunique()} patients\n")

    # ========================================================
    # 1. COMPUTE POWER VECTORS FOR ALL SOURCES
    # ========================================================
    print("Computing power vectors...")
    sources = {
        "cyclo":       ("target_SPH", "target_CYL", "target_AX"),
        "eyerobo_pre": ("eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_pre_AX"),
        "eyerobo_post":("eyerobo_post_SPH", "eyerobo_post_CYL", "eyerobo_post_AX"),
        "auto_pre":    ("auto_pre_SPH", "auto_pre_CYL", "auto_pre_AX"),
        "auto_post":   ("auto_post_SPH", "auto_post_CYL", "auto_post_AX"),
    }

    for src, (sph_col, cyl_col, ax_col) in sources.items():
        se, j0, j45 = compute_power_vectors(
            df[sph_col].values.astype(float),
            df[cyl_col].values.astype(float),
            df[ax_col].values.astype(float)
        )
        df[f"{src}_SE_pv"] = se
        df[f"{src}_J0"] = j0
        df[f"{src}_J45"] = j45

    # Targets
    target_SE = df["target_SE"].values.astype(float)
    target_J0 = df["cyclo_J0"].values.astype(float)
    target_J45 = df["cyclo_J45"].values.astype(float)
    targets = {"SE": target_SE, "J0": target_J0, "J45": target_J45}

    # ========================================================
    # 2. RAW DEVICE AGREEMENT (Table 1 style)
    # ========================================================
    print(f"\n{'='*70}")
    print("RAW DEVICE AGREEMENT (Table 1 style)")
    print(f"{'='*70}")

    raw_agreement = {}
    device_configs = [
        ("eyerobo_pre", "Eyerobo Pre"),
        ("eyerobo_post", "Eyerobo Post"),
        ("auto_pre", "Auto-ref Pre"),
        ("auto_post", "Auto-ref Post"),
    ]
    parameters = ["SPH", "CYL", "SE", "J0", "J45"]

    for dev_key, dev_label in device_configs:
        raw_agreement[dev_key] = {}
        print(f"\n  --- {dev_label} ---")
        for param in parameters:
            if param == "SPH":
                src = sources[dev_key][0]
                tgt = sources["cyclo"][0]
                dv = df[src].values.astype(float)
                cv = df[tgt].values.astype(float)
            elif param == "CYL":
                src = sources[dev_key][1]
                tgt = sources["cyclo"][1]
                dv = df[src].values.astype(float)
                cv = df[tgt].values.astype(float)
            elif param in ["SE", "J0", "J45"]:
                suffix = {"SE": "SE_pv", "J0": "J0", "J45": "J45"}[param]
                dv = df[f"{dev_key}_{suffix}"].values
                cv = df[f"cyclo_{suffix}"].values

            ag = compute_agreement(dv, cv)
            if ag:
                raw_agreement[dev_key][param] = ag
                print(f"    {param:4s}: n={ag['n']:4d}  Δ={ag['mean_diff']:+.3f}±{ag['sd_diff']:.3f}  "
                      f"LoA=[{ag['LoA_lower']:+.3f},{ag['LoA_upper']:+.3f}]  "
                      f"r={ag['pearson_r']:.3f}  p={ag['p_value']:.4f}")

    # ========================================================
    # 3. ML PREDICTIONS: SE + J0 + J45 × 12 scenarios × 4 models
    # ========================================================
    print(f"\n{'='*70}")
    print("ML PREDICTIONS: 3 targets × 12 scenarios × 4 models")
    print(f"{'='*70}")

    ml_results = {}
    oof_store = {}

    for target_name, target_vals in targets.items():
        ml_results[target_name] = {}
        # Skip rows where target is NaN
        valid_target = ~np.isnan(target_vals)
        y = target_vals[valid_target]
        g = groups[valid_target]
        df_valid = df[valid_target]

        print(f"\n  TARGET: {target_name} (n={len(y)} valid)")

        for sname, fcols in SCENARIOS.items():
            X = df_valid[fcols].values.astype(float)
            ml_results[target_name][sname] = {}

            for mname, ctor in MODELS.items():
                agg, fmaes, oof_p = evaluate(mname, ctor, X, y, g, fcols)
                ml_results[target_name][sname][mname] = agg
                oof_store[f"{mname}_{sname}_{target_name}"] = {
                    "preds": oof_p.tolist(),
                    "valid_mask": valid_target.tolist(),
                }

                mae = agg.get("MAE", np.nan)
                r2 = agg.get("R2", np.nan)
                print(f"    {sname:22s} {mname:8s}: MAE={mae:.4f}  R2={r2:.4f}")

    # ========================================================
    # 4. SIGNIFICANCE TESTS (TabPFN vs others, per target per scenario)
    # ========================================================
    print(f"\n{'='*70}")
    print("SIGNIFICANCE TESTS")
    print(f"{'='*70}")

    # Run fresh fold-level tests for key comparisons
    sig_tests = {}
    for target_name, target_vals in targets.items():
        sig_tests[target_name] = {}
        valid_target = ~np.isnan(target_vals)
        y = target_vals[valid_target]
        g = groups[valid_target]
        df_valid = df[valid_target]

        for sname in ["S01_AR_Pre", "S04_ER_Pre", "S07_AR_Pre_IOL", "S10_ER_Pre_IOL"]:
            sig_tests[target_name][sname] = {}
            fcols = SCENARIOS[sname]
            X = df_valid[fcols].values.astype(float)

            # Collect per-fold MAEs for TabPFN and XGBoost
            for mname in ["TabPFN", "XGBoost"]:
                _, fmaes, _ = evaluate(mname, MODELS[mname], X, y, g, fcols, seeds=[42])
                sig_tests[target_name][sname][mname] = np.concatenate(fmaes).tolist()

            # Wilcoxon test
            tab = np.array(sig_tests[target_name][sname].get("TabPFN", []))
            xgb_arr = np.array(sig_tests[target_name][sname].get("XGBoost", []))
            if len(tab) == len(xgb_arr) and len(tab) >= 5:
                try:
                    stat, pv = wilcoxon(tab, xgb_arr, alternative="two-sided")
                    sig_tests[target_name][sname]["wilcoxon_p"] = float(pv)
                    print(f"  {target_name} {sname}: TabPFN vs XGBoost p={pv:.4f}")
                except:
                    pass

    # ========================================================
    # 5. HEAD-TO-HEAD SUMMARY TABLE
    # ========================================================
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD: Eyerobo vs Auto-refractor (TabPFN)")
    print(f"{'='*70}")

    headtohead = {}
    pairs = [
        ("S01_AR_Pre", "S04_ER_Pre", "Pre"),
        ("S02_AR_Post", "S05_ER_Post", "Post"),
        ("S03_AR_PrePost", "S06_ER_PrePost", "Pre+Post"),
        ("S07_AR_Pre_IOL", "S10_ER_Pre_IOL", "Pre+IOL"),
        ("S09_AR_PrePost_IOL", "S12_ER_PrePost_IOL", "Pre+Post+IOL"),
    ]

    print(f"  {'Condition':15s} {'Target':4s}  {'AR MAE':>7s}  {'ER MAE':>7s}  {'Gap':>7s}")
    for ar_sc, er_sc, label in pairs:
        headtohead[label] = {}
        for target_name in ["SE", "J0", "J45"]:
            ar_mae = ml_results[target_name].get(ar_sc, {}).get("TabPFN", {}).get("MAE", np.nan)
            er_mae = ml_results[target_name].get(er_sc, {}).get("TabPFN", {}).get("MAE", np.nan)
            gap = er_mae - ar_mae
            headtohead[label][target_name] = {
                "AR_MAE": ar_mae, "ER_MAE": er_mae, "gap": gap
            }
            print(f"  {label:15s} {target_name:4s}   {ar_mae:7.4f}   {er_mae:7.4f}  {gap:+7.4f}")

    # ========================================================
    # 6. SAVE EVERYTHING
    # ========================================================
    elapsed = time.time() - t0

    results = {
        "raw_device_agreement": raw_agreement,
        "ml_results": ml_results,
        "headtohead": headtohead,
        "metadata": {
            "n_eyes": len(df),
            "n_patients": int(df["patient_id"].nunique()),
            "targets": ["SE", "J0", "J45"],
            "scenarios": {k: {"n_features": len(v), "features": v} for k, v in SCENARIOS.items()},
            "models": list(MODELS.keys()),
            "n_folds": N_FOLDS,
            "seeds": ALL_SEEDS,
            "total_time_seconds": round(elapsed, 1),
            "power_vector_formulas": {
                "SE": "SPH + CYL/2",
                "J0": "-(CYL/2) * cos(2 * axis_rad)",
                "J45": "-(CYL/2) * sin(2 * axis_rad)",
            },
        }
    }

    out_path = os.path.join(OUT_DIR, "power_vector_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"ALL DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
