"""
SCRIPT 1: Clean the March 2026 final dataset and generate descriptive statistics.
Produces: cleaned pickle, CSV, dataset_statistics.json, and 3 dataset figures.
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_and_clean_data as _load_original, FEATURE_GROUPS

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../data/dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../paper/figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")

plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.labelsize": 14,
    "axes.titlesize": 14, "legend.fontsize": 11, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "figure.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 1.2,
})

# ============================================================
# LOAD (reuse prepare.py column mapping)
# ============================================================
def load_raw():
    """Load new dataset using the same column mapping as prepare.py."""
    df = pd.read_excel(DATA_PATH, header=1)
    df = df.iloc[1:].copy().reset_index(drop=True)

    col_map = {
        "age": "age", "gender": "gender", "Eye position": "eye_position",
        "IOP": "IOP", "Glasses/OK lens/Atropine/No-myopia": "treatment",
        "Patient Name": "patient_id", "date": "date",
        "Height": "height", "Weight/kg": "weight",
        "SPH": "target_SPH", "CYL": "target_CYL", "AX": "target_AX", "SE": "target_SE",
        "AL": "iol_pre_AL", "LT": "iol_pre_LT", "WTW": "iol_pre_WTW",
        "lx": "iol_pre_lx", "ly": "iol_pre_ly", "CCT": "iol_pre_CCT",
        "CW": "iol_pre_CW", "Pupil": "iol_pre_Pupil", "SE.1": "iol_pre_SE",
        "K1": "iol_pre_K1", "K2": "iol_pre_K2", "ΔK": "iol_pre_dK",
        "TSE": "iol_pre_TSE", "TK1": "iol_pre_TK1", "TK2": "iol_pre_TK2",
        "ΔTK": "iol_pre_dTK", "ACD": "iol_pre_ACD",
        "AL.1": "iol_post_AL", "LT.1": "iol_post_LT", "WTW.1": "iol_post_WTW",
        "lx.1": "iol_post_lx", "ly.1": "iol_post_ly", "CCT.1": "iol_post_CCT",
        "CW.1": "iol_post_CW", "Pupil.1": "iol_post_Pupil", "SE.2": "iol_post_SE",
        "K1.1": "iol_post_K1", "K2.1": "iol_post_K2", "ΔK.1": "iol_post_dK",
        "TSE.1": "iol_post_TSE", "TK1.1": "iol_post_TK1", "TK2.1": "iol_post_TK2",
        "ΔTK.1": "iol_post_dTK", "ACD.1": "iol_post_ACD",
        "E-SPH": "eyerobo_pre_SPH", "E-CYL": "eyerobo_pre_CYL",
        "AX.1": "eyerobo_pre_AX", "Pupil diameter": "eyerobo_pre_pupil",
        "E'-SPH": "eyerobo_post_SPH", "E'-CYL": "eyerobo_post_CYL",
        "AX.2": "eyerobo_post_AX", "Pupil diameter.1": "eyerobo_post_pupil",
        "A-SPH": "auto_pre_SPH", "A-CYL": "auto_pre_CYL",
        "AX.3": "auto_pre_AX", "Pupil diameter.2": "auto_pre_pupil",
        "A'-SPH": "auto_post_SPH", "A'-CYL": "auto_post_CYL",
        "AX.4": "auto_post_AX", "Pupil diameter.3": "auto_post_pupil",
        "pupil distance": "eyerobo_pre_PD", "pupil distance.1": "eyerobo_post_PD",
    }
    df = df.rename(columns=col_map)

    # Handle Eyerobo out-of-range markers
    for col in ["eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_post_SPH", "eyerobo_post_CYL"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("＜", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert to numeric
    skip = {"gender", "eye_position", "treatment", "patient_id", "date"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode categoricals
    df["gender_enc"] = (df["gender"] == "M").astype(float)
    df["eye_enc"] = (df["eye_position"] == "OD").astype(float)
    df["treatment_enc"] = df["treatment"].map({"No glasses": 0, "Glasses": 1, "myopia": 2}).astype(float)

    return df


# ============================================================
# CLEANING
# ============================================================
def clean(df):
    print("Cleaning dataset...")
    n_before = len(df)

    # 1. Remove rows with missing target or patient_id
    df = df[df["target_SE"].notna() & df["patient_id"].notna()].copy()
    print(f"  After removing missing target/patient_id: {len(df)} rows")

    # 2. Remove exact duplicate rows for xie xinqi
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c not in ["date"]], keep="first")
    print(f"  After removing exact duplicates: {len(df)} rows (dropped {before - len(df)})")

    # 3. Handle wang zixi: drop rows where target_SPH is NaN
    wz_mask = (df["patient_id"] == "wang zixi") & df["target_SPH"].isna()
    df = df[~wz_mask].copy()
    print(f"  After handling wang zixi incomplete rows: {len(df)} rows")

    # 4. Fix obvious outliers
    fixes = 0
    # K2 = 4306 -> 43.06
    mask = df["iol_pre_K2"] > 1000
    fixes += mask.sum()
    df.loc[mask, "iol_pre_K2"] = df.loc[mask, "iol_pre_K2"] / 100

    # WTW > 50 -> divide by 10
    for col in ["iol_pre_WTW", "iol_post_WTW"]:
        if col in df.columns:
            mask = df[col] > 50
            fixes += mask.sum()
            df.loc[mask, col] = df.loc[mask, col] / 10

    # WTW < 0 -> NaN
    for col in ["iol_pre_WTW", "iol_post_WTW"]:
        if col in df.columns:
            mask = df[col] < 0
            fixes += mask.sum()
            df.loc[mask, col] = np.nan

    # Pupil > 15 -> NaN (impossible pupil diameter)
    for col in ["iol_pre_Pupil", "iol_post_Pupil"]:
        if col in df.columns:
            mask = df[col] > 15
            fixes += mask.sum()
            df.loc[mask, col] = np.nan

    # IOP > 45 -> NaN
    mask = df["IOP"] > 45
    fixes += mask.sum()
    df.loc[mask, "IOP"] = np.nan

    # K2 < 0 -> NaN (impossible keratometry)
    for col in ["iol_pre_K2", "iol_post_K2"]:
        if col in df.columns:
            mask = df[col] < 30
            n_bad = mask.sum()
            if n_bad > 0:
                fixes += n_bad
                df.loc[mask, col] = np.nan

    print(f"  Fixed {fixes} outlier values")

    # 5. Handle repeat visits: keep first visit only
    # Parse date
    df["_parsed_date"] = pd.to_datetime(df["date"], format="%y.%m.%d", errors="coerce")

    patients_multiple = df.groupby("patient_id").filter(lambda g: len(g) > 2)["patient_id"].unique()
    rows_to_drop = []
    for pid in patients_multiple:
        pdata = df[df["patient_id"] == pid]
        dates = pdata["_parsed_date"].dropna()
        if len(dates) > 0:
            earliest = dates.min()
            # Keep rows from earliest date, or rows with no date (original batch)
            keep_mask = (pdata["_parsed_date"] == earliest) | pdata["_parsed_date"].isna()
            drop_idx = pdata[~keep_mask].index
            rows_to_drop.extend(drop_idx)

    if rows_to_drop:
        df = df.drop(rows_to_drop).copy()
        print(f"  Dropped {len(rows_to_drop)} repeat-visit rows ({len(patients_multiple)} patients)")

    # Check remaining patients with >2 rows
    still_multi = df.groupby("patient_id").filter(lambda g: len(g) > 2)
    if len(still_multi) > 0:
        # For remaining, just keep first 2 rows per patient
        keep_idx = df.groupby("patient_id").head(2).index
        n_extra = len(df) - len(keep_idx)
        df = df.loc[keep_idx].copy()
        if n_extra > 0:
            print(f"  Trimmed {n_extra} extra rows to enforce max 2 per patient")

    df = df.drop(columns=["_parsed_date"], errors="ignore")
    print(f"  Final: {len(df)} rows, {df['patient_id'].nunique()} patients")
    return df.reset_index(drop=True)


# ============================================================
# STATISTICS
# ============================================================
def compute_stats(df):
    stats = {}
    stats["n_eyes"] = len(df)
    stats["n_patients"] = int(df["patient_id"].nunique())
    stats["n_male"] = int((df["gender"] == "M").sum())
    stats["n_female"] = int((df["gender"] == "F").sum())
    stats["pct_female"] = round(stats["n_female"] / stats["n_eyes"] * 100, 1)

    # Age
    age = df["age"].dropna()
    stats["age"] = {"mean": round(float(age.mean()), 1), "std": round(float(age.std()), 1),
                    "median": float(age.median()), "min": int(age.min()), "max": int(age.max())}

    # Target SE
    se = df["target_SE"].dropna()
    stats["target_SE"] = {"mean": round(float(se.mean()), 2), "std": round(float(se.std()), 2),
                          "median": round(float(se.median()), 2),
                          "min": round(float(se.min()), 2), "max": round(float(se.max()), 2)}

    # Refractive categories
    stats["refractive_groups"] = {
        "emmetropic_hyperopic_ge0": int((se >= 0).sum()),
        "mild_myopia_0_to_-3": int(((se < 0) & (se >= -3)).sum()),
        "moderate_myopia_-3_to_-6": int(((se < -3) & (se >= -6)).sum()),
        "high_myopia_lt_-6": int((se < -6).sum()),
    }

    # Eye position
    stats["eye_position"] = {"OD": int((df["eye_position"] == "OD").sum()),
                             "OS": int((df["eye_position"] == "OS").sum())}

    # Treatment
    tx = df["treatment"].value_counts().to_dict()
    stats["treatment"] = {str(k): int(v) for k, v in tx.items()}

    # Missing data rates for Eric's 12 scenarios
    scenario_features = {
        "auto_pre_SPH_CYL_AX": ["auto_pre_SPH", "auto_pre_CYL", "auto_pre_AX"],
        "auto_post_SPH_CYL_AX": ["auto_post_SPH", "auto_post_CYL", "auto_post_AX"],
        "eyerobo_pre_SPH_CYL_AX": ["eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_pre_AX"],
        "eyerobo_post_SPH_CYL_AX": ["eyerobo_post_SPH", "eyerobo_post_CYL", "eyerobo_post_AX"],
        "iol_biometry": ["iol_pre_AL", "iol_pre_LT", "iol_pre_WTW", "iol_pre_CCT",
                         "iol_pre_K1", "iol_pre_K2", "iol_pre_dK", "iol_pre_ACD", "iol_pre_SE"],
        "demographics": ["age", "IOP", "gender_enc", "eye_enc", "treatment_enc"],
    }

    missing = {}
    for group, cols in scenario_features.items():
        rates = {}
        for c in cols:
            if c in df.columns:
                rates[c] = round(float(df[c].isna().mean() * 100), 1)
        missing[group] = rates
    stats["missing_rates"] = missing

    # Per-scenario complete sample sizes
    demo = ["age", "IOP", "gender_enc", "eye_enc", "treatment_enc"]
    iol_bio = ["iol_pre_AL", "iol_pre_LT", "iol_pre_WTW", "iol_pre_CCT",
               "iol_pre_K1", "iol_pre_K2", "iol_pre_dK", "iol_pre_ACD", "iol_pre_SE"]

    # Note: models handle NaN natively, so "complete" = has target_SE + patient_id
    stats["total_available_eyes"] = len(df)

    return stats


# ============================================================
# FIGURES
# ============================================================
def make_figures(df):
    os.makedirs(FIG_DIR, exist_ok=True)
    se = df["target_SE"].dropna()

    # Fig: SE distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(se.min() - 0.5, se.max() + 1, 0.5)
    ax.hist(se, bins=bins, color="#3498db", edgecolor="white", linewidth=0.5, alpha=0.8)
    for thresh, lbl, col in [(0, "Emmetropia", "#e74c3c"), (-3, "-3 D", "#f39c12"), (-6, "-6 D", "#9b59b6")]:
        ax.axvline(thresh, color=col, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(thresh + 0.15, ax.get_ylim()[1] * 0.9, lbl, fontsize=10, color=col, rotation=90, va="top")

    # Annotate counts
    n_emm = (se >= 0).sum()
    n_mild = ((se < 0) & (se >= -3)).sum()
    n_mod = ((se < -3) & (se >= -6)).sum()
    n_high = (se < -6).sum()
    txt = (f"Emmet./Hyperop.: {n_emm} ({n_emm/len(se)*100:.1f}%)\n"
           f"Mild myopia: {n_mild} ({n_mild/len(se)*100:.1f}%)\n"
           f"Moderate myopia: {n_mod} ({n_mod/len(se)*100:.1f}%)\n"
           f"High myopia: {n_high} ({n_high/len(se)*100:.1f}%)")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffcc", alpha=0.9, edgecolor="0.7"))
    ax.set_xlabel("Cycloplegic Spherical Equivalent (D)")
    ax.set_ylabel("Number of Eyes")
    ax.set_title(f"Distribution of Cycloplegic SE (n = {len(se)} eyes)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_se_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_se_distribution.png")

    # Fig: Age distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ages = df["age"].dropna().astype(int)
    age_counts = ages.value_counts().sort_index()
    ax.bar(age_counts.index, age_counts.values, color="#2ecc71", edgecolor="white", linewidth=0.5)
    for x, y in zip(age_counts.index, age_counts.values):
        ax.text(x, y + 2, str(y), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Eyes")
    ax.set_title(f"Age Distribution (n = {len(ages)}, mean = {ages.mean():.1f} ± {ages.std():.1f})", fontsize=13)
    ax.set_xticks(range(int(ages.min()), int(ages.max()) + 1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_age_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_age_distribution.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DATASET CLEANING AND ANALYSIS")
    print("=" * 60)

    df = load_raw()
    print(f"Raw dataset: {len(df)} rows, {df['patient_id'].nunique()} patients")

    df = clean(df)

    stats = compute_stats(df)
    print(f"\nDataset summary:")
    print(f"  Eyes: {stats['n_eyes']}, Patients: {stats['n_patients']}")
    print(f"  Male: {stats['n_male']}, Female: {stats['n_female']} ({stats['pct_female']}%)")
    print(f"  Age: {stats['age']['mean']} ± {stats['age']['std']} (range {stats['age']['min']}-{stats['age']['max']})")
    print(f"  SE: {stats['target_SE']['mean']} ± {stats['target_SE']['std']} D")
    print(f"  Refractive groups: {stats['refractive_groups']}")

    make_figures(df)

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    pkl_path = os.path.join(DATA_DIR, "dataset.csv")
    csv_path = os.path.join(DATA_DIR, "cleaned_dataset_mar2026.csv")
    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved cleaned dataset: {pkl_path}")

    stats_path = os.path.join(OUT_DIR, "dataset_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved statistics: {stats_path}")

    print("\nDone!")
