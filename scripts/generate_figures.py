"""
FINAL PUBLICATION FIGURES — v4
All figures regenerated with consistent styling:
  - Teal (#0077B6) / Coral (#E63946) / Gold (#F4A261) / Slate (#264653) palette
  - 14pt body, 15pt labels, 17pt panel labels
  - Dark scatter dots (#264653, alpha=0.45, size=18)
  - Thick lines (2.5pt mean, 2pt LoA)
  - Clean annotations with white background boxes
  - No in-figure titles (captions in LaTeX)
  - Consistent axis ranges within each figure type
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, "../../experiments/code")
from prepare import load_and_clean_data, FEATURE_GROUPS

OUT = "../../figures/"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# UNIFIED STYLE
# ============================================================
TEAL     = "#0077B6"
CORAL    = "#E63946"
GOLD     = "#F4A261"
SLATE    = "#264653"
SAGE     = "#2A9D8F"
PURPLE   = "#6A4C93"
PINK     = "#E56B6F"
GRAY     = "#888888"
LGRAY    = "#CCCCCC"

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.labelsize": 15,
    "axes.titlesize": 15, "legend.fontsize": 12, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "figure.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 1.3,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "lines.linewidth": 2.0, "lines.markersize": 8,
})

MODEL_COLORS = {"TabPFN": TEAL, "XGBoost": GOLD, "RF": SAGE, "Ridge": GRAY}
MODEL_MARKERS = {"TabPFN": "o", "XGBoost": "s", "RF": "^", "Ridge": "D"}

def panel_label(ax, label, x=0.03, y=0.96):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=18, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

def bland_altman_panel(ax, means, diffs, title=None, show_n=True):
    """Draw a single Bland-Altman panel with consistent styling."""
    md = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    loa_u, loa_l = md + 1.96*sd, md - 1.96*sd

    ax.scatter(means, diffs, alpha=0.40, s=16, color=SLATE, edgecolors="none", zorder=2)
    ax.axhline(md, color=CORAL, linewidth=2.5, zorder=3)
    ax.axhline(loa_u, color=TEAL, linewidth=1.8, linestyle="--", zorder=3)
    ax.axhline(loa_l, color=TEAL, linewidth=1.8, linestyle="--", zorder=3)
    ax.axhline(0, color=LGRAY, linewidth=0.8, linestyle=":", zorder=1)

    # Annotations on left side with white boxes
    xmin = ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else means.min()
    for val, lbl, col in [(md, f"Mean: {md:+.2f}", CORAL),
                           (loa_u, f"+1.96SD: {loa_u:+.2f}", TEAL),
                           (loa_l, f"\u20131.96SD: {loa_l:+.2f}", TEAL)]:
        ax.text(0.97, val, f" {lbl} ", ha="right", va="center", fontsize=11,
                color=col, fontweight="bold", transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor="white", edgecolor=col, alpha=0.90, pad=2,
                          boxstyle="round,pad=0.3", linewidth=0.6))

    if show_n:
        ax.text(0.02, 0.02, f"n = {len(diffs)}", transform=ax.transAxes,
                fontsize=11, color=GRAY, va="bottom")
    ax.set_xlabel("Mean of Cycloplegic and Predicted SE (D)")
    ax.set_ylabel("Cycloplegic \u2013 Predicted SE (D)")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)


# ============================================================
# DATA
# ============================================================
def load_data():
    df = pd.read_csv("../../data/dataset.csv")
    df["eyerobo_pre_SE_raw"] = df["eyerobo_pre_SPH"] + df["eyerobo_pre_CYL"] / 2
    df["eyerobo_post_SE_raw"] = df["eyerobo_post_SPH"] + df["eyerobo_post_CYL"] / 2
    df["auto_pre_SE_raw"] = df["auto_pre_SPH"] + df["auto_pre_CYL"] / 2
    df["auto_post_SE_raw"] = df["auto_post_SPH"] + df["auto_post_CYL"] / 2
    return df

DEMO = ["age", "IOP", "gender_enc", "eye_enc", "treatment_enc"]
IOL_BIO = ["iol_pre_AL", "iol_pre_LT", "iol_pre_WTW", "iol_pre_CCT",
           "iol_pre_K1", "iol_pre_K2", "iol_pre_dK", "iol_pre_ACD", "iol_pre_SE"]

SCENARIOS = {
    "S01_AR_Pre":     ["auto_pre_SPH", "auto_pre_CYL", "auto_pre_AX"] + DEMO,
    "S04_ER_Pre":     ["eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_pre_AX"] + DEMO,
    "S07_AR_Pre_IOL": ["auto_pre_SPH", "auto_pre_CYL", "auto_pre_AX"] + IOL_BIO + DEMO,
    "S10_ER_Pre_IOL": ["eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_pre_AX"] + IOL_BIO + DEMO,
}


def get_oof(model_name, constructor, X, y, groups, fcols, seed=42):
    np.random.seed(seed)
    gkf = GroupKFold(n_splits=5)
    preds = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = constructor(seed)
        if model_name == "TabPFN":
            m.fit(pd.DataFrame(X[tr], columns=fcols), y[tr])
            preds[te] = m.predict(pd.DataFrame(X[te], columns=fcols))
        else:
            m.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
    return preds


def main():
    print("Loading data...")
    df = load_data()
    y_all = df["target_SE"].values.astype(float)
    groups = df["patient_id"].astype(str).values

    with open("../../experiments/comprehensive_results.json") as f:
        cr = json.load(f)

    # ================================================================
    # FIGURE 1: Demographics (2 panels)
    # ================================================================
    print("Figure 1: Demographics...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Age distribution
    ax = axes[0]
    ages = df["age"].dropna().astype(int)
    age_counts = ages.value_counts().sort_index()
    bars = ax.bar(age_counts.index, age_counts.values, color=TEAL, edgecolor="white",
                  linewidth=0.8, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 3, str(int(h)),
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=SLATE)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Eyes")
    ax.set_xticks(range(int(ages.min()), int(ages.max()) + 1))
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.97, 0.95, f"n = {len(ages)}\nMean = {ages.mean():.1f} ± {ages.std():.1f} yrs",
            transform=ax.transAxes, va="top", ha="right", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=LGRAY, alpha=0.95))
    panel_label(ax, "(a)")

    # (b) SE distribution with colored bins
    ax = axes[1]
    se = df["target_SE"].dropna()
    bins_edges = np.arange(se.min() - 0.25, se.max() + 0.75, 0.5)

    # Color bins by refractive category
    n_hyp = (se >= 0).sum()
    n_mild = ((se < 0) & (se >= -3)).sum()
    n_mod = ((se < -3) & (se >= -6)).sum()
    n_high = (se < -6).sum()

    # Plot as colored segments
    for label, mask, color in [
        ("High myopia", se < -6, PURPLE),
        ("Moderate myopia", (se >= -6) & (se < -3), GOLD),
        ("Mild myopia", (se >= -3) & (se < 0), SAGE),
        ("Emmetropic/Hyperopic", se >= 0, CORAL),
    ]:
        subset = se[mask]
        if len(subset) > 0:
            ax.hist(subset, bins=bins_edges, color=color, edgecolor="white",
                    linewidth=0.5, alpha=0.8, label=f"{label} (n={len(subset)})")

    for thresh, ls in [(-6, ":"), (-3, "--"), (0, "-")]:
        ax.axvline(thresh, color=SLATE, linestyle=ls, linewidth=1.2, alpha=0.6)

    ax.set_xlabel("Cycloplegic Spherical Equivalent (D)")
    ax.set_ylabel("Number of Eyes")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95, edgecolor=LGRAY,
              bbox_to_anchor=(0.0, 0.88))  # shift legend down to make room for (b)
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.97, 0.55, f"Mean = {se.mean():.2f} ± {se.std():.2f} D",
            transform=ax.transAxes, va="top", ha="right", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=LGRAY, alpha=0.95))
    panel_label(ax, "(b)", x=0.03, y=0.96)  # top-left, matching (a)

    plt.tight_layout()
    plt.savefig(f"{OUT}fig_demographics.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT}fig_demographics.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_demographics.png + .pdf")

    # ================================================================
    # FIGURE: Dataset overview (device measurements + missing data)
    # ================================================================
    print("Figure: Dataset overview...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Box plot: device SE vs cycloplegic SE
    ax = axes[0]
    box_data = []
    box_labels = []
    box_colors_list = []
    for label, col, color in [
        ("Cycloplegic\n(gold std)", "target_SE", SLATE),
        ("AR Pre", "auto_pre_SE_raw", TEAL),
        ("AR Post", "auto_post_SE_raw", TEAL),
        ("ER Pre", "eyerobo_pre_SE_raw", CORAL),
        ("ER Post", "eyerobo_post_SE_raw", CORAL),
    ]:
        vals = df[col].dropna().values
        box_data.append(vals)
        box_labels.append(label)
        box_colors_list.append(color)

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.5, color=SLATE),
                    capprops=dict(linewidth=1.5, color=SLATE),
                    flierprops=dict(marker=".", markersize=3, alpha=0.3, color=GRAY))
    for patch, color in zip(bp["boxes"], box_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)
        patch.set_edgecolor(SLATE)

    ax.set_ylabel("Spherical Equivalent (D)")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", labelsize=11)
    panel_label(ax, "(a)")

    # (b) Missing data rates
    ax = axes[1]
    missing_groups = {
        "AR Pre SPH/CYL/AX": ["auto_pre_SPH", "auto_pre_CYL", "auto_pre_AX"],
        "AR Post SPH/CYL/AX": ["auto_post_SPH", "auto_post_CYL", "auto_post_AX"],
        "ER Pre SPH/CYL/AX": ["eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_pre_AX"],
        "ER Post SPH/CYL/AX": ["eyerobo_post_SPH", "eyerobo_post_CYL", "eyerobo_post_AX"],
        "IOL Pre (biometry)": IOL_BIO,
        "Demographics": DEMO,
    }
    labels_m = list(missing_groups.keys())
    rates = [np.mean([df[c].isna().mean() * 100 for c in cols]) for cols in missing_groups.values()]
    colors_m = [TEAL, TEAL, CORAL, CORAL, GOLD, SAGE]
    y_pos = np.arange(len(labels_m))

    bars = ax.barh(y_pos, rates, color=colors_m, edgecolor="white", linewidth=0.8, height=0.55, alpha=0.85)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{rate:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold", color=SLATE)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_m, fontsize=10)
    ax.set_xlabel("Missing Rate (%)")
    ax.set_xlim(0, max(rates) * 1.4)
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    panel_label(ax, "(b)", x=0.03, y=0.96)  # top-left, in line with (a)

    plt.tight_layout()
    plt.savefig(f"{OUT}fig_dataset_overview.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT}fig_dataset_overview.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_dataset_overview.png + .pdf")

    # ================================================================
    # FIGURE 2: Raw Bland-Altman (4 panels)
    # ================================================================
    print("Figure 2: Raw Bland-Altman...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    configs_raw = [
        ("auto_pre_SE_raw", "(a) Auto-refractor Pre"),
        ("eyerobo_pre_SE_raw", "(b) Eyerobo VS Pre"),
        ("auto_post_SE_raw", "(c) Auto-refractor Post"),
        ("eyerobo_post_SE_raw", "(d) Eyerobo VS Post"),
    ]
    for idx, (col, title) in enumerate(configs_raw):
        ax = axes[idx // 2][idx % 2]
        valid = df[col].notna() & df["target_SE"].notna()
        dv = df.loc[valid, col].values
        cv = df.loc[valid, "target_SE"].values
        means = (cv + dv) / 2
        diffs = cv - dv
        bland_altman_panel(ax, means, diffs)
        ax.set_xlabel("Mean of Cycloplegic and Device SE (D)")
        ax.set_ylabel("Cycloplegic \u2013 Device SE (D)")
        panel_label(ax, title[:3])

    plt.tight_layout(h_pad=3, w_pad=3)
    plt.savefig(f"{OUT}fig_bland_altman_raw_4panel.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT}fig_bland_altman_raw_4panel.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_bland_altman_raw_4panel.png + .pdf")

    # ================================================================
    # COMPUTE OOF PREDICTIONS (for ML figures)
    # ================================================================
    print("Computing OOF predictions for ML figures...")
    from tabpfn import TabPFNRegressor

    def make_tabpfn(s=42):
        return TabPFNRegressor.create_default_for_version("v2", device="auto", n_estimators=8)

    oof = {}
    for sc_name, fcols in SCENARIOS.items():
        X = df[fcols].values.astype(float)
        print(f"  TabPFN {sc_name}...")
        oof[sc_name] = get_oof("TabPFN", make_tabpfn, X, y_all, groups, fcols)

    # ================================================================
    # FIGURE 3: ML Bland-Altman (4 panels)
    # ================================================================
    print("Figure 3: ML Bland-Altman...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ml_configs = [
        ("S01_AR_Pre", "(a) AR Pre [TabPFN]"),
        ("S04_ER_Pre", "(b) ER Pre [TabPFN]"),
        ("S07_AR_Pre_IOL", "(c) AR Pre + IOL [TabPFN]"),
        ("S10_ER_Pre_IOL", "(d) ER Pre + IOL [TabPFN]"),
    ]
    for idx, (sc, title) in enumerate(ml_configs):
        ax = axes[idx // 2][idx % 2]
        p = oof[sc]
        valid = ~np.isnan(p)
        yv, pv = y_all[valid], p[valid]
        means = (yv + pv) / 2
        diffs = yv - pv
        bland_altman_panel(ax, means, diffs)
        panel_label(ax, title[:3])

    plt.tight_layout(h_pad=3, w_pad=3)
    plt.savefig(f"{OUT}fig_bland_altman_ml_4panel.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT}fig_bland_altman_ml_4panel.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_bland_altman_ml_4panel.png + .pdf")

    # ================================================================
    # FIGURE 4: Scatter plots (4 panels)
    # ================================================================
    print("Figure 4: Scatter plots...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    scatter_configs = [
        ("S01_AR_Pre", "(a) AR Pre"),
        ("S04_ER_Pre", "(b) ER Pre"),
        ("S07_AR_Pre_IOL", "(c) AR Pre + IOL"),
        ("S10_ER_Pre_IOL", "(d) ER Pre + IOL"),
    ]
    for idx, (sc, title) in enumerate(scatter_configs):
        ax = axes[idx // 2][idx % 2]
        p = oof[sc]
        valid = ~np.isnan(p)
        yv, pv = y_all[valid], p[valid]
        mae = np.mean(np.abs(yv - pv))
        r2 = 1 - np.sum((yv-pv)**2) / np.sum((yv-yv.mean())**2)
        r_p = pearsonr(yv, pv)[0]
        w50 = np.mean(np.abs(yv-pv) <= 0.50) * 100

        ax.scatter(yv, pv, alpha=0.35, s=16, color=SLATE, edgecolors="none", zorder=2)
        lims = [min(yv.min(), pv.min()) - 0.5, max(yv.max(), pv.max()) + 0.5]
        ax.plot(lims, lims, color=CORAL, linestyle="--", linewidth=2.0, alpha=0.8, zorder=3)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Measured Cycloplegic SE (D)")
        ax.set_ylabel("Predicted Cycloplegic SE (D)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

        txt = f"MAE = {mae:.3f} D\nR\u00b2 = {r2:.3f}\nr = {r_p:.3f}\n\u22640.50 D: {w50:.1f}%"
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, va="bottom", ha="right",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=SLATE, alpha=0.95, linewidth=1.2))
        panel_label(ax, title[:3])

    plt.tight_layout(h_pad=3, w_pad=3)
    plt.savefig(f"{OUT}fig_scatter_4panel.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT}fig_scatter_4panel.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_scatter_4panel.png + .pdf")

    # ================================================================
    # FIGURE 5: Learning curve
    # ================================================================
    print("Figure 5: Learning curve...")
    lc = cr["learning_curve"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_offsets = {"TabPFN": -15, "XGBoost": 12, "RF": 10, "Ridge": -12}

    for mn in ["TabPFN", "XGBoost", "RF", "Ridge"]:
        r = lc[mn]
        n = np.array(r["n_train"])
        m_arr = np.array(r["mae_mean"])
        s_arr = np.array(r["mae_std"])
        ax.plot(n, m_arr, marker=MODEL_MARKERS[mn], color=MODEL_COLORS[mn],
                label=mn, linewidth=2.5, markersize=9)
        ax.fill_between(n, m_arr - s_arr, m_arr + s_arr, alpha=0.12, color=MODEL_COLORS[mn])
        ax.annotate(f"{m_arr[-1]:.3f}", xy=(n[-1], m_arr[-1]),
                    xytext=(12, y_offsets[mn]), textcoords="offset points",
                    fontsize=11, color=MODEL_COLORS[mn], fontweight="bold")

    ax.axhline(0.50, color=LGRAY, linestyle="--", linewidth=1.2)
    ax.text(n[0], 0.51, "0.50 D clinical threshold", fontsize=10, color=GRAY, va="bottom")
    ax.set_xlabel("Approximate Training Set Size (eyes)")
    ax.set_ylabel("Mean Absolute Error (D)")
    ax.legend(fontsize=12, loc="upper right", framealpha=0.95, edgecolor=LGRAY)
    ax.set_ylim(0.45, 1.15)
    ax.grid(alpha=0.2)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    plt.tight_layout()
    plt.savefig(f"{OUT}fig_learning_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT}fig_learning_curve.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_learning_curve.png + .pdf")

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
