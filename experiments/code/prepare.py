"""
Cycloplegic Refraction Prediction — Data Loading and Evaluation Harness

This module:
1. Loads the anonymized dataset (CSV)
2. Defines feature groups and clinical scenarios
3. Provides patient-level GroupKFold cross-validation
4. Computes evaluation metrics (MAE, RMSE, R², Bland-Altman, clinical thresholds)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/dataset.csv")
SEEDS = [42, 123, 456]
N_FOLDS = 5
TIME_BUDGET = 600  # seconds


def load_and_clean_data():
    """Load the anonymized CSV dataset with standardized column names."""
    df = pd.read_csv(DATA_PATH)

    # Convert feature columns to numeric (handle any remaining non-numeric entries)
    skip_cols = {"gender", "eye_position", "treatment", "patient_id"}
    for col in df.columns:
        if col not in skip_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {
    "eyerobo_pre": [
        "eyerobo_pre_SPH", "eyerobo_pre_CYL", "eyerobo_pre_AX", "eyerobo_pre_pupil"
    ],
    "eyerobo_post": [
        "eyerobo_post_SPH", "eyerobo_post_CYL", "eyerobo_post_AX", "eyerobo_post_pupil"
    ],
    "auto_pre": ["auto_pre_SPH", "auto_pre_CYL", "auto_pre_AX"],
    "auto_post": ["auto_post_SPH", "auto_post_CYL", "auto_post_AX"],
    "iol_pre": [
        "iol_pre_AL", "iol_pre_LT", "iol_pre_WTW", "iol_pre_CCT", "iol_pre_Pupil",
        "iol_pre_SE", "iol_pre_K1", "iol_pre_K2", "iol_pre_dK", "iol_pre_ACD",
    ],
    "iol_post": [
        "iol_post_AL", "iol_post_LT", "iol_post_WTW", "iol_post_CCT", "iol_post_Pupil",
        "iol_post_SE", "iol_post_K1", "iol_post_K2", "iol_post_dK", "iol_post_ACD",
    ],
    "demographics": ["age", "IOP", "gender_enc", "eye_enc", "treatment_enc"],
}

SCENARIOS = {
    "S1_eyerobo_pre_only": ["eyerobo_pre", "demographics"],
    "S2_auto_pre_only": ["auto_pre", "demographics"],
    "S3_iol_pre_only": ["iol_pre", "demographics"],
    "S4_all_pre": ["eyerobo_pre", "auto_pre", "iol_pre", "demographics"],
    "S5_all_pre_post": [
        "eyerobo_pre", "eyerobo_post", "auto_pre", "auto_post",
        "iol_pre", "iol_post", "demographics",
    ],
    "S6_eyerobo_post_only": ["eyerobo_post", "demographics"],
}


def get_features_for_scenario(scenario_name):
    """Return list of feature column names for a scenario."""
    groups = SCENARIOS[scenario_name]
    features = []
    for g in groups:
        features.extend(FEATURE_GROUPS[g])
    return features


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    """Compute all evaluation metrics for cycloplegic refraction prediction."""
    abs_errors = np.abs(y_true - y_pred)
    diff = y_true - y_pred
    std_diff = np.std(diff, ddof=1)
    mean_diff = np.mean(diff)

    return {
        "MAE": float(np.mean(abs_errors)),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(np.mean(diff**2)),
        "MedAE": float(np.median(abs_errors)),
        "within_0.25D": float(np.mean(abs_errors <= 0.25) * 100),
        "within_0.50D": float(np.mean(abs_errors <= 0.50) * 100),
        "within_1.00D": float(np.mean(abs_errors <= 1.00) * 100),
        "mean_diff": float(mean_diff),
        "LoA_lower": float(mean_diff - 1.96 * std_diff),
        "LoA_upper": float(mean_diff + 1.96 * std_diff),
    }


# ============================================================
# TIME BUDGET
# ============================================================

class TimeBudget:
    def __init__(self, budget_seconds=TIME_BUDGET):
        self.budget = budget_seconds
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def remaining(self):
        return max(0, self.budget - self.elapsed())


if __name__ == "__main__":
    df = load_and_clean_data()
    print(f"Loaded {len(df)} eyes from {df['patient_id'].nunique()} patients")
    print(f"Target SE: mean={df['target_SE'].mean():.3f}, std={df['target_SE'].std():.3f}")
