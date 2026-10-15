"""
Machine Learning Model Engine
Implements and benchmarks:
1. Standard Refill-Only Baseline Model
2. Multimodal Explainable Adherence Risk Model (Refills + Clinical Biomarker Trajectories)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


REFILL_ONLY_FEATURES = [
    "pdc_rate",
    "refill_gap_days",
    "on_time_refill_ratio",
    "medication_count",
    "age"
]

MULTIMODAL_FEATURES = [
    "pdc_rate",
    "refill_gap_days",
    "on_time_refill_ratio",
    "medication_count",
    "daily_dose_frequency",
    "copay_monthly_usd",
    "age",
    "baseline_hba1c",
    "followup_hba1c",
    "delta_hba1c",
    "baseline_sbp",
    "followup_sbp",
    "delta_sbp",
    "delta_ldl",
    "refill_biomarker_discordance",
    "refill_gap_clinical_discordance"
]


class AdherenceModelEngine:
    """
    Dual model engine training both baseline refill-only and advanced multimodal models.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.baseline_model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.multimodal_model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        )
        self.is_trained = False
        self.feature_names = MULTIMODAL_FEATURES
        self.metrics: Dict[str, Any] = {}

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains both models and computes comparative benchmarks.
        """
        X_refill = df[REFILL_ONLY_FEATURES]
        X_multi = df[MULTIMODAL_FEATURES]
        y = df["actual_non_adherence"]

        X_train_ref, X_test_ref, X_train_mul, X_test_mul, y_train, y_test, idx_train, idx_test = (
            train_test_split(
                X_refill, X_multi, y, df.index, test_size=0.25, random_state=self.random_state, stratify=y
            )
        )

        # Train Baseline Refill-Only Model
        self.baseline_model.fit(X_train_ref, y_train)
        pred_ref_prob = self.baseline_model.predict_proba(X_test_ref)[:, 1]
        pred_ref = (pred_ref_prob >= 0.50).astype(int)

        # Train Multimodal Model
        self.multimodal_model.fit(X_train_mul, y_train)
        pred_mul_prob = self.multimodal_model.predict_proba(X_test_mul)[:, 1]
        pred_mul = (pred_mul_prob >= 0.50).astype(int)

        test_df = df.loc[idx_test].copy()
        test_df["baseline_pred_prob"] = pred_ref_prob
        test_df["baseline_pred"] = pred_ref
        test_df["multimodal_pred_prob"] = pred_mul_prob
        test_df["multimodal_pred"] = pred_mul

        # Evaluate Blind Spot 1: False Alarms on Out-of-Network Patients
        out_of_network_sub = test_df[test_df["clinical_archetype"] == "OutOfNetwork_Refill"]
        oon_baseline_false_alarms = (out_of_network_sub["baseline_pred"] == 1).sum()
        oon_multi_false_alarms = (out_of_network_sub["multimodal_pred"] == 1).sum()
        oon_total = max(1, len(out_of_network_sub))
        oon_false_alarm_reduction_pct = (
            (oon_baseline_false_alarms - oon_multi_false_alarms) / max(1, oon_baseline_false_alarms) * 100
        ) if oon_baseline_false_alarms > 0 else 0.0

        # Evaluate Blind Spot 2: Missed Cases on Silent Non-Adherence ("Refilled but not taken")
        silent_sub = test_df[test_df["clinical_archetype"] == "Silent_NonAdherence"]
        silent_baseline_caught = (silent_sub["baseline_pred"] == 1).sum()
        silent_multi_caught = (silent_sub["multimodal_pred"] == 1).sum()
        silent_total = max(1, len(silent_sub))
        silent_catch_rate_baseline = (silent_baseline_caught / silent_total) * 100
        silent_catch_rate_multi = (silent_multi_caught / silent_total) * 100

        self.metrics = {
            "baseline": {
                "roc_auc": round(roc_auc_score(y_test, pred_ref_prob), 4),
                "accuracy": round(accuracy_score(y_test, pred_ref), 4),
                "precision": round(precision_score(y_test, pred_ref, zero_division=0), 4),
                "recall": round(recall_score(y_test, pred_ref, zero_division=0), 4),
                "f1": round(f1_score(y_test, pred_ref, zero_division=0), 4),
                "false_alarms_count": int(oon_baseline_false_alarms),
                "false_alarm_rate_oon": round((oon_baseline_false_alarms / oon_total) * 100, 1),
                "silent_non_adherence_caught": int(silent_baseline_caught),
                "silent_catch_rate_pct": round(silent_catch_rate_baseline, 1),
                "silent_total": int(silent_total)
            },
            "multimodal": {
                "roc_auc": round(roc_auc_score(y_test, pred_mul_prob), 4),
                "accuracy": round(accuracy_score(y_test, pred_mul), 4),
                "precision": round(precision_score(y_test, pred_mul, zero_division=0), 4),
                "recall": round(recall_score(y_test, pred_mul, zero_division=0), 4),
                "f1": round(f1_score(y_test, pred_mul, zero_division=0), 4),
                "false_alarms_count": int(oon_multi_false_alarms),
                "false_alarm_rate_oon": round((oon_multi_false_alarms / oon_total) * 100, 1),
                "silent_non_adherence_caught": int(silent_multi_caught),
                "silent_catch_rate_pct": round(silent_catch_rate_multi, 1),
                "silent_total": int(silent_total)
            },
            "comparisons": {
                "false_alarm_reduction_pct": round(oon_false_alarm_reduction_pct, 1),
                "silent_catch_rate_gain_pct": round(silent_catch_rate_multi - silent_catch_rate_baseline, 1),
                "roc_auc_gain": round(roc_auc_score(y_test, pred_mul_prob) - roc_auc_score(y_test, pred_ref_prob), 4)
            }
        }
        self.is_trained = True
        return self.metrics

    def predict_cohort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores an entire patient cohort with risk probabilities, risk tiers, and clinical archetypes.
        """
        X_multi = df[MULTIMODAL_FEATURES]
        X_ref = df[REFILL_ONLY_FEATURES]

        multi_probs = self.multimodal_model.predict_proba(X_multi)[:, 1]
        ref_probs = self.baseline_model.predict_proba(X_ref)[:, 1]

        scored_df = df.copy()
        scored_df["adherence_risk_score"] = [round(p * 100, 1) for p in multi_probs]
        scored_df["baseline_refill_risk_score"] = [round(p * 100, 1) for p in ref_probs]

        # Clinical Risk Tier Assignment
        risk_tiers = []
        archetype_labels = []

        for i, row in scored_df.iterrows():
            score = row["adherence_risk_score"]
            pdc = row["pdc_rate"]
            d_hba1c = row["delta_hba1c"]
            d_sbp = row["delta_sbp"]

            # Check for Missed Cases vs False Alarms
            if pdc >= 0.80 and (d_hba1c >= 0.7 or d_sbp >= 12.0):
                tier = "⚠️ Refilled But NOT Taken (Missed Case)"
                archetype = "Refilled but not taken"
            elif pdc < 0.60 and d_hba1c <= 0.0 and d_sbp <= 0.0:
                tier = "ℹ️ Different Pharmacy (False Alarm)"
                archetype = "Refilled at different pharmacy"
            elif score >= 70:
                tier = "🚨 High Risk (Skipping Medicine)"
                archetype = "Skipping refills and health worsening"
            elif score >= 40:
                tier = "⚠️ Moderate Risk (Late Refills)"
                archetype = "Late refill warnings"
            else:
                tier = "✅ On Track (Taking Medicine Regularly)"
                archetype = "Taking medicine regularly"

            risk_tiers.append(tier)
            archetype_labels.append(archetype)

        scored_df["risk_tier"] = risk_tiers
        scored_df["detected_archetype"] = archetype_labels
        return scored_df


if __name__ == "__main__":
    from src.data_generator import generate_patient_cohort
    df = generate_patient_cohort(n_patients=1000)
    engine = AdherenceModelEngine()
    metrics = engine.train_and_evaluate(df)
    print("Baseline Metrics:", metrics["baseline"])
    print("Multimodal Metrics:", metrics["multimodal"])
    print("Comparative Gains:", metrics["comparisons"])
