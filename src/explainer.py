"""
Plain-English Explainer Engine
Converts mathematical SHAP values into simple, easy-to-understand explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


SIMPLE_FEATURE_NAMES = {
    "pdc_rate": "On-Time Refill Rate (%)",
    "refill_gap_days": "Days Late for Refill",
    "on_time_refill_ratio": "Refill Pickup Consistency",
    "medication_count": "Number of Medicines Taken Daily",
    "daily_dose_frequency": "Pills Taken Per Day",
    "copay_monthly_usd": "Monthly Medicine Cost (₹)",
    "age": "Patient Age",
    "baseline_hba1c": "Previous Blood Sugar (HbA1c)",
    "followup_hba1c": "Current Blood Sugar (HbA1c)",
    "delta_hba1c": "Blood Sugar Change (HbA1c Increase)",
    "baseline_sbp": "Previous Blood Pressure (Systolic)",
    "followup_sbp": "Current Blood Pressure (Systolic)",
    "delta_sbp": "Blood Pressure Increase",
    "delta_ldl": "Cholesterol Increase",
    "refill_biomarker_discordance": "Health Worsened Despite Picking Up Medicine",
    "refill_gap_clinical_discordance": "Health is Normal Despite Late Refills"
}


class ClinicalShapExplainer:
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or list(SIMPLE_FEATURE_NAMES.keys())
        self.tree_explainer = None
        self._init_shap()

    def _init_shap(self):
        try:
            import shap
            self.tree_explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.tree_explainer = None

    def explain_patient(self, patient_row: pd.Series) -> Dict[str, Any]:
        feature_vals = patient_row[self.feature_names].values.reshape(1, -1)
        if self.tree_explainer is not None:
            try:
                shap_res = self.tree_explainer.shap_values(feature_vals)
                if isinstance(shap_res, list) and len(shap_res) == 2:
                    raw_shap = shap_res[1][0]
                elif isinstance(shap_res, np.ndarray) and len(shap_res.shape) == 3:
                    raw_shap = shap_res[0, :, 1]
                else:
                    raw_shap = np.array(shap_res).flatten()
            except Exception:
                raw_shap = self._fallback_impacts(patient_row)
        else:
            raw_shap = self._fallback_impacts(patient_row)

        contributions = []
        for feat, val, sh in zip(self.feature_names, feature_vals[0], raw_shap):
            contributions.append({
                "feature": feat,
                "display_name": SIMPLE_FEATURE_NAMES.get(feat, feat),
                "value": val,
                "shap_impact": round(float(sh), 4),
                "impact_direction": "Increases Risk" if sh > 0 else "Lowers Risk / Good Sign"
            })

        sorted_contribs = sorted(contributions, key=lambda x: abs(x["shap_impact"]), reverse=True)
        top_risk_drivers = [c for c in sorted_contribs if c["shap_impact"] > 0][:3]
        top_protective = [c for c in sorted_contribs if c["shap_impact"] < 0][:2]

        narrative = self._generate_simple_narrative(patient_row, top_risk_drivers)

        return {
            "all_contributions": sorted_contribs,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective,
            "plain_english_narrative": narrative
        }

    def _fallback_impacts(self, row: pd.Series) -> np.ndarray:
        importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)) / len(self.feature_names))
        impacts = []
        for i, feat in enumerate(self.feature_names):
            val, imp = float(row[feat]), importances[i]
            if feat == "pdc_rate": impact = -imp * (val - 0.75) * 4.0
            elif feat == "refill_gap_days": impact = imp * (val - 10.0) / 15.0
            elif feat == "delta_hba1c": impact = imp * (val - 0.0) * 3.5
            elif feat == "delta_sbp": impact = imp * (val - 0.0) / 5.0
            elif feat == "refill_biomarker_discordance": impact = imp * (2.5 if val == 1 else -0.5)
            elif feat == "refill_gap_clinical_discordance": impact = -imp * (2.0 if val == 1 else 0.0)
            elif feat == "medication_count": impact = imp * (val - 4.0) / 3.0
            elif feat == "copay_monthly_usd": impact = imp * (val - 35.0) / 25.0
            else: impact = imp * 0.1
            impacts.append(impact)
        return np.array(impacts)

    def _generate_simple_narrative(self, row: pd.Series, top_risk_drivers: List[Dict[str, Any]]) -> Dict[str, str]:
        risk_tier = row.get("risk_tier", "Moderate Risk")
        pdc = row.get("pdc_rate", 0.8) * 100
        gap = int(row.get("refill_gap_days", 0))
        d_hba1c = row.get("delta_hba1c", 0.0)
        d_sbp = row.get("delta_sbp", 0.0)
        base_hba1c, fup_hba1c = row.get("baseline_hba1c", 7.0), row.get("followup_hba1c", 7.0)
        base_sbp, fup_sbp = row.get("baseline_sbp", 130), row.get("followup_sbp", 130)
        med_cnt = row.get("medication_count", 3)
        copay = row.get("copay_monthly_usd", 25.0)

        # 1. Main simple headline
        if "Refilled But NOT Taken" in risk_tier or "Silent" in risk_tier:
            headline = "⚠️ MISSED CASE CAUGHT: Patient picks up medicines on time, but is NOT taking them."
            reason = (
                f"Even though this patient refilled on time ({pdc:.0f}% refill rate), their blood sugar went up "
                f"from {base_hba1c}% to {fup_hba1c}% (+{d_hba1c:.1f}%) and blood pressure increased by +{d_sbp:.0f} mmHg. "
                "Standard pharmacy systems miss this because they only check pickup dates."
            )
        elif "False Alarm" in risk_tier or "Out-of-Network" in risk_tier:
            headline = "ℹ️ FALSE ALARM AVOIDED: Patient refilled at a different pharmacy."
            reason = (
                f"This pharmacy record shows a late refill ({gap} days late), but the patient's blood sugar and blood pressure "
                "are completely normal and healthy. The patient likely got their medicine from mail-order or another clinic."
            )
        elif row.get("adherence_risk_score", 0) >= 70:
            headline = "🚨 HIGH RISK: Patient is skipping medicine and health is worsening."
            reason = (
                f"The patient is {gap} days late picking up their prescription, and their health numbers have worsened "
                f"(Blood sugar: {base_hba1c}% → {fup_hba1c}%, Blood pressure: {base_sbp:.0f} → {fup_sbp:.0f} mmHg)."
            )
        else:
            headline = "✅ ON TRACK: Patient is taking medicine regularly and health is stable."
            reason = f"The patient picks up refills on time ({pdc:.0f}% on-time rate) and health numbers are well controlled."

        # Key simple reasons
        reasons_list = []
        if gap > 14:
            reasons_list.append(f"• **Refill Delay:** {gap} days late picking up medicine.")
        if d_hba1c >= 0.5:
            reasons_list.append(f"• **Blood Sugar Jump:** Rose by +{d_hba1c:.1f}% (from {base_hba1c}% to {fup_hba1c}%).")
        if d_sbp >= 10:
            reasons_list.append(f"• **Blood Pressure Jump:** Rose by +{d_sbp:.0f} mmHg.")
        if med_cnt >= 5:
            reasons_list.append(f"• **Too Many Medicines:** Taking {med_cnt} different medicines every day.")
        if copay > 50:
            reasons_list.append(f"• **Cost Barrier:** High monthly cost of ₹{copay * 70:.0f}.")

        summary_points = "\n".join(reasons_list) if reasons_list else "• Refills and health readings are in a healthy range."

        return {
            "headline": headline,
            "reason": reason,
            "summary_points": summary_points
        }
