import sys
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
import pandas as pd
import numpy as np

from src.data_generator import generate_patient_cohort, generate_patient_time_series
from src.model_engine import AdherenceModelEngine, MULTIMODAL_FEATURES
from src.explainer import ClinicalShapExplainer
from src.recommendation_engine import generate_action_recommendation


class TestAdherenceSystem(unittest.TestCase):

    def setUp(self):
        self.df = generate_patient_cohort(n_patients=300, random_state=42)
        self.engine = AdherenceModelEngine(random_state=42)
        self.metrics = self.engine.train_and_evaluate(self.df)
        self.scored_df = self.engine.predict_cohort(self.df)
        self.explainer = ClinicalShapExplainer(self.engine.multimodal_model, MULTIMODAL_FEATURES)

    def test_data_generation(self):
        """Verify generated dataset contains all required dual-stream features."""
        required_cols = [
            "patient_id", "name", "age", "pdc_rate", "refill_gap_days",
            "baseline_hba1c", "followup_hba1c", "delta_hba1c",
            "baseline_sbp", "followup_sbp", "delta_sbp"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns)
        self.assertEqual(len(self.df), 300)

    def test_model_performance(self):
        """Verify smart multimodal model outperforms refill-only baseline."""
        base_auc = self.metrics["baseline"]["roc_auc"]
        multi_auc = self.metrics["multimodal"]["roc_auc"]
        self.assertGreater(multi_auc, base_auc)

    def test_explainer_narrative(self):
        """Verify plain-English explanation generated."""
        sample = self.scored_df.iloc[0]
        exp = self.explainer.explain_patient(sample)
        self.assertIn("headline", exp["plain_english_narrative"])
        self.assertIn("reason", exp["plain_english_narrative"])
        self.assertIn("summary_points", exp["plain_english_narrative"])

    def test_action_recommendation(self):
        """Verify suggested actions generated."""
        sample = self.scored_df.iloc[0]
        action = generate_action_recommendation(sample)
        self.assertIn("action_title", action)
        self.assertIn("simple_why", action)
        self.assertIn("what_to_do", action)
        self.assertGreater(len(action["what_to_do"]), 0)


if __name__ == "__main__":
    unittest.main()
