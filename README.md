# Explainable Medication Adherence Risk Detection System

An AI-driven clinical decision support system that predicts medication non-adherence risk by fusing **pharmacy refill patterns** with **longitudinal clinical biomarker trends** (e.g., HbA1c, Blood Pressure, LDL) and explains every prediction in **plain English** using SHAP (SHapley Additive exPlanations).

---

## 🎯 The Core Problem & The Two Blind Spots
Traditional healthcare and pharmacy systems measure adherence solely using prescription pickup logs (such as PDC – *Proportion of Days Covered* or refill gaps). This creates two dangerous clinical blind spots:

| Blind Spot | Refill-Only System Flaw | Real Patient Reality | Our Multimodal AI Solution |
| :--- | :--- | :--- | :--- |
| **1. False Alarms** *(Out-of-Network Refills)* | Patient looks non-adherent because local pharmacy records show low fills / large gaps ($PDC < 50\%$). | Patient fills their prescriptions at a 90-day mail order, VA clinic, or secondary retail pharmacy. 


