# Explainable Medication Adherence Risk Detection System

An AI-driven clinical decision support system that predicts medication non-adherence risk by fusing **pharmacy refill patterns** with **longitudinal clinical biomarker trends** (e.g., HbA1c, Blood Pressure, LDL) and explains every prediction in **plain English** using SHAP (SHapley Additive exPlanations).

---

## 🎯 The Core Problem & The Two Blind Spots
Traditional healthcare and pharmacy systems measure adherence solely using prescription pickup logs (such as PDC – *Proportion of Days Covered* or refill gaps). This creates two dangerous clinical blind spots:

| Blind Spot | Refill-Only System Flaw | Real Patient Reality | Our Multimodal AI Solution |
| :--- | :--- | :--- | :--- |
| **1. False Alarms** *(Out-of-Network Refills)* | Patient looks non-adherent because local pharmacy records show low fills / large gaps ($PDC < 50\%$). | Patient fills their prescriptions at a 90-day mail order, VA clinic, or secondary retail pharmacy. Their HbA1c ($6.1\%$) and BP ($118/76$) are perfectly controlled. | Detects the discordance (stable clinical outcomes despite low local fills) and **suppresses alert fatigue** by flagging probable external sourcing. |
| **2. Missed Cases** *(Silent Non-Adherence)* | Patient looks perfectly adherent because they pick up their pills every 30 days ($PDC > 90\%$). | Patient hoards pills, fears side effects, or splits pills. Their HbA1c surged from $7.2\% \to 9.8\%$ and BP rose by $+20\text{ mmHg}$. | Catches this **"Refilled but Not Taken"** discordance immediately and triggers a critical pharmacist tolerability audit. |

---

## 🚀 1-Week Project Roadmap & Implementation Guide

Here is your exact day-by-day plan to build, refine, validate, and present this project within one week:

```
Week Roadmap:
Day 1: Problem Definition, Architecture & Data Stream Formulation
Day 2: Multimodal Feature Engineering & Dual-Stream Data Pipeline
Day 3: Model Development & Baseline Benchmark Comparison
Day 4: SHAP Attribution & Plain-English Narrative Engine
Day 5: Clinical Action Recommendation System & Decision Support
Day 6: Interactive Dashboard UI & What-If Scenario Sandbox
Day 7: Validation, Testing, Metrics Packaging & Final Presentation
```

### Day 1: Problem Formulation & Architecture
- Define the two data streams:
  1. **Pharmacy Refill Stream**: PDC rate, Refill Gap (days past schedule), Regimen Complexity (med count, daily doses), Monthly Copay.
  2. **Clinical EHR / Biomarker Stream**: Baseline & Follow-up HbA1c, Systolic/Diastolic Blood Pressure, LDL Cholesterol, and $\Delta \text{Biomarker Velocity}$.
- Define the 4 Clinical Archetypes: *True Adherent*, *True Non-Adherent*, *Silent Non-Adherence*, and *Out-of-Network Refills*.

### Day 2: Multimodal Feature Engineering
- Build `src/data_generator.py` to model realistic chronic disease patient cohorts (Diabetes, Hypertension, Polypharmacy).
- Formulate discordance features:
  $$\text{Discordance}_{\text{Silent}} = \mathbb{I}(PDC \ge 0.80 \land \Delta\text{HbA1c} > 0.6\%)$$
  $$\text{Discordance}_{\text{Out-of-Network}} = \mathbb{I}(PDC < 0.60 \land \Delta\text{HbA1c} \le 0.0\%)$$

### Day 3: Model Development & Benchmarking
- Build `src/model_engine.py`.
- Train **Baseline Model** (Refill-Only Logistic Regression) vs. **Multimodal Model** (Gradient Boosting / Random Forest).
- Demonstrate statistical superiority on ROC-AUC, False Alarm reduction rate, and Silent Non-Adherence catch rate.

### Day 4: SHAP Explainability & Plain-English Synthesis
- Build `src/explainer.py`.
- Compute individual TreeSHAP feature contributions.
- Translate quantitative SHAP values into clinician-friendly prose (Headline, Clinical Evidence, and Primary Risk Drivers).

### Day 5: Clinical Action Recommendation Engine
- Build `src/recommendation_engine.py`.
- Map risk tiers and SHAP root causes (e.g., cost barrier vs. polypharmacy vs. adverse tolerability) into targeted care team actions, execution checklists, and patient dialogue scripts.

### Day 6: Modern Interactive Dashboard UI
- Build `app.py` using Streamlit and Plotly.
- Implement 5 core modules: Executive Benchmark, Patient Risk Triage Cohort, Patient Deep-Dive Dossier, What-If Simulation Sandbox, and Data Export/Upload.

### Day 7: Testing, Documentation & Project Presentation
- Run test suite `tests/test_system.py`.
- Prepare demo talking points, slides, and highlight the dual blind-spot innovation.

---

## 🛠️ Project Structure

```
medication-adherence-risk-system/
├── app.py                      # Main Streamlit Interactive Dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Complete 1-Week Guide & Documentation
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Dual-stream synthetic EHR/Pharmacy data generator
│   ├── model_engine.py         # Baseline & Multimodal ML models + benchmark metrics
│   ├── explainer.py            # SHAP attribution & Plain-English narrative engine
│   └── recommendation_engine.py# Evidence-based clinical action generator
└── tests/
    └── test_system.py          # Automated verification test suite
```

---

## ⚡ Quickstart & How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
*The web dashboard will open in your default browser at `http://localhost:8501`.*

### 3. Run the Automated Tests
```bash
python -m unittest discover -s tests
```

---

## 🌟 Key Features of the Dashboard

1. **📊 Executive Benchmark & Impact**:
   - Compares the Explainable Multimodal AI against standard Refill-Only systems.
   - Highlights **~75% False Alarm Reduction** and **~95% Silent Non-Adherence Detection**.
   - Interactive 4-quadrant plot mapping $PDC$ vs. $\Delta\text{HbA1c}$.

2. **📋 Patient Risk Triage Cohort**:
   - Filterable, searchable registry of patients prioritized by risk score and clinical urgency.
   - Fast filtering by condition, risk tier, and score thresholds.

3. **🔍 Patient Deep-Dive Dossier**:
   - **Dual-Stream Timeline**: Monthly refill bar chart juxtaposed with HbA1c/BP progression curves.
   - **SHAP Waterfall / Bar Plot**: Quantifies exact positive and negative feature contributions.
   - **Plain-English Clinical Narrative**: Readable summary for doctors and pharmacists.
   - **Action Plan & Checklist**: Step-by-step care protocol with call scripts.

4. **🧪 Interactive "What-If" Adherence Simulator**:
   - Clinicians can adjust refill gaps, monthly copays, medication counts, and biomarker values to see risk scores, SHAP explanations, and care recommendations update in real time.

5. **📁 Data Manager & Export**:
   - One-click CSV export of prioritized patient intervention rosters.
