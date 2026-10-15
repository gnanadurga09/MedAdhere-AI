"""
Synthetic Healthcare Multimodal Dataset Generator
Simulates dual-stream healthcare data:
1. Pharmacy Refill Stream (PDC, Refill Gap, Regimen Complexity, Pharmacy Switching)
2. Clinical EHR Stream (Longitudinal Biomarkers: HbA1c, Blood Pressure, LDL Trends)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Margaret",
    "Matthew", "Lisa", "Anthony", "Betty", "Mark", "Dorothy", "Donald", "Sandra",
    "Steven", "Ashley", "Paul", "Kimberly", "Andrew", "Donna", "Joshua", "Emily",
    "Kenneth", "Carol", "Kevin", "Michelle", "Brian", "Amanda", "George", "Melissa",
    "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Laura",
    "Jeffrey", "Sharon", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Shirley", "Eric", "Angela", "Jonathan", "Helen", "Stephen", "Anna",
    "Larry", "Brenda", "Justin", "Pamela", "Scott", "Nicole", "Brandon", "Emma",
    "Benjamin", "Samantha", "Samuel", "Katherine", "Gregory", "Christine", "Alexander", "Debra",
    "Frank", "Rachel", "Patrick", "Catherine", "Raymond", "Carolyn", "Jack", "Janet",
    "Dennis", "Ruth", "Jerry", "Maria", "Tyler", "Heather", "Aaron", "Diane"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
    "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers"
]

PRIMARY_CONDITIONS = [
    "Type 2 Diabetes",
    "Hypertension",
    "Type 2 Diabetes + Hypertension",
    "Hypertension + Dyslipidemia",
    "Triple: T2D + HTN + Dyslipidemia",
    "Cardiovascular Disease + T2D"
]


def generate_patient_cohort(
    n_patients: int = 1200,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic cohort of chronic disease patients with both pharmacy
    refill metrics and longitudinal clinical biomarker trends.
    """
    np.random.seed(random_state)

    patient_ids = [f"PT-{10000 + i}" for i in range(n_patients)]
    first_names = np.random.choice(FIRST_NAMES, size=n_patients)
    last_names = np.random.choice(LAST_NAMES, size=n_patients)
    names = [f"{fn} {ln}" for fn, ln in zip(first_names, last_names)]

    ages = np.random.normal(loc=61, scale=11, size=n_patients).clip(28, 89).astype(int)
    genders = np.random.choice(["Female", "Male"], size=n_patients, p=[0.52, 0.48])
    conditions = np.random.choice(
        PRIMARY_CONDITIONS,
        size=n_patients,
        p=[0.22, 0.25, 0.23, 0.12, 0.12, 0.06]
    )

    # Regimen complexity & medication count
    med_counts = []
    daily_doses = []
    copay_monthly = []

    for cond, age in zip(conditions, ages):
        base_meds = 2 if "Triple" in cond else (4 if "Cardiovascular" in cond else (3 if "+" in cond else 2))
        age_bonus = 1 if age > 65 else 0
        med_cnt = int(np.random.poisson(base_meds + age_bonus)) + 1
        med_cnt = max(1, min(12, med_cnt))
        med_counts.append(med_cnt)

        # Regimen complexity: doses per day
        dose = med_cnt * np.random.choice([1.0, 1.5, 2.0, 2.5], p=[0.35, 0.35, 0.20, 0.10])
        daily_doses.append(round(dose, 1))

        # Out of pocket cost
        copay = np.random.exponential(scale=28) + (med_cnt * 6)
        copay_monthly.append(round(copay, 2))

    # Four Clinical Archetypes distribution in patient population:
    # 1. True Adherent (~50%)
    # 2. True Non-Adherent (~24%)
    # 3. Silent Non-Adherence ("Refilled but not taken") (~14%) -> Missed by standard systems
    # 4. Out-of-Network Refill ("False Alarm") (~12%) -> Erroneously flagged by standard systems
    archetype_probs = [0.50, 0.24, 0.14, 0.12]
    archetypes = np.random.choice(
        ["True_Adherent", "True_NonAdherent", "Silent_NonAdherence", "OutOfNetwork_Refill"],
        size=n_patients,
        p=archetype_probs
    )

    # Pharmacy refill metrics & Biomarker generation based on archetype
    pdc_list = []
    refill_gap_list = []
    refills_on_time_ratio_list = []
    secondary_pharmacy_flags = []
    
    # Biomarkers: Baseline vs Follow-up (6 months)
    baseline_hba1c_list = []
    followup_hba1c_list = []
    baseline_sbp_list = []
    followup_sbp_list = []
    baseline_dbp_list = []
    followup_dbp_list = []
    baseline_ldl_list = []
    followup_ldl_list = []

    # Ground truth actual adherence indicator (0 = adherent, 1 = true non-adherent risk)
    actual_non_adherent_risk = []

    for i in range(n_patients):
        arch = archetypes[i]
        cond = conditions[i]
        has_t2d = "Diabetes" in cond or "T2D" in cond
        has_htn = "Hypertension" in cond or "HTN" in cond

        # Baseline clinical values
        base_hba1c = np.random.normal(7.6, 0.7) if has_t2d else np.random.normal(5.5, 0.3)
        base_sbp = np.random.normal(142, 12) if has_htn else np.random.normal(124, 8)
        base_dbp = np.random.normal(88, 7) if has_htn else np.random.normal(76, 6)
        base_ldl = np.random.normal(128, 22)

        if arch == "True_Adherent":
            # High PDC, small refill gap, stable/improved biomarkers
            pdc = np.random.uniform(0.82, 0.99)
            refill_gap = int(np.random.exponential(scale=4))  # 0-10 days
            ontime_ratio = np.random.uniform(0.85, 1.0)
            sec_pharm = np.random.choice([0, 1], p=[0.90, 0.10])

            delta_hba1c = np.random.normal(-0.5, 0.3) if has_t2d else np.random.normal(0.0, 0.1)
            delta_sbp = np.random.normal(-12, 5) if has_htn else np.random.normal(-2, 3)
            delta_dbp = np.random.normal(-7, 3) if has_htn else np.random.normal(-1, 2)
            delta_ldl = np.random.normal(-18, 8)
            
            ground_truth = 0  # Adherent

        elif arch == "True_NonAdherent":
            # Low PDC, large refill gaps, worsening biomarkers
            pdc = np.random.uniform(0.30, 0.72)
            refill_gap = int(np.random.uniform(18, 75))
            ontime_ratio = np.random.uniform(0.20, 0.65)
            sec_pharm = np.random.choice([0, 1], p=[0.85, 0.15])

            delta_hba1c = np.random.normal(+0.9, 0.4) if has_t2d else np.random.normal(+0.1, 0.1)
            delta_sbp = np.random.normal(+14, 6) if has_htn else np.random.normal(+3, 3)
            delta_dbp = np.random.normal(+8, 4) if has_htn else np.random.normal(+2, 2)
            delta_ldl = np.random.normal(+22, 10)

            ground_truth = 1  # Non-Adherent

        elif arch == "Silent_NonAdherence":
            # "Refilled but not taken": High PDC, excellent pharmacy record, but severe clinical worsening!
            pdc = np.random.uniform(0.85, 0.98)
            refill_gap = int(np.random.exponential(scale=3))  # looks great in pharmacy!
            ontime_ratio = np.random.uniform(0.88, 1.0)
            sec_pharm = 0

            # Deteriorating clinical outcomes despite picking up pills
            delta_hba1c = np.random.normal(+1.4, 0.4) if has_t2d else np.random.normal(+0.2, 0.1)
            delta_sbp = np.random.normal(+20, 7) if has_htn else np.random.normal(+5, 3)
            delta_dbp = np.random.normal(+11, 4) if has_htn else np.random.normal(+3, 2)
            delta_ldl = np.random.normal(+30, 12)

            ground_truth = 1  # Actually Non-Adherent (Patient is picking up but not swallowing)

        else:  # "OutOfNetwork_Refill"
            # "False Alarm": Low PDC in our pharmacy system, but clinical outcomes are optimal and well-controlled!
            pdc = np.random.uniform(0.25, 0.55)
            refill_gap = int(np.random.uniform(22, 60))
            ontime_ratio = np.random.uniform(0.15, 0.50)
            sec_pharm = 1  # Uses mail order, VA, secondary clinic

            # Very well-controlled biomarkers (actually taking meds obtained elsewhere)
            delta_hba1c = np.random.normal(-0.6, 0.2) if has_t2d else np.random.normal(-0.1, 0.1)
            delta_sbp = np.random.normal(-14, 4) if has_htn else np.random.normal(-2, 2)
            delta_dbp = np.random.normal(-8, 3) if has_htn else np.random.normal(-1, 2)
            delta_ldl = np.random.normal(-24, 7)

            ground_truth = 0  # Adherent (False alarm in refill-only system)

        # Follow-up values
        fup_hba1c = max(4.5, base_hba1c + delta_hba1c)
        fup_sbp = max(90, base_sbp + delta_sbp)
        fup_dbp = max(55, base_dbp + delta_dbp)
        fup_ldl = max(40, base_ldl + delta_ldl)

        pdc_list.append(round(pdc, 3))
        refill_gap_list.append(refill_gap)
        refills_on_time_ratio_list.append(round(ontime_ratio, 3))
        secondary_pharmacy_flags.append(sec_pharm)

        baseline_hba1c_list.append(round(base_hba1c, 2))
        followup_hba1c_list.append(round(fup_hba1c, 2))
        baseline_sbp_list.append(round(base_sbp, 1))
        followup_sbp_list.append(round(fup_sbp, 1))
        baseline_dbp_list.append(round(base_dbp, 1))
        followup_dbp_list.append(round(fup_dbp, 1))
        baseline_ldl_list.append(round(base_ldl, 1))
        followup_ldl_list.append(round(fup_ldl, 1))

        actual_non_adherent_risk.append(ground_truth)

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "name": names,
        "age": ages,
        "gender": genders,
        "primary_condition": conditions,
        "medication_count": med_counts,
        "daily_dose_frequency": daily_doses,
        "copay_monthly_usd": copay_monthly,
        "pdc_rate": pdc_list,
        "refill_gap_days": refill_gap_list,
        "on_time_refill_ratio": refills_on_time_ratio_list,
        "secondary_pharmacy_activity": secondary_pharmacy_flags,
        "baseline_hba1c": baseline_hba1c_list,
        "followup_hba1c": followup_hba1c_list,
        "delta_hba1c": [round(f - b, 2) for f, b in zip(followup_hba1c_list, baseline_hba1c_list)],
        "baseline_sbp": baseline_sbp_list,
        "followup_sbp": followup_sbp_list,
        "delta_sbp": [round(f - b, 1) for f, b in zip(followup_sbp_list, baseline_sbp_list)],
        "baseline_dbp": baseline_dbp_list,
        "followup_dbp": followup_dbp_list,
        "delta_dbp": [round(f - b, 1) for f, b in zip(followup_dbp_list, baseline_dbp_list)],
        "baseline_ldl": baseline_ldl_list,
        "followup_ldl": followup_ldl_list,
        "delta_ldl": [round(f - b, 1) for f, b in zip(followup_ldl_list, baseline_ldl_list)],
        "clinical_archetype": archetypes,
        "actual_non_adherence": actual_non_adherent_risk
    })

    # Engineering composite & discordance features
    df["refill_biomarker_discordance"] = (
        (df["pdc_rate"] >= 0.80) & (df["delta_hba1c"] > 0.6) | (df["delta_sbp"] > 10.0)
    ).astype(int)

    df["refill_gap_clinical_discordance"] = (
        (df["pdc_rate"] < 0.60) & (df["delta_hba1c"] <= 0.0) & (df["delta_sbp"] <= 0.0)
    ).astype(int)

    # Standard Refill-Only rule prediction (Standard baseline threshold: PDC < 0.80)
    df["refill_only_flagged"] = (df["pdc_rate"] < 0.80).astype(int)

    return df


def generate_patient_time_series(patient_row: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates granular monthly timeline data for a single patient for deep-dive plotting:
    1. Pharmacy Refill Events (Refill Pickups, Gaps, Days Supply)
    2. Biomarker Trajectory (HbA1c, SBP over 6 months)
    """
    months = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]
    
    # Refill timeline
    pdc = patient_row["pdc_rate"]
    arch = patient_row.get("clinical_archetype", "True_Adherent")
    
    refill_statuses = []
    for m in range(6):
        if arch in ["True_Adherent", "Silent_NonAdherence"]:
            status = "Picked Up (On-Time)" if np.random.rand() > 0.1 else "Late Refill (1-3d)"
        elif arch == "True_NonAdherent":
            status = "Missed Refill" if (m in [2, 4] or np.random.rand() < 0.4) else "Delayed Pick Up"
        else: # Out of network
            status = "No Local Fill (External/Mail-Order)" if m in [1, 3, 5] else "Local Refill"
        refill_statuses.append(status)

    df_refills = pd.DataFrame({
        "Month": months,
        "Pharmacy_Refill_Status": refill_statuses,
        "Days_Covered": [30 if "Picked Up" in s or "Local" in s else (15 if "Delayed" in s else 0) for s in refill_statuses]
    })

    # Biomarker progression curve (interpolated from baseline to followup)
    base_hba1c = patient_row["baseline_hba1c"]
    fup_hba1c = patient_row["followup_hba1c"]
    hba1c_vals = np.linspace(base_hba1c, fup_hba1c, 6) + np.random.normal(0, 0.08, 6)

    base_sbp = patient_row["baseline_sbp"]
    fup_sbp = patient_row["followup_sbp"]
    sbp_vals = np.linspace(base_sbp, fup_sbp, 6) + np.random.normal(0, 1.5, 6)

    df_biomarkers = pd.DataFrame({
        "Month": months,
        "HbA1c": [round(v, 2) for v in hba1c_vals],
        "Systolic_BP": [round(v, 1) for v in sbp_vals]
    })

    return df_refills, df_biomarkers


if __name__ == "__main__":
    df = generate_patient_cohort(n_patients=500)
    print("Generated cohort shape:", df.shape)
    print("\nArchetype distribution:")
    print(df["clinical_archetype"].value_counts())
    print("\nSample records:")
    print(df[["patient_id", "name", "pdc_rate", "delta_hba1c", "clinical_archetype", "actual_non_adherence"]].head())
