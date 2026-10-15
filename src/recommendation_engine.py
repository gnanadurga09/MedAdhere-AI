"""
Simple Clinical Action Recommendation Engine
Maps each flagged patient to simple, clear suggested actions.
"""

from typing import Dict, List, Any
import pandas as pd


def generate_action_recommendation(patient_row: pd.Series, top_risk_drivers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    risk_tier = patient_row.get("risk_tier", "Moderate Risk")
    score = patient_row.get("adherence_risk_score", 50.0)
    med_cnt = patient_row.get("medication_count", 3)
    copay = patient_row.get("copay_monthly_usd", 25.0)
    gap = patient_row.get("refill_gap_days", 0)

    # 1. Missed Case: Refilled on paper, but not taking
    if "Refilled But NOT Taken" in risk_tier or "Silent" in risk_tier:
        return {
            "action_title": "📞 Side-Effect & Pill-Taking Check (Urgent Call)",
            "urgency": "High Priority (Within 48 Hours)",
            "who_should_act": "Pharmacist or Nurse",
            "simple_why": "The patient is picking up their medicine, but their health is still getting worse. They may have side effects, confusion about how to take it, or are afraid to swallow the pills.",
            "what_to_do": [
                "Call patient to ask if the medicine is causing stomach pain, dizziness, or any side effects.",
                "Ask how many pills are currently left in their medicine bottle at home.",
                "Verify if they are taking it with meals or at the right time of day.",
                "Tell the doctor NOT to increase dosage until we confirm if the patient is actually taking it."
            ],
            "phone_script": "\"Hello! We noticed your blood sugar has gone up recently, even though you picked up your refill. We wanted to check in—are you feeling any uncomfortable side effects from your pills?\""
        }

    # 2. False Alarm: Refilled at a different pharmacy
    elif "False Alarm" in risk_tier or "Out-of-Network" in risk_tier:
        return {
            "action_title": "📋 Confirm Pharmacy Location (No Doctor Alert Needed)",
            "urgency": "Low Priority (Informational)",
            "who_should_act": "Clinic Assistant",
            "simple_why": "Our pharmacy system shows a late refill, but the patient's health numbers are healthy. They probably refilled at mail-order or a different pharmacy.",
            "what_to_do": [
                "Send a quick text/portal message asking the patient which pharmacy they currently use.",
                "Update their primary pharmacy in the clinic records.",
                "Do NOT bother the doctor with an adherence alert (prevents false alarm fatigue)."
            ],
            "phone_script": "\"Hi! Your health numbers look great! We just want to check if you refilled your prescription at another pharmacy or through mail-order so we can update your chart.\""
        }

    # 3. High Risk: Not taking medicine due to cost or too many pills
    elif score >= 70 or "High" in risk_tier:
        if copay > 50:
            act_title = "💵 Cheaper Medicine Review (Switch to Low-Cost Generics)"
            why_text = f"The patient is late for refills because their monthly medicine cost (₹{copay * 70:.0f}) is too expensive."
            steps = [
                "Review cheaper generic alternatives or low-cost Jan Aushadhi / generic pharmacy options.",
                "Connect patient with copay or financial assistance programs.",
                "Call patient to offer affordable medication substitution options."
            ]
            script = "\"Hello! We noticed your refill is overdue. We want to make sure your medicine is affordable—would you like us to switch you to a lower-cost generic brand?\""
        elif med_cnt >= 5:
            act_title = "📦 Medicine Box Simplification (All-in-one Pack)"
            why_text = f"The patient is taking {med_cnt} different medicines every day, making it easy to forget doses."
            steps = [
                "Package all daily medicines into a simple morning/evening blister pack.",
                "Set up daily reminder alarms on their phone.",
                "Synchronize all prescriptions to refill on the exact same day each month."
            ]
            script = "\"Hello! Taking multiple pills every day can be tricky. We can organize all your pills into one easy daily pack so you never have to guess.\""
        else:
            act_title = "🔔 Automated Refill Reminder & Follow-Up Call"
            why_text = f"The patient is {gap} days overdue for their refill."
            steps = [
                "Send an automated SMS text reminder to pick up refills.",
                "Schedule a 5-minute nurse check-in call.",
                "Offer home delivery for their prescription."
            ]
            script = "\"Friendly reminder from your clinic: Your refill is ready for pickup. Would you like free delivery to your home?\""

        return {
            "action_title": act_title,
            "urgency": "Medium Priority (Within 1 Week)",
            "who_should_act": "Care Coordinator",
            "simple_why": why_text,
            "what_to_do": steps,
            "phone_script": script
        }

    # 4. Low Risk: Doing great
    else:
        return {
            "action_title": "👍 Keep Up the Good Work (Routine Monitoring)",
            "urgency": "Normal Routine",
            "who_should_act": "Clinic Staff",
            "simple_why": "The patient is taking their medicine regularly and their health numbers look great. No action needed.",
            "what_to_do": [
                "Send an encouraging SMS message congratulating the patient.",
                "Keep their next routine 6-month check-up."
            ],
            "phone_script": "\"Congratulations on staying on track with your health and medicine! Keep up the great work!\""
        }
