import pandas as pd
import joblib

from app.schemas.risk import RiskRequest

MODEL_PATH = "ml/risk_model.pkl"

FEATURE_COLUMNS = [
    "temperature_c",
    "rainfall_mm",
    "visibility_km",
    "distance_km",
    "experience",
    "time_of_day_evening",
    "time_of_day_morning",
    "time_of_day_night",
]

model = joblib.load(MODEL_PATH)

def _build_contextual_advice(payload: RiskRequest, risk_level: str) -> str:
    """Return contextual advice string based on inputs and overall risk level."""
    factors = []

    # Simple heuristic checks
    if payload.rainfall > 2:
        factors.append("rain")
    if payload.visibility < 5:
        factors.append("visibility")
    if payload.distance > 50:
        factors.append("distance")
    if payload.temperature > 35:
        factors.append("heat")
    if payload.experience < 2:
        factors.append("low_experience")
    if payload.time_of_day.lower() in ("evening", "night"):
        factors.append("darkness")

    # If nothing stands out, fall back to generic messaging
    if not factors:
        if risk_level == "Low":
            return "Conditions are generally safe. Stay alert and ride normally."
        elif risk_level == "Medium":
            return "Moderate risk detected. Ride defensively and be prepared for sudden changes."
        else:
            return "High risk detected. Avoid riding if possible, or ride with extreme caution."

    main = factors[0]

    # Contextual messages for the main factor
    if main == "rain":
        base = "Rain is increasing your risk. Consider waiting for the rain to ease or reducing your speed significantly."
    elif main == "visibility":
        base = "Low visibility is a major risk. Slow down, increase following distance, and use lights to stay visible."
    elif main == "distance":
        base = "Long trip distance can cause fatigue. Consider shortening the ride or planning more frequent rest breaks."
    elif main == "heat":
        base = "High temperature increases fatigue and dehydration risk. Stay hydrated and plan shaded rest stops."
    elif main == "low_experience":
        base = "Given your limited riding experience, conditions today may be challenging. Reduce speed and avoid aggressive maneuvers."
    elif main == "darkness":
        base = "Riding in low light increases risk. Ensure your lights are bright and visible, and ride more slowly than usual."
    else:
        base = "Conditions require caution. Adjust your riding style and stay highly alert."

    # Optionally append a sentence based on overall risk level
    if risk_level == "High":
        base += " Overall risk is high; avoid riding if possible."
    elif risk_level == "Medium":
        base += " Overall risk is moderate; ride defensively and give yourself extra margin for error."

    return base


def predict_risk(payload: RiskRequest):
    raw_features = {
        "temperature_c": payload.temperature,
        "rainfall_mm": payload.rainfall,
        "visibility_km": payload.visibility,
        "distance_km": payload.distance,
        "experience": payload.experience,
    }

    tod = payload.time_of_day.lower()

    raw_features["time_of_day_morning"] = 1 if tod == "morning" else 0
    raw_features["time_of_day_evening"] = 1 if tod == "evening" else 0
    raw_features["time_of_day_night"] = 1 if tod == "night" else 0

    df = pd.DataFrame([raw_features])
    df = df[FEATURE_COLUMNS]

    pred = model.predict(df)[0]

    pred = max(0, min(1, float(pred)))

    if pred < 0.33:
        risk_level = "Low"
    elif pred < 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    advice = _build_contextual_advice(payload, risk_level)

    return {
        "risk_score": round(pred, 3),
        "risk_level": risk_level,
        "advice": advice,
    }
