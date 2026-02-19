import pandas as pd
import numpy as np
import joblib

model = joblib.load("risk_score_model.pkl")
feature_names = model.feature_names_in_


def weighted_risk_prediction_with_contributions(
    user_input, model=model, feature_names=feature_names, w_ml=0.7, w_importance=0.3
):

    # Normalize weights
    total_w = w_ml + w_importance
    if total_w == 0:
        raise ValueError("w_ml and w_importance cannot both be zero.")
    w_ml /= total_w
    w_importance /= total_w

    # Ensure all features exist in input
    for f in feature_names:
        user_input.setdefault(f, 0)

    # Convert dict to DataFrame in correct feature order
    input_vector = np.array([user_input[f] for f in feature_names])
    input_df = pd.DataFrame([input_vector], columns=feature_names)

    # Model prediction
    probabilities = model.predict_proba(input_df)[0]
    # predicted_class = model.classes_[np.argmax(probabilities)]
    # kinaki esle highest prob value aako lai choose garcha tara risk predict garna chai ml_risk_score (probability of high risk)
    # allows you to blend it meaningfully with feature importance (importance_score) to get a smooth reality_score
    # Using predicted_class directly would make reality_score mostly discrete, defeating the purpose of a weighted continuous score

    # ML Risk Score
    high_risk_index = list(model.classes_).index(2)
    ml_risk_score = probabilities[high_risk_index]

    # Importance-based score
    importances = model.feature_importances_
    weighted_sum = np.dot(input_vector, importances)
    max_possible = np.dot(np.ones(len(importances)), importances)
    importance_score = weighted_sum / max_possible if max_possible > 0 else 0
    importance_score = float(np.clip(importance_score, 0, 1))

    # Feature contributions
    feature_contributions = pd.DataFrame(
        {
            "Feature": feature_names,
            "Value": input_vector,
            "Importance": importances,
            "Contribution": input_vector * importances,
        }
    ).sort_values(by="Contribution", ascending=False, ignore_index=True)

    # Final weighted score
    reality_score = w_ml * ml_risk_score + w_importance * importance_score

    # Risk interpretation
    if reality_score < 0.3:
        risk_level = "🔋Low Risk"
    elif reality_score < 0.6:
        risk_level = "🪫Moderate Risk"
    else:
        risk_level = "🚨High Risk"

    return {
        # "predicted_class": predicted_class,
        "class_probabilities": {
            int(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)
        },
        "ml_risk_score": ml_risk_score,
        "importance_score": importance_score,
        "final_weighted_score": reality_score,
        "interpreted_risk": risk_level,
        "feature_contributions": feature_contributions,
    }


def predict(user_input):
    return weighted_risk_prediction_with_contributions(user_input)
