"""
SME Credit Risk Detection - Flask API (v2, Pipeline-based model)
-------------------------------------------------------------------
মডেলটা একটা sklearn Pipeline (ColumnTransformer -> ToDense -> GradientBoostingClassifier),
তাই raw applicant data সরাসরি DataFrame আকারে দিলেই pipeline নিজে preprocessing করে নেয়।
আলাদা করে one-hot encoding করার দরকার নেই (আগের ভার্সনে যেটা করতে হতো)।

লোকাল রান: python app.py  (তারপর http://localhost:5000 এ যাও)
"""

import sys
import json
import pickle
import scipy.sparse as sp
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.base import BaseEstimator, TransformerMixin


# --- Custom transformer: মডেলের ভেতরে এই class টা ব্যবহার হয়েছে, তাই
#     unpickle করার আগে ঠিক এই নামেই এখানে ডেফাইন করা থাকতে হবে। ---
class ToDense(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray() if sp.issparse(X) else X


# pickle module __main__-এ class খোঁজে, তাই এভাবে রেজিস্টার করে দেওয়া হলো
sys.modules["__main__"].ToDense = ToDense
sys.modules["__main__"].sp = sp

app = Flask(__name__)

# --- মডেল ও মেটাডেটা লোড ---
with open("best_model_3.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.json") as f:
    feature_columns = json.load(f)   # raw column order, pipeline এর ColumnTransformer এই অর্ডার আশা করে

with open("class_labels.json") as f:
    class_labels = json.load(f)      # ['HIGH', 'LOW', 'MEDIUM'] -- model.classes_ এর সাথে মিলে যাওয়া উচিত

DISTRICTS = ["Barishal", "Chattogram", "Dhaka", "Khulna", "Rajshahi", "Rangpur", "Sylhet"]
LOCATION_TYPES = ["rural", "semi_urban", "urban"]
SECTORS = ["agriculture", "food_processing", "manufacturing", "retail", "services", "textiles", "trading"]


def build_input_row(data: dict) -> pd.DataFrame:
    """ইউজারের raw ইনপুট থেকে ঠিক feature_columns.json-এর অর্ডারে একটা DataFrame row বানায়।
    কোনো manual encoding লাগে না -- pipeline নিজেই (ColumnTransformer) সেটা করবে।"""
    row = {
        "district": data["district"],
        "location_type": data["location_type"],
        "sector": data["sector"],
        "business_age_years": float(data["business_age_years"]),
        "owner_experience_years": float(data["owner_experience_years"]),
        "banking_years": float(data["banking_years"]),
        "annual_revenue": float(data["annual_revenue"]),
        "loan_amount": float(data["loan_amount"]),
        "debt_to_asset_ratio": float(data["debt_to_asset_ratio"]),
        "num_employees": float(data["num_employees"]),
        "collateral_value": float(data["collateral_value"]),
        "past_defaults": int(data["past_defaults"]),
    }
    return pd.DataFrame([row])[feature_columns]


@app.route("/")
def home():
    return render_template(
        "index.html",
        districts=DISTRICTS,
        location_types=LOCATION_TYPES,
        sectors=SECTORS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        X = build_input_row(data)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        probabilities = dict(zip(model.classes_, proba.round(4)))

        # Ensure consistent order: HIGH, MEDIUM, LOW
        ordered_probs = {}
        for k in ["HIGH", "MEDIUM", "LOW"]:
            if k in probabilities:
                ordered_probs[k] = float(probabilities[k])
        for k, v in probabilities.items():
            if k not in ordered_probs:
                ordered_probs[k] = float(v)

        max_prob = float(max(proba))
        confidence = round(max_prob * 100, 1)

        result = {
            "risk_level": prediction,
            "confidence": confidence,
            "probabilities": ordered_probs,
        }

        if request.is_json:
            return jsonify(result)
        return render_template(
            "index.html",
            districts=DISTRICTS,
            location_types=LOCATION_TYPES,
            sectors=SECTORS,
            result=result,
            form_data=data,
        )
    except Exception as e:
        error = {"error": str(e)}
        if request.is_json:
            return jsonify(error), 400
        return render_template(
            "index.html",
            districts=DISTRICTS,
            location_types=LOCATION_TYPES,
            sectors=SECTORS,
            error=str(e),
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
