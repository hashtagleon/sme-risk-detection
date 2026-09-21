"""
SME Credit Risk Detection - Flask API
--------------------------------------
এই অ্যাপটা trained Random Forest model লোড করে একটা web form + JSON API দিয়ে
risk_level (LOW / MEDIUM / HIGH) predict করে।

লোকাল রান: python app.py  (তারপর http://localhost:5000 এ যাও)
"""

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- মডেল ও মেটাডেটা লোড ---
model = joblib.load("sme_risk_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
districts = sorted(joblib.load("districts.pkl"))
location_types = sorted(joblib.load("location_types.pkl"))
sectors = sorted(joblib.load("sectors.pkl"))


def build_feature_row(data: dict) -> pd.DataFrame:
    """
    ইউজারের ইনপুট (raw form/JSON data) থেকে model-এর expected feature format
    (one-hot encoded columns) বানানো হয় এখানে।
    """
    business_age = float(data["business_age_years"])
    owner_exp = float(data["owner_experience_years"])
    banking_years = float(data["banking_years"])
    annual_revenue = float(data["annual_revenue"])
    loan_amount = float(data["loan_amount"])
    debt_to_asset_ratio = float(data["debt_to_asset_ratio"])
    num_employees = float(data["num_employees"])
    collateral_value = float(data["collateral_value"])
    past_defaults = int(data["past_defaults"])

    district = data["district"]
    location_type = data["location_type"]
    sector = data["sector"]

    # derived features (training-এর সময় যেগুলো বানানো হয়েছিল, একই লজিক এখানে)
    loan_to_revenue = loan_amount / annual_revenue if annual_revenue else 0
    collateral_coverage = collateral_value / loan_amount if loan_amount else 0
    has_collateral = 1 if collateral_value > 0 else 0

    row = {
        "business_age_years": business_age,
        "owner_experience_years": owner_exp,
        "banking_years": banking_years,
        "annual_revenue": annual_revenue,
        "loan_amount": loan_amount,
        "debt_to_asset_ratio": debt_to_asset_ratio,
        "num_employees": num_employees,
        "collateral_value": collateral_value,
        "past_defaults": past_defaults,
        "loan_to_revenue": loan_to_revenue,
        "collateral_coverage": collateral_coverage,
        "has_collateral": has_collateral,
    }

    # one-hot encoding manually (training এর drop_first=True এর সাথে সামঞ্জস্যপূর্ণ)
    for d in districts:
        col = f"district_{d}"
        if col in feature_columns:
            row[col] = 1 if district == d else 0
    for lt in location_types:
        col = f"location_type_{lt}"
        if col in feature_columns:
            row[col] = 1 if location_type == lt else 0
    for s in sectors:
        col = f"sector_{s}"
        if col in feature_columns:
            row[col] = 1 if sector == s else 0

    df = pd.DataFrame([row])
    # কোনো column missing থাকলে 0 দিয়ে ভরে, আর ঠিক training-এর অর্ডারে সাজানো
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


@app.route("/")
def home():
    return render_template(
        "index.html",
        districts=districts,
        location_types=location_types,
        sectors=sectors,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        X = build_feature_row(data)
        prediction = model.predict(X)[0]
        probabilities = dict(zip(model.classes_, model.predict_proba(X)[0].round(3)))

        result = {
            "risk_level": prediction,
            "probabilities": {k: float(v) for k, v in probabilities.items()},
        }

        if request.is_json:
            return jsonify(result)
        return render_template(
            "index.html",
            districts=districts,
            location_types=location_types,
            sectors=sectors,
            result=result,
            form_data=data,
        )
    except Exception as e:
        error = {"error": str(e)}
        if request.is_json:
            return jsonify(error), 400
        return render_template(
            "index.html",
            districts=districts,
            location_types=location_types,
            sectors=sectors,
            error=str(e),
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
