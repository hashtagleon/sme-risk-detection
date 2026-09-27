"""
SME Credit Risk Detection - Flask API & Web Application
------------------------------------------------------
এই অ্যাপটি trained Random Forest মডেল লোড করে ওয়েব ফর্ম, ডেভলপার REST API এবং
হাই-পারফর্ম্যান্স ভেক্টরাইজড ব্যাচ CSV প্রসেসিং দিয়ে risk_level (LOW / MEDIUM / HIGH)
এবং এক্সপ্লেইনেবল ইনসাইটস প্রদান করে।
"""

import io
import csv
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, Response

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
    (one-hot encoded columns) তৈরি করে।
    """
    try:
        business_age = max(0.0, float(data.get("business_age_years", 0)))
        owner_exp = max(0.0, float(data.get("owner_experience_years", 0)))
        banking_years = max(0.0, float(data.get("banking_years", 0)))
        annual_revenue = max(0.0, float(data.get("annual_revenue", 0)))
        loan_amount = max(0.0, float(data.get("loan_amount", 0)))
        debt_to_asset_ratio = max(0.0, float(data.get("debt_to_asset_ratio", 0)))
        num_employees = max(0.0, float(data.get("num_employees", 0)))
        collateral_value = max(0.0, float(data.get("collateral_value", 0)))
        past_defaults = max(0, int(data.get("past_defaults", 0)))
    except (ValueError, TypeError):
        business_age = owner_exp = banking_years = annual_revenue = loan_amount = debt_to_asset_ratio = num_employees = collateral_value = past_defaults = 0

    district = str(data.get("district", districts[0] if districts else ""))
    location_type = str(data.get("location_type", location_types[0] if location_types else ""))
    sector = str(data.get("sector", sectors[0] if sectors else ""))

    loan_to_revenue = (loan_amount / annual_revenue) if annual_revenue > 0 else 0.0
    collateral_coverage = (collateral_value / loan_amount) if loan_amount > 0 else 0.0
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

    # One-hot encoding
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
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def build_batch_feature_matrix(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    ভেক্টরাইজড পদ্ধতিতে পুরো DataFrame-কে একবারে মডেলের ফিচার ম্যাট্রিক্সে রূপান্তর করে।
    """
    df = df_raw.copy()
    
    # Numeric conversions
    num_cols = [
        "business_age_years", "owner_experience_years", "banking_years",
        "annual_revenue", "loan_amount", "debt_to_asset_ratio",
        "num_employees", "collateral_value", "past_defaults"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived features
    df["loan_to_revenue"] = np.where(df["annual_revenue"] > 0, df["loan_amount"] / df["annual_revenue"], 0.0)
    df["collateral_coverage"] = np.where(df["loan_amount"] > 0, df["collateral_value"] / df["loan_amount"], 0.0)
    df["has_collateral"] = (df["collateral_value"] > 0).astype(int)

    # One-hot encoding for categorical columns
    for d in districts:
        col = f"district_{d}"
        if col in feature_columns:
            df[col] = (df["district"].astype(str) == d).astype(int)
    for lt in location_types:
        col = f"location_type_{lt}"
        if col in feature_columns:
            df[col] = (df["location_type"].astype(str) == lt).astype(int)
    for s in sectors:
        col = f"sector_{s}"
        if col in feature_columns:
            df[col] = (df["sector"].astype(str) == s).astype(int)

    # Ensure all feature columns are present and strictly aligned
    X = df.reindex(columns=feature_columns, fill_value=0)
    return X


def generate_insights(data: dict, risk_level: str, probabilities: dict) -> dict:
    """
    মডেলের প্রেডিকশন এবং ইনপুট তথ্যের ওপর ভিত্তি করে ব্যাখ্যামূলক রিস্ক ফ্যাক্টর ও সিদ্ধান্ত তৈরি করে।
    """
    try:
        annual_revenue = float(data.get("annual_revenue", 0))
        loan_amount = float(data.get("loan_amount", 0))
        debt_to_asset_ratio = float(data.get("debt_to_asset_ratio", 0))
        collateral_value = float(data.get("collateral_value", 0))
        past_defaults = int(data.get("past_defaults", 0))
        banking_years = float(data.get("banking_years", 0))
        owner_exp = float(data.get("owner_experience_years", 0))
        business_age = float(data.get("business_age_years", 0))
    except (ValueError, TypeError):
        annual_revenue = loan_amount = debt_to_asset_ratio = collateral_value = past_defaults = banking_years = owner_exp = business_age = 0

    collateral_coverage = (collateral_value / loan_amount * 100) if loan_amount > 0 else 0
    loan_to_revenue = (loan_amount / annual_revenue * 100) if annual_revenue > 0 else 0

    drivers = []
    
    # 1. Past Defaults Driver (English First - Bengali Second)
    if past_defaults == 0:
        drivers.append({"type": "positive", "text": "Clean CIB Track Record — পূর্বে কোনো ঋণখেলাপির রেকর্ড নেই"})
    elif past_defaults == 1:
        drivers.append({"type": "warning", "text": "Minor Credit Concern — অতীতে ১টি ঋণখেলাপির রেকর্ড রয়েছে"})
    else:
        drivers.append({"type": "negative", "text": f"Severe Default History — অতীতে {past_defaults}টি ঋণখেলাপির রেকর্ড বিদ্যমান"})

    # 2. Collateral Coverage Driver
    if collateral_coverage >= 100:
        drivers.append({"type": "positive", "text": f"Adequate Collateral Coverage — পর্যাপ্ত স্থাবর জামানত সুরক্ষা বিদ্যমান (LTV: {collateral_coverage:.1f}%)"})
    elif collateral_coverage > 0:
        drivers.append({"type": "warning", "text": f"Partial Collateral Coverage — আংশিক জামানত কভারেজ (LTV: {collateral_coverage:.1f}%)"})
    else:
        drivers.append({"type": "negative", "text": "Unsecured Exposure Risk — কোনো স্থাবর জামানত জমা দেওয়া হয়নি"})

    # 3. Debt to Asset Ratio Driver
    if debt_to_asset_ratio < 0.35:
        drivers.append({"type": "positive", "text": f"Healthy Debt-to-Asset Ratio — ঋণ-সম্পদ অনুপাত খুবই স্থিতিশীল ({debt_to_asset_ratio:.2f})"})
    elif debt_to_asset_ratio <= 0.65:
        drivers.append({"type": "neutral", "text": f"Moderate Debt-to-Asset Ratio — ঋণ-সম্পদ অনুপাত সহনশীল সীমার মধ্যে রয়েছে ({debt_to_asset_ratio:.2f})"})
    else:
        drivers.append({"type": "negative", "text": f"High Debt-to-Asset Leverage — উচ্চ ঋণ-সম্পদ অনুপাত ও অতিরিক্ত আর্থিক দায় ({debt_to_asset_ratio:.2f})"})

    # 4. Experience, Tenure & Banking Relationship
    if business_age >= 5 and owner_exp >= 5:
        drivers.append({"type": "positive", "text": f"Established Enterprise & Trade Experience — দীর্ঘ ব্যবসায়িক সুনাম ({int(business_age)} বছর) ও ট্রেড অভিজ্ঞতা ({int(owner_exp)} বছর)"})
    elif business_age < 3:
        drivers.append({"type": "warning", "text": "Early-Stage Enterprise — ব্যবসা প্রতিষ্ঠার সময়কাল ৩ বছরের কম"})

    if banking_years >= 5:
        drivers.append({"type": "positive", "text": f"Strong Banking Relationship — দীর্ঘ ব্যাংকিং লেনদেন সম্পর্ক ({int(banking_years)} বছর)"})

    # Calculate calibrated risk score (0-100 scale)
    p_low = probabilities.get("LOW", 0.33)
    p_med = probabilities.get("MEDIUM", 0.33)
    p_high = probabilities.get("HIGH", 0.33)
    calculated_risk_score = round((p_med * 50) + (p_high * 100), 1)

    # Underwriting Recommendation (English First - Bengali Second)
    if risk_level == "LOW":
        decision_title = "Standard Prime Terms — ঋণ অনুমোদনযোগ্য"
        decision_badge = "Approved — অনুমোদনযোগ্য"
        decision_summary = "Low Credit Risk: Recommended for sanction under standard lending terms and prevailing interest rate. — গ্রাহকের ঋণ পরিশোধ ঝুঁকি নিম্নমানের। নিয়মিত ব্যাংকিং শর্ত ও প্রচলিত সুদের হারে ক্রেডিট সুবিধা মঞ্জুর করা যেতে পারে।"
        decision_color = "low"
    elif risk_level == "MEDIUM":
        decision_title = "Conditional Underwriting — শর্তসাপেক্ষে বিবেচনাযোগ্য"
        decision_badge = "Conditional — শর্তসাপেক্ষ"
        decision_summary = "Moderate Credit Risk: Recommended for sanction with enhanced monitoring, personal guarantee, and structured cash-flow repayment. — গ্রাহকের ঋণ পরিশোধে পরিমিত ঝুঁকি রয়েছে। অতিরিক্ত পার্সোনাল গ্যারান্টি, তদারকি ও ক্যাশ-ফ্লো ভিত্তিক কিস্তির শর্তে অনুমোদন বিবেচনাযোগ্য।"
        decision_color = "medium"
    else:
        decision_title = "High Risk / Subprime Alert — উচ্চ ঝুঁকিপূর্ণ"
        decision_badge = "High Risk — উচ্চ ঝুঁকি"
        decision_summary = "Critical Default Risk: Sanction is strictly discouraged without 100% liquid cash/FDR collateral. — আবেদনে খেলাপি হওয়ার ঝুঁকি অত্যধিক। প্রচলিত কাঠামোতে আবেদনটি নামঞ্জুর বা শতভাগ লিকুইড এফডিআর জামানত ব্যতীত অনুমোদন অনুচিত।"
        decision_color = "high"

    return {
        "risk_score": calculated_risk_score,
        "collateral_coverage": round(collateral_coverage, 1),
        "loan_to_revenue": round(loan_to_revenue, 1),
        "drivers": drivers,
        "recommendation": {
            "title": decision_title,
            "badge": decision_badge,
            "summary": decision_summary,
            "color": decision_color
        }
    }


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

        if not data:
            raise ValueError("কোনো ইনপুট ডেটা পাওয়া যায়নি।")

        X = build_feature_row(data)
        prediction = model.predict(X)[0]
        probabilities = dict(zip(model.classes_, model.predict_proba(X)[0].round(3)))
        
        prob_dict = {k: float(v) for k, v in probabilities.items()}
        insights = generate_insights(data, prediction, prob_dict)

        result = {
            "risk_level": prediction,
            "probabilities": prob_dict,
            "insights": insights
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


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    হাই-স্পিড ভেক্টরাইজড ব্যাচ প্রেডিকশন ইঞ্জিন: হাজার হাজার রো এক নিমিষে প্রসেস করে।
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "কোনো CSV ফাইল পাওয়া যায়নি।"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "অনুগ্রহ করে একটি বৈধ .csv ফাইল আপলোড করুন।"}), 400

        df_input = pd.read_csv(file)
        
        required_cols = [
            "business_age_years", "owner_experience_years", "banking_years",
            "annual_revenue", "loan_amount", "debt_to_asset_ratio",
            "num_employees", "collateral_value", "past_defaults",
            "district", "location_type", "sector"
        ]

        missing = [col for col in required_cols if col not in df_input.columns]
        if missing:
            return jsonify({"error": f"CSV ফাইলে এই আবশ্যক কলামগুলো অনুপস্থিত: {', '.join(missing)}"}), 400

        # High-Speed Vectorized Batch Inference
        X_batch = build_batch_feature_matrix(df_input)
        predictions = model.predict(X_batch)
        probabilities = model.predict_proba(X_batch).round(3)
        classes = list(model.classes_)

        low_idx = classes.index("LOW") if "LOW" in classes else 0
        med_idx = classes.index("MEDIUM") if "MEDIUM" in classes else 1
        high_idx = classes.index("HIGH") if "HIGH" in classes else 2

        results = []
        for idx in range(len(df_input)):
            p_low = float(probabilities[idx][low_idx])
            p_med = float(probabilities[idx][med_idx])
            p_high = float(probabilities[idx][high_idx])
            pred_risk = str(predictions[idx])
            
            score = round((p_med * 50.0) + (p_high * 100.0), 1)

            results.append({
                "index": idx + 1,
                "district": str(df_input.iloc[idx].get("district", "")),
                "sector": str(df_input.iloc[idx].get("sector", "")),
                "loan_amount": float(pd.to_numeric(df_input.iloc[idx].get("loan_amount", 0), errors="coerce") or 0),
                "annual_revenue": float(pd.to_numeric(df_input.iloc[idx].get("annual_revenue", 0), errors="coerce") or 0),
                "predicted_risk": pred_risk,
                "risk_score": score,
                "prob_low": p_low,
                "prob_medium": p_med,
                "prob_high": p_high
            })

        return jsonify({
            "status": "success",
            "total_records": len(results),
            "data": results
        })

    except Exception as e:
        return jsonify({"error": f"ব্যাচ প্রক্রিয়াকরণে ত্রুটি: {str(e)}"}), 400


@app.route("/sample_template_csv")
def sample_template_csv():
    """
    ব্যবহারকারীদের টেস্ট করার জন্য নমুনা CSV ফাইল ডাউনলোড সুবিধা।
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    headers = [
        "business_age_years", "owner_experience_years", "banking_years",
        "annual_revenue", "loan_amount", "debt_to_asset_ratio",
        "num_employees", "collateral_value", "past_defaults",
        "district", "location_type", "sector"
    ]
    writer.writerow(headers)

    d1 = districts[0] if districts else "Dhaka"
    d2 = districts[1] if len(districts) > 1 else d1
    d3 = districts[2] if len(districts) > 2 else d1

    l1 = location_types[0] if location_types else "Urban"
    s1 = sectors[0] if sectors else "Retail"
    s2 = sectors[1] if len(sectors) > 1 else s1

    # Realistic Sample rows
    writer.writerow([10, 12, 8, 8500000, 2000000, 0.28, 25, 3500000, 0, d1, l1, s1]) # Prime Low Risk
    writer.writerow([4, 5, 3, 3200000, 1800000, 0.55, 8, 1000000, 0, d2, l1, s2])     # Moderate
    writer.writerow([2, 1, 1, 1400000, 3000000, 0.88, 3, 0, 2, d3, l1, s1])           # Subprime High Risk

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sme_credit_sample_template.csv"}
    )


if __name__ == "__main__":
    app.run(debug=True)
