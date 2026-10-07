"""
SME Credit Risk Detection - Flask API (v2, Pipeline-based model with Full Fintech Suite)
-----------------------------------------------------------------------------------------
Features:
1. Explainable AI (XAI) & Factor Impact Analysis
2. Interactive What-If Scenario Simulator
3. Official Printable / PDF Credit Assessment Memo
4. Batch Prediction via CSV Upload & Export
5. Analytics Dashboard & Historical Assessment Logging (SQLite)
6. Bilingual Support (Bangla ⇄ English)
"""

import sys
import json
import pickle
import sqlite3
import datetime
import io
import csv
import scipy.sparse as sp
import pandas as pd
from flask import Flask, request, jsonify, render_template, Response
from sklearn.base import BaseEstimator, TransformerMixin

# --- Custom transformer required for unpickling the trained pipeline ---
class ToDense(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray() if sp.issparse(X) else X

sys.modules["__main__"].ToDense = ToDense
sys.modules["__main__"].sp = sp

app = Flask(__name__)

# --- Model and Metadata Load ---
with open("best_model_3.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.json") as f:
    feature_columns = json.load(f)

with open("class_labels.json") as f:
    class_labels = json.load(f)

DISTRICTS = ["Barishal", "Chattogram", "Dhaka", "Khulna", "Rajshahi", "Rangpur", "Sylhet"]
LOCATION_TYPES = ["rural", "semi_urban", "urban"]
SECTORS = ["agriculture", "food_processing", "manufacturing", "retail", "services", "textiles", "trading"]

DB_PATH = "predictions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            district TEXT,
            location_type TEXT,
            sector TEXT,
            business_age_years REAL,
            owner_experience_years REAL,
            banking_years REAL,
            annual_revenue REAL,
            loan_amount REAL,
            debt_to_asset_ratio REAL,
            num_employees REAL,
            collateral_value REAL,
            past_defaults INTEGER,
            risk_level TEXT,
            confidence REAL,
            high_prob REAL,
            medium_prob REAL,
            low_prob REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()


def save_prediction(data: dict, result: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        probs = result.get("probabilities", {})
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, district, location_type, sector,
                business_age_years, owner_experience_years, banking_years,
                annual_revenue, loan_amount, debt_to_asset_ratio,
                num_employees, collateral_value, past_defaults,
                risk_level, confidence, high_prob, medium_prob, low_prob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            data.get("district", ""),
            data.get("location_type", ""),
            data.get("sector", ""),
            float(data.get("business_age_years", 0)),
            float(data.get("owner_experience_years", 0)),
            float(data.get("banking_years", 0)),
            float(data.get("annual_revenue", 0)),
            float(data.get("loan_amount", 0)),
            float(data.get("debt_to_asset_ratio", 0)),
            float(data.get("num_employees", 0)),
            float(data.get("collateral_value", 0)),
            int(data.get("past_defaults", 0)),
            result.get("risk_level", ""),
            float(result.get("confidence", 0)),
            float(probs.get("HIGH", 0)),
            float(probs.get("MEDIUM", 0)),
            float(probs.get("LOW", 0))
        ))
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id
    except Exception as e:
        print("Database save error:", e)
        return None


def build_input_row(data: dict) -> pd.DataFrame:
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


def analyze_risk_factors(data: dict, result: dict) -> dict:
    """Explainable AI (XAI) engine generating key risk drivers, positive mitigants, and bank recommendations."""
    loan_amt = float(data.get("loan_amount", 0))
    collateral = float(data.get("collateral_value", 0))
    revenue = float(data.get("annual_revenue", 1))
    debt_ratio = float(data.get("debt_to_asset_ratio", 0))
    past_defaults = int(data.get("past_defaults", 0))
    biz_age = float(data.get("business_age_years", 0))
    banking_yrs = float(data.get("banking_years", 0))
    owner_exp = float(data.get("owner_experience_years", 0))

    coverage_ratio = (collateral / loan_amt) if loan_amt > 0 else 0
    loan_to_rev = (loan_amt / revenue) if revenue > 0 else 0

    risk_factors = []
    positive_factors = []

    # 1. Past Defaults
    if past_defaults > 0:
        risk_factors.append({
            "factor_en": f"Past Defaults History ({past_defaults} count)",
            "factor_bn": f"অতীত ঋণখেলাপির ইতিহাস ({past_defaults} বার)",
            "impact": "High Negative",
            "impact_class": "badge-danger",
            "description_en": "Prior delinquencies significantly elevate default probability.",
            "description_bn": "পূর্ববর্তী ঋণখেলাপি রিস্ক স্কোর উল্লেখযোগ্যভাবে বৃদ্ধি করেছে।"
        })
    else:
        positive_factors.append({
            "factor_en": "Clean Credit History (0 Defaults)",
            "factor_bn": "নির্ভুল ক্রেডিট ইতিহাস (০ ঋণখেলাপি)",
            "impact": "High Positive",
            "impact_class": "badge-success",
            "description_en": "No past defaults reflects exemplary borrower repayment discipline.",
            "description_bn": "কোনো ঋণখেলাপি না থাকায় সময়মতো পরিশোধের নির্ভরযোগ্যতা প্রকাশ পায়।"
        })

    # 2. Collateral Coverage
    if coverage_ratio >= 1.0:
        positive_factors.append({
            "factor_en": f"Strong Collateral Coverage ({coverage_ratio*100:.1f}%)",
            "factor_bn": f"শক্তিশালী জামানত কভারেজ ({coverage_ratio*100:.1f}%)",
            "impact": "High Positive",
            "impact_class": "badge-success",
            "description_en": "Collateral exceeds requested loan, mitigating recovery risks.",
            "description_bn": "জামানতের মূল্য ঋণের পরিমাণকে ছাড়িয়ে গেছে, যা ঝুঁকি কমায়।"
        })
    elif coverage_ratio < 0.4:
        risk_factors.append({
            "factor_en": f"Low Collateral Coverage ({coverage_ratio*100:.1f}%)",
            "factor_bn": f"স্বল্প জামানত কভারেজ ({coverage_ratio*100:.1f}%)",
            "impact": "High Negative",
            "impact_class": "badge-danger",
            "description_en": "Collateral covers less than 40% of loan facility.",
            "description_bn": "জামানত ঋণের ৪০% এর কম হওয়ায় ব্যাংকের আর্থিক ঝুঁকি বেশি।"
        })
    else:
        positive_factors.append({
            "factor_en": f"Moderate Collateral Coverage ({coverage_ratio*100:.1f}%)",
            "factor_bn": f"মাঝারি জামানত কভারেজ ({coverage_ratio*100:.1f}%)",
            "impact": "Medium Positive",
            "impact_class": "badge-info",
            "description_en": "Partial tangible security provided against the facility.",
            "description_bn": "আংশিক জামানত কভারেজ বিদ্যমান।"
        })

    # 3. Debt to Asset Ratio
    if debt_ratio >= 0.8:
        risk_factors.append({
            "factor_en": f"High Debt-to-Asset Leverage ({debt_ratio:.2f})",
            "factor_bn": f"উচ্চ ঋণ-সম্পদ অনুপাত ({debt_ratio:.2f})",
            "impact": "High Negative",
            "impact_class": "badge-danger",
            "description_en": "High existing debt relative to total enterprise assets.",
            "description_bn": "ব্যবসার মোট সম্পদের তুলনায় ঋণের বোঝা অতিরিক্ত বেশি।"
        })
    elif debt_ratio <= 0.45:
        positive_factors.append({
            "factor_en": f"Conservative Leverage Profile ({debt_ratio:.2f})",
            "factor_bn": f"স্বস্তিদায়ক ঋণ-সম্পদ অনুপাত ({debt_ratio:.2f})",
            "impact": "Medium Positive",
            "impact_class": "badge-info",
            "description_en": "Healthy balance sheet headroom and low existing debt.",
            "description_bn": "ব্যবসায় অতিরিক্ত ঋণের চাপ নেই, আর্থিক সক্ষমতা ইতিবাচক।"
        })

    # 4. Loan to Revenue Ratio
    if loan_to_rev > 1.5:
        risk_factors.append({
            "factor_en": f"Heavy Loan-to-Revenue Burden ({loan_to_rev:.1f}x)",
            "factor_bn": f"বার্ষিক আয়ের তুলনায় অতিরিক্ত ঋণ ({loan_to_rev:.1f} গুণ)",
            "impact": "Medium Negative",
            "impact_class": "badge-warning",
            "description_en": "Requested facility exceeds 150% of annual turnover.",
            "description_bn": "ঋণের পরিমাণ ব্যবসার বার্ষিক আয়ের চেয়ে অনেক বেশি।"
        })

    # 5. Experience & Vintage
    if biz_age >= 5 and owner_exp >= 5:
        positive_factors.append({
            "factor_en": f"Established Business Track Record ({biz_age:.0f} yrs biz, {owner_exp:.0f} yrs exp)",
            "factor_bn": f"দীর্ঘমেয়াদী ব্যবসায়িক অভিজ্ঞতা ({biz_age:.0f} বছর ব্যবসা, {owner_exp:.0f} বছর অভিজ্ঞতা)",
            "impact": "Medium Positive",
            "impact_class": "badge-info",
            "description_en": "Experienced management and proven business longevity.",
            "description_bn": "অভিজ্ঞ পরিচালনা পর্ষদ এবং স্থিতিশীল ব্যবসা।"
        })
    elif biz_age <= 1:
        risk_factors.append({
            "factor_en": f"Early Stage / Nascent Business ({biz_age:.0f} yr)",
            "factor_bn": f"নতুন ব্যবসা প্রতিষ্ঠান ({biz_age:.0f} বছর)",
            "impact": "Medium Negative",
            "impact_class": "badge-warning",
            "description_en": "New enterprises inherently face higher initial volatility.",
            "description_bn": "ব্যবসা নতুন হওয়ায় প্রাথমিক পরিচালন অনিশ্চয়তা বিদ্যমান।"
        })

    if banking_yrs >= 3:
        positive_factors.append({
            "factor_en": f"Established Banking Tie ({banking_yrs:.0f} yrs)",
            "factor_bn": f"সন্তোষজনক ব্যাংকিং সম্পর্ক ({banking_yrs:.0f} বছর)",
            "impact": "Low Positive",
            "impact_class": "badge-info",
            "description_en": "Multi-year formal banking relationship history.",
            "description_bn": "ব্যাংকের সাথে একাধিক বছরের লেনদেনের সন্তোষজনক রেকর্ড।"
        })

    # Recommendations
    recommendations_en = []
    recommendations_bn = []

    if result["risk_level"] == "HIGH":
        if coverage_ratio < 0.8:
            target_collateral = loan_amt * 0.8
            recommendations_en.append(f"Enhance collateral security to at least BDT {target_collateral:,.0f} (80%+ coverage) or obtain a personal third-party guarantee.")
            recommendations_bn.append(f"জামানতের পরিমাণ কমপক্ষে BDT {target_collateral:,.0f} (৮০%+ কভারেজ) এ উন্নীত করা বা নির্ভরযোগ্য জামিনদারের নিশ্চয়তা নেওয়া।")
        if loan_amt > revenue * 0.75 and revenue > 0:
            suggested_loan = revenue * 0.6
            recommendations_en.append(f"Restructure facility down to ~BDT {suggested_loan:,.0f} to align with cash-flow debt service capacity.")
            recommendations_bn.append(f"আবেদনকারীর ক্যাশফ্লোর সাথে সামঞ্জস্য রেখে ঋণের পরিমাণ BDT {suggested_loan:,.0f} এ পুনঃনির্ধারণ করা।")
        if past_defaults > 0:
            recommendations_en.append("Require audited quarterly balance sheets and escrow daily sales receivables into a debt service reserve account.")
            recommendations_bn.append("পূর্ববর্তী খেলাপির কারণে ত্রৈমাসিক অডিট ও দৈনিক বিক্রয় আয়ের মাধ্যমে ঋণ আদায়ের বিশেষ শর্তারোপ করা।")
    elif result["risk_level"] == "MEDIUM":
        recommendations_en.append("Conditional sanction recommended: Disburse facility in phased tranches linked to verifiable stock inventory.")
        recommendations_bn.append("শর্তসাপেক্ষ অনুমোদন: যাচাইকৃত স্টক ইনভেন্টরির ভিত্তিতে কিস্তিতে লোন ছাড় করার শর্ত দেওয়া যেতে পারে।")
        recommendations_en.append("Conduct semi-annual business performance reviews.")
        recommendations_bn.append("প্রতি ৬ মাস অন্তর ব্যবসার কার্যক্ষমতা ও ঋণ পরিশোধ ট্র্যাকিং নিশ্চিত করা।")
    else:
        recommendations_en.append("Fast-track credit approval recommended: Eligible for standard preferential SME lending tier.")
        recommendations_bn.append("দ্রুত অনুমোদনযোগ্য: আদর্শ সুদের হারে সরাসরি ঋণ মঞ্জুরির জন্য উপযুক্ত।")
        recommendations_en.append("Opportunity to cross-sell working capital and trade finance facilities.")
        recommendations_bn.append("ভবিষ্যতে ওয়ার্কিং ক্যাপিটাল ও অন্যান্য ব্যাংকিং সেবা প্রদানের সুযোগ রয়েছে।")

    return {
        "risk_factors": risk_factors,
        "positive_factors": positive_factors,
        "recommendations_en": recommendations_en,
        "recommendations_bn": recommendations_bn,
        "coverage_ratio": round(coverage_ratio * 100, 1)
    }


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

        # Ordering probabilities in HIGH, MEDIUM, LOW
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

        # Explainable AI analysis
        xai_analysis = analyze_risk_factors(data, result)
        result["xai"] = xai_analysis

        # Save to SQLite database
        assessment_id = save_prediction(data, result)
        result["assessment_id"] = f"SME-{datetime.datetime.now().strftime('%Y%m')}-{assessment_id or 1001:04d}"
        result["timestamp"] = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

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


# --- Feature 2: What-If Live Simulator API ---
@app.route("/api/simulate", methods=["POST"])
def simulate():
    try:
        data = request.get_json()
        X = build_input_row(data)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        probabilities = dict(zip(model.classes_, proba.round(4)))

        ordered_probs = {}
        for k in ["HIGH", "MEDIUM", "LOW"]:
            if k in probabilities:
                ordered_probs[k] = float(probabilities[k])

        max_prob = float(max(proba))
        confidence = round(max_prob * 100, 1)

        result = {
            "risk_level": prediction,
            "confidence": confidence,
            "probabilities": ordered_probs,
        }
        xai = analyze_risk_factors(data, result)
        result["xai"] = xai
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --- Feature 4: Batch Prediction & CSV Export API ---
@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No CSV file uploaded."}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported."}), 400

        df = pd.read_csv(file)
        required_cols = feature_columns
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns in CSV: {', '.join(missing_cols)}"}), 400

        X = df[feature_columns]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        df["Predicted_Risk"] = predictions
        
        # Add class probs
        classes = list(model.classes_)
        for i, cls in enumerate(classes):
            df[f"Prob_{cls}"] = probabilities[:, i].round(4)
        
        max_probs = probabilities.max(axis=1) * 100
        df["Confidence_%"] = max_probs.round(1)

        # Log predictions to DB in bulk
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            res_dict = {
                "risk_level": row_dict["Predicted_Risk"],
                "confidence": row_dict["Confidence_%"],
                "probabilities": {cls: row_dict.get(f"Prob_{cls}", 0) for cls in classes}
            }
            save_prediction(row_dict, res_dict)

        results_list = df.to_dict(orient="records")
        return jsonify({
            "total_rows": len(df),
            "results": results_list,
            "risk_summary": df["Predicted_Risk"].value_counts().to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/sample-csv")
def sample_csv():
    """Generates a sample CSV template for users to download."""
    sample_data = [
        {
            "district": "Dhaka",
            "location_type": "urban",
            "sector": "manufacturing",
            "business_age_years": 7,
            "owner_experience_years": 10,
            "banking_years": 6,
            "annual_revenue": 8500000.0,
            "loan_amount": 2500000.0,
            "debt_to_asset_ratio": 0.42,
            "num_employees": 20,
            "collateral_value": 3000000.0,
            "past_defaults": 0,
        },
        {
            "district": "Barishal",
            "location_type": "rural",
            "sector": "agriculture",
            "business_age_years": 2,
            "owner_experience_years": 1,
            "banking_years": 1,
            "annual_revenue": 1400000.0,
            "loan_amount": 3000000.0,
            "debt_to_asset_ratio": 0.88,
            "num_employees": 3,
            "collateral_value": 0.0,
            "past_defaults": 2,
        },
        {
            "district": "Chattogram",
            "location_type": "semi_urban",
            "sector": "textiles",
            "business_age_years": 4,
            "owner_experience_years": 5,
            "banking_years": 3,
            "annual_revenue": 4500000.0,
            "loan_amount": 1800000.0,
            "debt_to_asset_ratio": 0.65,
            "num_employees": 12,
            "collateral_value": 1500000.0,
            "past_defaults": 0,
        }
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=feature_columns)
    writer.writeheader()
    for row in sample_data:
        writer.writerow(row)
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sme_risk_sample_template.csv"}
    )


# --- Feature 5: Analytics & History API ---
@app.route("/api/analytics")
def get_analytics():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total, AVG(confidence) as avg_conf, AVG(loan_amount) as avg_loan FROM predictions")
        row = cursor.fetchone()
        total = row["total"] if row else 0
        avg_conf = round(row["avg_conf"] or 0, 1)
        avg_loan = round(row["avg_loan"] or 0, 0)

        # Risk distribution
        cursor.execute("SELECT risk_level, COUNT(*) as count FROM predictions GROUP BY risk_level")
        risk_dist = {r["risk_level"]: r["count"] for r in cursor.fetchall()}

        # Sector distribution
        cursor.execute("SELECT sector, risk_level, COUNT(*) as count FROM predictions GROUP BY sector, risk_level")
        sector_dist = {}
        for r in cursor.fetchall():
            s = r["sector"]
            if s not in sector_dist:
                sector_dist[s] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            sector_dist[s][r["risk_level"]] = r["count"]

        # District distribution
        cursor.execute("SELECT district, risk_level, COUNT(*) as count FROM predictions GROUP BY district, risk_level")
        district_dist = {}
        for r in cursor.fetchall():
            d = r["district"]
            if d not in district_dist:
                district_dist[d] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            district_dist[d][r["risk_level"]] = r["count"]

        conn.close()
        return jsonify({
            "total_assessments": total,
            "average_confidence": avg_conf,
            "average_loan": avg_loan,
            "risk_distribution": risk_dist,
            "sector_distribution": sector_dist,
            "district_distribution": district_dist
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return jsonify({"history": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear-history", methods=["POST"])
def clear_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
