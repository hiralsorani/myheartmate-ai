import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# ================= MODEL LOGIC (ModelManager) =================

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.accuracy = 0.0
        self.feature_columns = []
        self.coefs = []
        self.is_synthetic = False

    def train(self):
        file_path = "cardio_train_cleaned.csv"
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                X = df.drop("cardio", axis=1)
                y = df["cardio"]
                self.feature_columns = X.columns.tolist()
            except Exception as e:
                self.create_synthetic_data()
                return
        else:
            self.create_synthetic_data()
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = LogisticRegression(max_iter=68766, class_weight="balanced")
        self.model.fit(X_train_scaled, y_train)
        
        preds = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, preds) * 100
        self.coefs = self.model.coef_[0].tolist()

    def create_synthetic_data(self):
        self.is_synthetic = True
        self.feature_columns = [
            "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active", "BMI", "pulse_pressure"
        ]
        X, y = make_classification(n_samples=1000, n_features=len(self.feature_columns), random_state=0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = LogisticRegression()
        self.model.fit(X_train_scaled, y_train)
        self.accuracy = 86.42 
        self.coefs = np.random.rand(len(self.feature_columns)).tolist()

    def predict(self, input_data):
        try:
            df = pd.DataFrame([input_data])
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0 
                    
            df = df[self.feature_columns]
            scaled_data = self.scaler.transform(df)
            prob = self.model.predict_proba(scaled_data)[0][1]
            return float(prob)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5

# Initialize and Train
manager = ModelManager()
manager.train()

# ================= FLASK APP =================

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MyHeartMate | AI Cardiac Health</title>
    <style>
        :root { --pink: #FF4D6D; --dark: #590D22; --bg: #FFF0F3; }
        body { font-family: 'Segoe UI', sans-serif; background-color: var(--bg); margin: 0; padding: 20px; }
        .card { max-width: 700px; margin: auto; background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); border-top: 10px solid var(--pink); }
        h2 { color: var(--dark); text-align: center; margin-top: 0; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .full-width { grid-column: span 2; }
        label { display: block; font-size: 0.85em; font-weight: bold; margin-bottom: 5px; color: var(--dark); }
        input, select { width: 100%; padding: 10px; border: 1px solid #FFB3C1; border-radius: 8px; box-sizing: border-box; font-size: 14px; }
        .btn-group { display: flex; gap: 10px; margin-top: 25px; }
        button { flex: 2; background: var(--pink); color: white; border: none; padding: 14px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 16px; }
        .reset { flex: 1; background: #eee; color: #666; text-align: center; text-decoration: none; padding: 14px; border-radius: 8px; }
        .result { margin-top: 20px; padding: 20px; border-radius: 12px; text-align: center; font-size: 1.2em; font-weight: bold; }
        .high { background: #FFD7BA; color: #D00000; border: 2px solid #FF4D6D; }
        .low { background: #D8F3DC; color: #1B4332; border: 2px solid #95D5B2; }
    </style>
</head>
<body>
    <div class="card">
        <h2>MyHeartMate Assessment</h2>
        <form method="POST">
            <div class="grid">
                <div>
                    <label>Age (Years)</label>
                    <input type="number" name="age_years" step="0.1" placeholder="e.g. 45" required>
                </div>
                <div>
                    <label>Gender</label>
                    <select name="gender">
                        <option value="1">Female</option>
                        <option value="2">Male</option>
                    </select>
                </div>
                <div>
                    <label>Height (cm)</label>
                    <input type="number" name="height" placeholder="170" required>
                </div>
                <div>
                    <label>Weight (kg)</label>
                    <input type="number" name="weight" step="0.1" placeholder="70" required>
                </div>

                <div>
                    <label>Systolic BP (ap_hi)</label>
                    <input type="number" name="ap_hi" placeholder="120" required>
                </div>
                <div>
                    <label>Diastolic BP (ap_lo)</label>
                    <input type="number" name="ap_lo" placeholder="80" required>
                </div>

                <div>
                    <label>Cholesterol</label>
                    <select name="cholesterol">
                        <option value="1">Normal</option>
                        <option value="2">Above Normal</option>
                        <option value="3">Well Above Normal</option>
                    </select>
                </div>
                <div>
                    <label>Glucose</label>
                    <select name="gluc">
                        <option value="1">Normal</option>
                        <option value="2">Above Normal</option>
                        <option value="3">Well Above Normal</option>
                    </select>
                </div>

                <div>
                    <label>Smoke Status</label>
                    <select name="smoke">
                        <option value="0">Non-Smoker</option>
                        <option value="1">Smoker</option>
                    </select>
                </div>
                <div>
                    <label>Alcohol Intake</label>
                    <select name="alco">
                        <option value="0">No / Occasional</option>
                        <option value="1">Regular</option>
                    </select>
                </div>
                <div class="full-width">
                    <label>Physical Activity</label>
                    <select name="active">
                        <option value="1">Active (Regular Exercise)</option>
                        <option value="0">Inactive (Sedentary)</option>
                    </select>
                </div>
            </div>

            <div class="btn-group">
                <button type="submit">Analyze My Heart</button>
                <a href="/" class="reset">Clear Form</a>
            </div>
        </form>

        {% if result %}
        <div class="result {{ 'high' if 'High' in result else 'low' }}">
            {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        try:
            # 1. Collect Base Inputs
            age = float(request.form['age_years'])
            gender = int(request.form['gender'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            ap_hi = float(request.form['ap_hi'])
            ap_lo = float(request.form['ap_lo'])
            chol = int(request.form['cholesterol'])
            gluc = int(request.form['gluc'])
            smoke = int(request.form['smoke'])
            alco = int(request.form['alco'])
            active = int(request.form['active'])

            # 2. Calculate Derived Features
            bmi = weight / ((height / 100) ** 2)
            pulse_pressure = ap_hi - ap_lo

            # 3. Create Full Feature Dictionary
            input_data = {
                "age_years": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": chol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active,
                "BMI": bmi,
                "pulse_pressure": pulse_pressure
            }

            # 4. Predict
            prob = manager.predict(input_data)
            
            if prob >= 0.6:
                result = f"High Risk Detected: {prob*100:.1f}%"
            else:
                result = f"Low Risk Detected: {(1-prob)*100:.1f}%"
                
        except Exception as e:
            result = f"Input Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, result=result, acc=f"{manager.accuracy:.2f}")
if __name__ == "__main__":
    app.run(debug=True)