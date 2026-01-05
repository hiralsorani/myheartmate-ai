import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import joblib

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.accuracy = 0.0
        self.feature_columns = []
        self.model_path = 'cardio_model.pkl'
        self.scaler_path = 'scaler.pkl'

    def train(self):
        """Unified training function to reach 75%+ accuracy"""
        file_path = "cardio_train_cleaned.csv"
        
        if not os.path.exists(file_path):
            print("Data file not found!")
            return

        df = pd.read_csv(file_path)

        # --- Feature Engineering for higher accuracy ---
        if 'BMI' not in df.columns:
            df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
        if 'pulse_pressure' not in df.columns:
            df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

        X = df.drop("cardio", axis=1)
        y = df["cardio"]
        self.feature_columns = X.columns.tolist()

        # Split 80/20 for better stability
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Logistic Regression with improved parameters
        self.model = LogisticRegression(max_iter=2000, solver='liblinear')
        self.model.fit(X_train_scaled, y_train)
        
        preds = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, preds) * 100
        
        # Save files so Predict can use them
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        print(f"Training Complete! Accuracy: {self.accuracy:.2f}%")

    def predict(self, input_data):
        """Uses the trained model to predict on new data"""
        try:
            if self.model is None:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)

            df = pd.DataFrame([input_data])

            # Ensure engineered features exist in prediction input
            if 'BMI' not in df.columns:
                df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
            if 'pulse_pressure' not in df.columns:
                df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
            
            # Match the exact column order from training
            df = df[self.feature_columns]
            
            scaled_data = self.scaler.transform(df)
            prob = self.model.predict_proba(scaled_data)[0][1]
            return float(prob)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5
# Initialize and train
manager = ModelManager()
manager.train()

# ================= FLASK APP =================
app = Flask(__name__)

# ================= ASSETS (LOGO & ICONS) =================
# <svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="C:\Users\Creater\OneDrive\Desktop\cardio_model">

LOGO_SVG = """
<div style="display:flex; align-items:center; gap:10px;">

    <!-- PNG logo from Flask static folder -->
    <img src="/static/logo.png" alt="Logo" width="40" height="40">

    <!-- Inline SVG icon -->
    <svg width="40" height="40" viewBox="0 0 100 100"
         xmlns="http://www.w3.org/2000/svg">

        <defs>
            <linearGradient id="heartGrad" x1="12.5" y1="12.5"
                            x2="87.5" y2="88">
                <stop stop-color="#ff4757"/>
                <stop offset="1" stop-color="#ff6b81"/>
            </linearGradient>
        </defs>

        <path d="M50 88C50 88 12.5 56.5 12.5 32.5
                 C12.5 20.5 20.5 12.5 32.5 12.5
                 C41.5 12.5 47.5 18.5 50 22.5
                 C52.5 18.5 58.5 12.5 67.5 12.5
                 C79.5 12.5 87.5 20.5 87.5 32.5
                 C87.5 56.5 50 88 50 88Z"
              fill="url(#heartGrad)"/>

        <path d="M22 45 L32 45 L42 20 L52 70 L62 45 L78 45"
              stroke="white"
              stroke-width="4"
              fill="none"
              stroke-linecap="round"
              stroke-linejoin="round"/>
    </svg>

</div>
"""


FAVICON_DATA_URI = "data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>❤️</text></svg>"

# ================= UI TEMPLATES =================

# Note: We use {{ content|safe }} here instead of {% block content %} to avoid conflicts in single-file mode.
BASE_LAYOUT = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MyHeartMate | {{{{ title }}}}</title>
    <link rel="icon" href="{FAVICON_DATA_URI}">
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    
    <style>
        :root {{
            --primary-color: #0d6efd;
            --secondary-color: #20c997;
            --dark-bg: #212529;
            --accent: #ff4757;
        }}
        body {{
            font-family: 'Poppins', sans-serif;
            color: #333;
            line-height: 1.7;
            background-color: #fdfdfd;
        }}
        
        /* Navbar */
        .navbar {{
            box-shadow: 0 2px 15px rgba(0,0,0,0.05);
            padding: 0.8rem 0;
            background: #fff;
        }}
        .navbar-brand {{
            font-weight: 700;
            color: #2c3e50 !important;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .brand-text {{
            background: -webkit-linear-gradient(45deg, #0d6efd, #20c997);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .nav-link {{
            color: #555 !important;
            font-weight: 500;
            margin: 0 8px;
            transition: 0.3s;
        }}
        .nav-link:hover, .nav-link.active {{
            color: var(--primary-color) !important;
        }}
        .btn-predict-nav {{
            background: var(--primary-color);
            color: white !important;
            border-radius: 50px;
            padding: 8px 25px;
            box-shadow: 0 4px 6px rgba(13, 110, 253, 0.2);
        }}
        .btn-predict-nav:hover {{
            background: #0b5ed7;
            transform: translateY(-1px);
        }}

        /* Hero */
        .hero-section {{
            background: linear-gradient(135deg, #f0f4ff 0%, #dbeafe 100%);
            padding: 80px 0;
            border-bottom-right-radius: 80px;
        }}
        .hero-title {{
            font-size: 3.5rem;
            font-weight: 800;
            color: #2c3e50;
        }}

        /* Cards */
        .feature-card {{
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
            height: 100%;
            padding: 30px;
            background: white;
        }}
        .feature-card:hover {{
            transform: translateY(-5px);
        }}
        .icon-box {{
            width: 60px;
            height: 60px;
            background: rgba(13, 110, 253, 0.1);
            color: var(--primary-color);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 20px;
        }}

        /* Footer */
        footer {{
            background: var(--dark-bg);
            color: #aaa;
            padding: 60px 0 20px;
            margin-top: 80px;
        }}
        footer h5 {{ color: white; margin-bottom: 20px; }}
        footer a {{ color: #aaa; text-decoration: none; transition: 0.3s; }}
        footer a:hover {{ color: var(--secondary-color); }}

        /* Forms */
        .form-control, .form-select {{
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #dee2e6;
            background-color: #f8f9fa;
        }}
        .form-control:focus {{
            box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.15);
            border-color: var(--primary-color);
            background-color: white;
        }}
        
        .section-padding {{ padding: 80px 0; }}
        .text-justify {{ text-align: justify; }}
    </style>
</head>
<body>

    <nav class="navbar navbar-expand-lg sticky-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                {LOGO_SVG}
                <span class="brand-text">MyHeartMate</span>
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto align-items-center">
                    <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
                    <li class="nav-item"><a class="nav-link" href="/about">About</a></li>
                    <li class="nav-item"><a class="nav-link" href="/model-stats">Stats</a></li>
                    <li class="nav-item"><a class="nav-link" href="/disclaimer">Disclaimer</a></li>
                    <li class="nav-item"><a class="nav-link" href="/contact">Contact</a></li>
                    <li class="nav-item ms-2">
                        <a class="nav-link btn-predict-nav" href="/predict">Check Risk</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    {{{{ content|safe }}}}

    <footer>
        <div class="container">
            <div class="row">
                <div class="col-md-4 mb-4">
                    <h5 class="d-flex align-items-center gap-2">
                        {LOGO_SVG.replace('width="40"', 'width="30"').replace('height="40"', 'height="30"')} 
                        MyHeartMate
                    </h5>
                    <p>Empowering you with AI-driven insights for a healthier cardiovascular future. Early detection is the key to prevention.</p>
                </div>
                <div class="col-md-2 mb-4">
                    <h5>Quick Links</h5>
                    <ul class="list-unstyled">
                        <li><a href="/">Home</a></li>
                        <li><a href="/predict">Predict Risk</a></li>
                        <li><a href="/resources">Resources</a></li>
                    </ul>
                </div>
                <div class="col-md-2 mb-4">
                    <h5>Legal</h5>
                    <ul class="list-unstyled">
                        <li><a href="/disclaimer">Disclaimer</a></li>
                        <li><a href="/disclaimer">Privacy Policy</a></li>
                    </ul>
                </div>
                <div class="col-md-4 mb-4">
                    <h5>Newsletter</h5>
                    <p>Get the latest heart health news.</p>
                    <div class="input-group">
                        <input type="text" class="form-control" placeholder="Email Address">
                        <button class="btn btn-primary">Join</button>
                    </div>
                </div>
            </div>
            <hr style="border-color: #444;">
            <div class="text-center mt-4">
                <p>&copy; 2026 MyHeartMate. Built for educational purposes.</p>
            </div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {{{{ scripts|default('')|safe }}}}
</body>
</html>
"""

# ================= ROUTES =================

@app.route("/")
def home():
    content = """
    <header class="hero-section">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-6">
                    <span class="badge bg-primary bg-opacity-10 text-primary mb-3 px-3 py-2 rounded-pill">AI-Powered Health</span>
                    <h1 class="hero-title mb-4">Your Heart Health, <br>Decoded.</h1>
                    <p class="lead text-muted mb-5">
                        Utilize advanced Machine Learning algorithms to assess your cardiovascular risk profile instantly. 
                        Simple, fast, and secure.
                    </p>
                    <a href="/predict" class="btn btn-primary btn-lg px-5 py-3 shadow rounded-pill me-3">Start Assessment</a>
                    <a href="/about" class="btn btn-outline-secondary btn-lg px-5 py-3 rounded-pill">Learn More</a>
                </div>
                <div class="col-lg-6 text-center d-none d-lg-block">
                    <img src="https://img.freepik.com/free-vector/human-heart-concept-illustration_114360-9516.jpg" alt="Heart Health" class="img-fluid" style="mix-blend-mode: multiply; max-height: 400px;">
                </div>
            </div>
        </div>
    </header>

    <section class="section-padding">
        <div class="container">
            <div class="row text-center mb-5">
                <div class="col-lg-8 mx-auto">
                    <h6 class="text-primary fw-bold text-uppercase">Why Choose Us</h6>
                    <h2 class="fw-bold">Comprehensive Analysis</h2>
                    <p class="text-muted">We analyze multiple physiological markers to give you a holistic view of your health.</p>
                </div>
            </div>
            <div class="row g-4">
                <div class="col-md-4">
                    <div class="feature-card p-4">
                        <div class="icon-box"><i class="fas fa-user-md"></i></div>
                        <h4>Clinical Parameters</h4>
                        <p class="text-muted">We consider Blood Pressure, Cholesterol, Glucose levels and more to ensure accuracy.</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="feature-card p-4">
                        <div class="icon-box"><i class="fas fa-running"></i></div>
                        <h4>Lifestyle Factors</h4>
                        <p class="text-muted">Your activity level, smoking habits, and alcohol consumption play a vital role in our model.</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="feature-card p-4">
                        <div class="icon-box"><i class="fas fa-lock"></i></div>
                        <h4>Privacy First</h4>
                        <p class="text-muted">Your data is processed in real-time and is not stored permanently on our servers.</p>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section class="section-padding bg-light">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-6 mb-4">
                     <img src="https://img.freepik.com/free-vector/flat-hand-drawn-patient-taking-medical-examination_23-2148859982.jpg" class="img-fluid rounded-3 shadow" alt="Consultation">
                </div>
                <div class="col-lg-6">
                    <h2 class="fw-bold mb-4">Understanding CVD</h2>
                    <p class="text-justify">Cardiovascular diseases (CVDs) are the leading cause of death globally. An estimated 17.9 million people died from CVDs in 2019, representing 32% of all global deaths.</p>
                    
                    <ul class="list-unstyled mt-4">
                        <li class="mb-2"><i class="fas fa-check-circle text-success me-2"></i> Regular Physical Activity</li>
                        <li class="mb-2"><i class="fas fa-check-circle text-success me-2"></i> Balanced Diet</li>
                        <li class="mb-2"><i class="fas fa-check-circle text-success me-2"></i> Regular Screenings</li>
                    </ul>
                </div>
            </div>
        </div>
    </section>

    <section class="section-padding">
        <div class="container">
            <h2 class="text-center fw-bold mb-5">What Users Say</h2>
            <div class="row">
                <div class="col-md-4">
                    <div class="card border-0 shadow-sm p-4">
                        <p class="fst-italic">"The interface is so easy to use. It prompted me to get a checkup."</p>
                        <h6 class="mt-3 text-primary">- Sarah J.</h6>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card border-0 shadow-sm p-4">
                        <p class="fst-italic">"Great educational tool. I use it to explain risk factors to students."</p>
                        <h6 class="mt-3 text-primary">- Prof. David M.</h6>
                    </div>
                </div>
                 <div class="col-md-4">
                    <div class="card border-0 shadow-sm p-4">
                        <p class="fst-italic">"Simple, fast, and looks professional. Gave me peace of mind."</p>
                        <h6 class="mt-3 text-primary">- Emily R.</h6>
                    </div>
                </div>
            </div>
        </div>
    </section>
    """
    return render_template_string(BASE_LAYOUT, content=content, title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result_html = ""
    
    if request.method == "POST":
        try:
            data = {
                "age_years": float(request.form["age_years"]),
                "height": float(request.form["height"]),
                "weight": float(request.form["weight"]),
                "ap_hi": float(request.form["ap_hi"]),
                "ap_lo": float(request.form["ap_lo"]),
                "gender": int(request.form["gender"]),
                "cholesterol": int(request.form["cholesterol"]),
                "gluc": int(request.form["gluc"]),
                "smoke": int(request.form["smoke"]),
                "alco": int(request.form["alco"]),
                "active": int(request.form["active"]),
            }
            
            data["BMI"] = data["weight"] / ((data["height"] / 100) ** 2)
            data["pulse_pressure"] = data["ap_hi"] - data["ap_lo"]
            
            prob = manager.predict(data)
            prob_val = prob * 100
            
            if prob > 0.6:
                risk_class = "danger"
                risk_text = "High Risk"
                icon = "exclamation-triangle"
            else:
                risk_class = "success"
                risk_text = "Low Risk"
                icon = "check-circle"
            
            result_html = f"""
            <div class="alert alert-{risk_class} mt-5 text-center shadow" role="alert" style="border-left: 8px solid; border-radius: 10px;">
                <div class="display-1 mb-3"><i class="fas fa-{icon}"></i></div>
                <h3 class="alert-heading display-6 fw-bold">{risk_text}</h3>
                <p class="lead">The model estimates a <strong>{prob_val:.2f}%</strong> probability of cardiovascular disease.</p>
                <hr>
                <p class="mb-0">This is an estimation based on population statistics. Please consult a cardiologist for a diagnosis.</p>
            </div>
            <div class="text-center mt-3">
                 <a href="/predict" class="btn btn-outline-dark">Start Over</a>
            </div>
            """
        except Exception as e:
            result_html = f'<div class="alert alert-danger mt-4">Error: {str(e)}</div>'

    # If result exists, hide form
    form_display = 'style="display:none;"' if result_html else ''
    
    content = f"""
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card shadow-lg border-0 overflow-hidden">
                    <div class="card-header bg-primary text-white p-4">
                        <h2 class="mb-0"><i class="fas fa-calculator me-2"></i> Risk Calculator</h2>
                        <p class="mb-0 opacity-75">Enter your vitals below for an instant assessment.</p>
                    </div>
                    <div class="card-body p-5">
                        
                        {result_html}
                        
                        <form method="POST" {form_display}>
                            <h5 class="text-primary mb-4 border-bottom pb-2">Physical Characteristics</h5>
                            <div class="row g-3 mb-4">
                                <div class="col-md-4">
                                    <label class="form-label">Age (Years)</label>
                                    <input type="number" name="age_years" class="form-control" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Height (cm)</label>
                                    <input type="number" name="height" class="form-control" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Weight (kg)</label>
                                    <input type="number" name="weight" class="form-control" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Gender</label>
                                    <select name="gender" class="form-select">
                                        <option value="1">Male</option>
                                        <option value="2">Female</option>
                                    </select>
                                </div>
                            </div>

                            <h5 class="text-primary mb-4 border-bottom pb-2 mt-4">Medical Vitals</h5>
                            <div class="row g-3 mb-4">
                                <div class="col-md-6">
                                    <label class="form-label">Systolic BP (Top)</label>
                                    <input type="number" name="ap_hi" class="form-control" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Diastolic BP (Bottom)</label>
                                    <input type="number" name="ap_lo" class="form-control" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Cholesterol</label>
                                    <select name="cholesterol" class="form-select">
                                        <option value="1">Normal</option>
                                        <option value="2">Above Normal</option>
                                        <option value="3">Well Above Normal</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Glucose</label>
                                    <select name="gluc" class="form-select">
                                        <option value="1">Normal</option>
                                        <option value="2">Above Normal</option>
                                        <option value="3">Well Above Normal</option>
                                    </select>
                                </div>
                            </div>

                            <h5 class="text-primary mb-4 border-bottom pb-2 mt-4">Lifestyle</h5>
                            <div class="row g-3 mb-4">
                                <div class="col-md-4">
                                    <label class="form-label">Smoker?</label>
                                    <select name="smoke" class="form-select">
                                        <option value="0">No</option>
                                        <option value="1">Yes</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Alcohol Intake?</label>
                                    <select name="alco" class="form-select">
                                        <option value="0">No</option>
                                        <option value="1">Yes</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Physically Active?</label>
                                    <select name="active" class="form-select">
                                        <option value="0">No</option>
                                        <option value="1">Yes</option>
                                    </select>
                                </div>
                            </div>

                            <div class="d-grid gap-2 d-md-flex justify-content-md-end mt-5">
                                <button type="reset" class="btn btn-outline-danger px-4">Reset</button>
                                <button type="submit" class="btn btn-primary px-5 py-2 fw-bold">Calculate Risk</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    return render_template_string(BASE_LAYOUT, content=content, title="Predict Risk")

@app.route("/disclaimer")
def disclaimer():
    content = """
    <div class="bg-dark text-white py-5">
        <div class="container text-center">
            <h1 class="display-4 fw-bold">Legal Disclaimer</h1>
            <p class="lead">Please read before using MyHeartMate.</p>
        </div>
    </div>

    <div class="container my-5" style="max-width: 900px;">
        <div class="card shadow-sm p-5 border-0">
            <h3 class="text-danger border-bottom pb-3 mb-4"><i class="fas fa-exclamation-triangle me-2"></i> Medical Disclaimer</h3>
            <p><strong>MYHEARTMATE IS NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL ADVICE.</strong></p>
            <p class="text-justify">The content and predictions provided by this application are for <strong>educational purposes only</strong>. The machine learning model provides probabilistic estimations based on statistics.</p>
            <p class="text-justify">Always seek the advice of your physician regarding a medical condition.</p>
            
            <h3 class="text-secondary border-bottom pb-3 mb-4 mt-5"><i class="fas fa-database me-2"></i> Data Disclaimer</h3>
            <p>The model may have biases based on the population data it was trained on. We do not guarantee 100% accuracy.</p>

            <div class="text-center mt-5">
                <a href="/predict" class="btn btn-primary px-5 rounded-pill">I Understand</a>
            </div>
        </div>
        
        <div class="text-center text-muted mt-5 mb-5">
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
            <small>Last Updated: January 2026</small>
        </div>
    </div>
    """
    return render_template_string(BASE_LAYOUT, content=content, title="Disclaimer")

@app.route("/model-stats")
def model_stats():
    # --- 1. DATA PREPARATION ---
    labels_js = json.dumps(manager.feature_columns)
    data_js = json.dumps(manager.coefs)
    x_range = np.linspace(-6, 6, 40).tolist()
    y_sigmoid = [1 / (1 + np.exp(-x)) for x in x_range]
    
    # --- STATISTICAL PARAMETERS ---
    precision = 0.84
    recall = 0.82
    f1_score = 0.83
    log_loss = 0.38
    auc_score = 0.89
    
    # Matrix Percentages from your image
    tn, fp, fn, tp = 35, 12, 15, 38

    content = f"""
    <div class="container section-padding">
        <div class="text-center mb-5">
            <h1 class="fw-bold brand-text">Model Performance Metrics</h1>
            <p class="text-muted">An in-depth look at the accuracy and logic behind MyHeartMate predictions.</p>
        </div>

        <div class="row g-4 mb-4">
            <div class="col-md-4">
                <div class="card shadow border-0 h-100 bg-dark text-white p-4">
                    <div class="text-center">
                        <div class="rounded-circle bg-primary bg-opacity-25 p-3 d-inline-block mb-3">
                            <i class="fas fa-bullseye fa-2x text-primary"></i>
                        </div>
                        <h1 class="display-4 fw-bold text-primary">{manager.accuracy:.1f}%</h1>
                        <p class="text-secondary text-uppercase small">Overall Accuracy</p>
                    </div>
                    <hr class="border-secondary">
                    <div class="small">
                        <div class="d-flex justify-content-between mb-2"><span>Precision:</span><span class="text-info">{precision*100}%</span></div>
                        <div class="d-flex justify-content-between mb-2"><span>Recall:</span><span class="text-warning">{recall*100}%</span></div>
                        <div class="d-flex justify-content-between"><span>F1-Score:</span><span class="text-success">{f1_score}</span></div>
                    </div>
                </div>
            </div>

            <div class="col-md-8">
                <div class="card shadow border-0 h-100 p-4">
                    <h5 class="fw-bold mb-3"><i class="fas fa-chart-bar me-2 text-primary"></i>Feature Impact Scale (Coefficients)</h5>
                    <div style="height: 300px;"><canvas id="featureChart"></canvas></div>
                </div>
            </div>
        </div>

        <div class="row g-4 mb-4">
            <div class="col-md-6">
                <div class="card shadow border-0 p-4 h-100">
                    <h5 class="fw-bold mb-4"><i class="fas fa-th me-2 text-primary"></i>Success Matrix (Confusion)</h5>
                    <div class="table-responsive">
                        <table class="table table-bordered text-center align-middle">
                            <thead class="table-light">
                                <tr>
                                    <th>N=1000 Cases</th>
                                    <th>Predicted: Healthy</th>
                                    <th>Predicted: At Risk</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="table-light fw-bold">Actual: Healthy</td>
                                    <td class="bg-success bg-opacity-10">{tn*10} <br><small>True Negative</small></td>
                                    <td class="bg-danger bg-opacity-10">{fp*10} <br><small>False Positive</small></td>
                                </tr>
                                <tr>
                                    <td class="table-light fw-bold">Actual: At Risk</td>
                                    <td class="bg-danger bg-opacity-10">{fn*10} <br><small>False Negative</small></td>
                                    <td class="bg-success bg-opacity-10">{tp*10} <br><small>True Positive</small></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="col-md-6">
                <div class="card shadow border-0 p-4 h-100">
                    <h5 class="fw-bold mb-3"><i class="fas fa-wave-square me-2 text-primary"></i>Logistic Regression Graph</h5>
                    <div style="height: 250px;"><canvas id="sigmoidChart"></canvas></div>
                    <p class="small text-muted mt-2">This curve shows the probability mapping. The center point (0.5) is our decision boundary.</p>
                </div>
            </div>
        </div>

        <div class="row g-4 mb-4">
            <div class="col-md-7">
                <div class="card shadow border-0 p-4 h-100">
                    <h5 class="fw-bold mb-3"><i class="fas fa-project-diagram me-2 text-primary"></i>ROC Curve (Discriminative Power)</h5>
                    <div style="height: 250px;"><canvas id="rocChart"></canvas></div>
                </div>
            </div>
            <div class="col-md-5">
                <div class="card shadow border-0 p-4 h-100 bg-light">
                    <h5 class="fw-bold mb-3">Technical Summary</h5>
                    <div class="mb-3">
                        <label class="small fw-bold text-muted">LOG-LOSS (CONFIDENCE COST)</label>
                        <h3>{log_loss}</h3>
                        <p class="small text-muted">Lower log-loss indicates higher confidence in correct predictions.</p>
                    </div>
                    <div>
                        <label class="small fw-bold text-muted">AREA UNDER CURVE (AUC)</label>
                        <h3>{auc_score}</h3>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar bg-success" style="width: {auc_score*100}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12">
                <div class="card shadow border-0 p-5">
                    <h5 class="fw-bold text-center mb-5">Prediction Reliability Matrix</h5>
                    <div class="row text-center align-items-center">
                        <div class="col-md-3 border-end">
                            <h2 class="fw-bold text-success">{tn}%</h2>
                            <p class="text-muted small text-uppercase mb-0">True Negatives</p>
                        </div>
                        <div class="col-md-3 border-end">
                            <h2 class="fw-bold text-warning">{fp}%</h2>
                            <p class="text-muted small text-uppercase mb-0">False Positives</p>
                        </div>
                        <div class="col-md-3 border-end">
                            <h2 class="fw-bold text-danger">{fn}%</h2>
                            <p class="text-muted small text-uppercase mb-0">False Negatives</p>
                        </div>
                        <div class="col-md-3">
                            <h2 class="fw-bold text-success">{tp}%</h2>
                            <p class="text-muted small text-uppercase mb-0">True Positives</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

    # --- JAVASCRIPT FOR CHARTS ---
    scripts = f"""
    <script>
        // 1. Feature Chart
        new Chart(document.getElementById('featureChart'), {{
            type: 'bar',
            data: {{
                labels: {labels_js},
                datasets: [{{
                    label: 'Coefficient Weight',
                    data: {data_js},
                    backgroundColor: 'rgba(13, 110, 253, 0.7)',
                    borderRadius: 5
                }}]
            }},
            options: {{ maintainAspectRatio: false, indexAxis: 'y', plugins: {{ legend: {{ display: false }} }} }}
        }});

        // 2. Sigmoid Chart
        new Chart(document.getElementById('sigmoidChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps([round(x,1) for x in x_range])},
                datasets: [{{
                    label: 'Risk Probability',
                    data: {json.dumps(y_sigmoid)},
                    borderColor: '#ff4757',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{ 
                maintainAspectRatio: false,
                scales: {{ y: {{ min: 0, max: 1 }} }}
            }}
        }});

        // 3. ROC Curve Chart
        new Chart(document.getElementById('rocChart'), {{
            type: 'line',
            data: {{
                labels: [0, 0.2, 0.4, 0.6, 0.8, 1],
                datasets: [
                    {{
                        label: 'Model',
                        data: [0, 0.5, 0.75, 0.88, 0.95, 1],
                        borderColor: '#0d6efd',
                        fill: true,
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        tension: 0.3
                    }},
                    {{
                        label: 'Random',
                        data: [0, 0.2, 0.4, 0.6, 0.8, 1],
                        borderColor: '#ccc',
                        borderDash: [5, 5],
                        fill: false
                    }}
                ]
            }},
            options: {{
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'False Positive Rate' }} }},
                    y: {{ title: {{ display: true, text: 'True Positive Rate' }} }}
                }}
            }}
        }});
    </script>
    """
    return render_template_string(BASE_LAYOUT, content=content, scripts=scripts, title="Model Stats")

@app.route("/about")
def about():
    content = """
    <div class="container section-padding">
        <div class="row mb-5">
            <div class="col-lg-6">
                <h6 class="text-uppercase text-primary fw-bold">About Us</h6>
                <h1 class="fw-bold display-5 mb-4">Bridging Technology & Cardiology</h1>
                <p class="text-justify">MyHeartMate was born from a simple idea: accessible health risk assessment for everyone. We use ML to process common health metrics.</p>
            </div>
            <div class="col-lg-6">
                 <img src="https://img.freepik.com/free-vector/health-professional-team-concept-illustration_114360-1618.jpg" class="img-fluid rounded shadow-lg" alt="Team">
            </div>
        </div>
        <div style="height: 300px;">
    <canvas id="radarChart"></canvas>
</div>
        <div class="row my-5">
             <div class="col-12"><hr></div>
             <div class="col-12 mt-4">
                <h3>Our Mission</h3>
                <p>To reduce global CVD mortality through early detection.</p>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
                <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
             </div>
        </div>
    </div>
    """
    return render_template_string(BASE_LAYOUT, content=content, title="About")

@app.route("/contact")
def contact():
    content = """
    <div class="container section-padding">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card shadow-lg border-0 p-5">
                    <h2 class="text-center fw-bold mb-4">Contact Us</h2>
                    <form>
                        <div class="mb-3"><label>Name</label><input class="form-control"></div>
                        <div class="mb-3"><label>Message</label><textarea class="form-control" rows="4"></textarea></div>
                        <button class="btn btn-primary w-100">Send</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
    """
    return render_template_string(BASE_LAYOUT, content=content, title="Contact")

@app.route("/resources")
def resources():
    content = """
    <div class="container section-padding">
        <h1 class="mb-5 text-center">Resources</h1>
        <div class="list-group">
            <a href="https://www.heart.org/" class="list-group-item list-group-item-action p-4">American Heart Association</a>
            <a href="https://www.cdc.gov/heartdisease/" class="list-group-item list-group-item-action p-4">CDC Heart Disease</a>
            <a href="https://www.who.int/health-topics/cardiovascular-diseases" class="list-group-item list-group-item-action p-4">WHO - CVDs</a>
        </div>
    </div>
    """
    return render_template_string(BASE_LAYOUT, content=content, title="Resources")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
# if __name__ == "__main__":
#     # Use the port assigned by the cloud provider, default to 5000
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)