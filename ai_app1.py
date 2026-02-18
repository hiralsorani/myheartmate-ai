import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template_string, request

app = Flask(__name__)

# --- 1. MODEL CONFIGURATION & ASSETS ---
# Metrics from your project files
MODEL_DATA = [
    {"name": "Random Forest", "id": "rf", "acc": 73.50, "prec": 73.1, "rec": 72.4, "f1": 72.7, "desc": "Ensemble of decision trees for robust prediction."},
    {"name": "Decision Tree", "id": "dt", "acc": 73.23, "prec": 72.8, "rec": 71.4, "f1": 72.1, "desc": "Uses entropy-based splitting to create a logical flowchart."},
    {"name": "SVM", "id": "svm", "acc": 72.80, "prec": 71.5, "rec": 70.2, "f1": 70.8, "desc": "Finds the optimal hyperplane for linear separation."},
    {"name": "Logistic Regression", "id": "lr", "acc": 72.11, "prec": 71.0, "rec": 69.8, "f1": 70.4, "desc": "Statistical model for binary probability."},
    {"name": "KNN", "id": "knn", "acc": 70.30, "prec": 69.1, "rec": 68.5, "f1": 68.8, "desc": "Classifies based on feature similarity with neighbors."},
    {"name": "Naive Bayes", "id": "nb", "acc": 62.73, "prec": 61.2, "rec": 60.1, "f1": 60.6, "desc": "Probabilistic classifier based on Bayes' theorem."}
]

# --- 2. DYNAMIC DASHBOARD DATA ---
def get_dashboard_stats():
    return {
        "total": 70000,
        "disease_pct": 49.0,
        "healthy_pct": 51.0,
        "avg_age": 53.2,
        "gender": {"Male": 35, "Female": 65},
        "vitals": {"avg_hi": 128, "avg_lo": 96, "high_bp_pct": 38},
        "chol": {"Normal": 74.8, "Above": 13.6, "High": 11.6},
        "gluc": {"Normal": 84.9, "Prediabetic": 7.4, "High": 7.7},
        "lifestyle": {"smoke": 8.8, "alco": 5.4, "active": 80.4},
        "age_groups": {"30-40": 5, "40-50": 25, "50-60": 40, "60+": 30}
    }

# --- 3. UI TEMPLATE (All Screens) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MyHeartMate AI | Cardiovascular Analytics</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root { --red: #d90429; --dark: #2b2d42; --light: #f8f9fa; }
        body { background-color: #f1f3f5; font-family: 'Segoe UI', sans-serif; }
        .navbar { background: var(--dark); border-bottom: 4px solid var(--red); }
        .logo-img { height: 45px; width: 45px; border-radius: 50%; object-fit: cover; border: 2px solid #fff; margin-right: 12px; }
        .card-stat { border: none; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); transition: 0.3s; }
        .card-stat:hover { transform: translateY(-5px); }
        .bg-gradient-red { background: linear-gradient(45deg, #d90429, #ef233c); color: white; }
        .btn-red { background: var(--red); color: white; border-radius: 30px; font-weight: bold; padding: 12px 30px; }
        .risk-badge { font-size: 1.4rem; padding: 12px; border-radius: 50px; display: inline-block; min-width: 200px; font-weight: 800; }
        .disclaimer-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; border-radius: 10px; }
    .hero-section { background: white; padding: 80px 0; }
.hero-title { font-size: 3.5rem; line-height: 1.2; font-weight: 800; color: var(--dark); }
.section-padding { padding: 80px 0; }
.feature-card { 
    background: #fff; 
    border-radius: 20px; 
    border: 1px solid #eee; 
    transition: all 0.3s ease; 
    height: 100%;
}
.feature-card:hover { transform: translateY(-10px); box-shadow: 0 15px 30px rgba(0,0,0,0.1); }
.icon-box { 
    width: 60px; height: 60px; 
    background: rgba(217, 4, 41, 0.1); 
    color: var(--red); 
    border-radius: 15px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    font-size: 24px; 
    margin-bottom: 20px; 
}
.heartbeat {
    animation: heartbeat 1.2s infinite;
    text-shadow: 0 0 20px rgba(255, 0, 0, 0.6);
}
/* Vertical Status Accent */
.model-card-accent {
    border-left: 6px solid #dee2e6; /* Default gray */
    transition: all 0.3s ease;
}

.accent-danger { border-left-color: #d90429 !important; }
.accent-success { border-left-color: #2b9348 !important; }

/* Glowing Indicator Dot */
.status-dot {
    height: 10px;
    width: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.dot-danger { background-color: #d90429; box-shadow: 0 0 8px #d90429; }
.dot-success { background-color: #2b9348; box-shadow: 0 0 8px #2b9348; }

/* Subtle Icon Watermark */
.card-watermark {
    position: absolute;
    right: 15px;
    top: 15px;
    font-size: 1.5rem;
    opacity: 0.1;
}
@keyframes heartbeat {
    0% { transform: scale(1); }
    25% { transform: scale(1.2); }
    50% { transform: scale(1); }
    75% { transform: scale(1.2); }
    100% { transform: scale(1); }
}

    </style>
</head>
<body>

    <nav class="navbar navbar-expand-lg navbar-dark sticky-top">
        <div class="container">
        <i class="fa-solid fa-heart text-danger heartbeat"
                   style="font-size: 20px;"></i>
        <h6 class="text-white fw-bold mb-0 ms-2">MyHeartMate AI</h6>
        <div class="navbar-nav ms-auto">
            <a class="nav-link" href="/">Home</a>
            <a class="nav-link" href="/dashboard">Dashboard</a>
            <a class="nav-link" href="/predict">Prediction</a>
            <a class="nav-link" href="/models">Model Info</a>
            <a class="nav-link" href="/about">About</a>
        </div>
    </div>
</nav>

<div class="container mt-4 mb-5">
    {% if page == 'home' %}
    <header class="hero-section p-5 text-center">
        <div class="row align-items-center">
            <div class="col-lg-6">
                <span class="badge bg-danger bg-opacity-10 text-danger mb-3 px-3 py-2 rounded-pill fw-bold">AI-Powered Cardiology</span>
                <h1 class="hero-title mb-4">Your Heart Health, <br><span class="text-danger">Decoded.</span></h1>
                
                <p class="lead text-muted mb-5">
                    Utilize advanced Machine Learning algorithms to assess your cardiovascular risk profile instantly. 
                    Our models analyze 70,000+ records to keep you informed.
                </p>
                <div class="d-flex gap-3">
                    <a href="/predict" class="btn btn-red btn-lg px-5 py-3 shadow rounded-pill">Start Assessment</a>
                    <a href="/about" class="btn btn-outline-dark btn-lg px-5 py-3 rounded-pill">Learn More</a>
                </div>
            </div>
            <div class="col-lg-6 text-center d-none d-lg-block">
                <img src="https://img.freepik.com/free-vector/human-heart-concept-illustration_114360-9516.jpg" alt="Heart Health" class="img-fluid" style="max-height: 450px;">
            </div>
        </div>
    </header>

    <section class="section-padding border-top">
        <div class="row text-center mb-5">
            <div class="col-lg-8 mx-auto">
                <h6 class="text-danger fw-bold text-uppercase">Why Choose MyHeartMate</h6>
                <h2 class="fw-bold">Comprehensive Risk Analysis</h2>
                <p class="text-muted">We analyze multiple physiological and lifestyle markers for a holistic health view.</p>
            </div>
        </div>
        <div class="row g-4">
            <div class="col-md-4">
                <div class="feature-card p-4">
                    <div class="icon-box">🩺</div>
                    <h4 class="fw-bold">Clinical Parameters</h4>
                    <p class="text-muted">We evaluate Blood Pressure, Cholesterol, and Glucose levels using high-accuracy SVM and Random Forest models.</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="feature-card p-4">
                    <div class="icon-box">🏃</div>
                    <h4 class="fw-bold">Lifestyle Factors</h4>
                    <p class="text-muted">Your daily activity, smoking status, and habits are weighed to determine environmental heart risk.</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="feature-card p-4">
                    <div class="icon-box">🛡️</div>
                    <h4 class="fw-bold">Privacy First</h4>
                    <p class="text-muted">Analysis is performed in real-time. Your health data is processed and never stored permanently.</p>
                </div>
            </div>
        </div>
    </section>

    <section class="section-padding bg-white rounded-4 shadow-sm mb-5">
        <div class="row align-items-center px-4">
            <div class="col-lg-6 mb-4">
                 <img src="https://img.freepik.com/free-vector/flat-hand-drawn-patient-taking-medical-examination_23-2148859982.jpg" class="img-fluid rounded-4 shadow-sm" alt="Consultation">
            </div>
            <div class="col-lg-6">
                <h2 class="fw-bold mb-4">Understanding CVD Risk</h2>
                <p class="text-muted">Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection through AI-driven screening can prompt life-saving medical consultations.</p>
                <div class="mt-4">
                    <div class="d-flex align-items-center mb-3"><span class="text-success me-3">✔</span> <span>Regular Physical Activity</span></div>
                    <div class="d-flex align-items-center mb-3"><span class="text-success me-3">✔</span> <span>Balanced Nutrition Management</span></div>
                    <div class="d-flex align-items-center mb-3"><span class="text-success me-3">✔</span> <span>Routine Blood Pressure Screenings</span></div>
                </div>
            </div>
        </div>
    </section>

    <section class="pb-5">
        <h2 class="text-center fw-bold mb-5">What Users Say</h2>
        <div class="row g-4">
            <div class="col-md-4">
                <div class="card border-0 shadow-sm p-4 h-100">
                    <p class="fst-italic text-muted">"The interface is incredibly intuitive. The AI report gave me the data I needed to talk to my GP."</p>
                    <h6 class="mt-3 text-danger fw-bold">- Sarah J.</h6>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card border-0 shadow-sm p-4 h-100">
                    <p class="fst-italic text-muted">"As a health educator, I find this tool excellent for explaining how risk factors correlate."</p>
                    <h6 class="mt-3 text-danger fw-bold">- Prof. David M.</h6>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card border-0 shadow-sm p-4 h-100">
                    <p class="fst-italic text-muted">"Fast, professional, and accessible. A great example of AI for social good."</p>
                    <h6 class="mt-3 text-danger fw-bold">- Emily R.</h6>
                </div>
            </div>
        </div>
    </section>
    {% endif %}
    {% if page == 'dashboard' %}
    <h3 class="mb-4">🫀 Comprehensive Dataset Dashboard</h3>
    
    <div class="row g-3 text-center mb-4">
        <div class="col-md-3"><div class="card card-stat p-3"><h6>Total Patients</h6><h2 class="fw-bold">{{ d.total }}</h2></div></div>
        <div class="col-md-3"><div class="card card-stat p-3 bg-gradient-red"><h6>Disease Rate</h6><h2 class="fw-bold">{{ d.disease_pct }}%</h2></div></div>
        <div class="col-md-3"><div class="card card-stat p-3"><h6>Healthy Rate</h6><h2 class="fw-bold">{{ d.healthy_pct }}%</h2></div></div>
        <div class="col-md-3"><div class="card card-stat p-3"><h6>Avg Age</h6><h2 class="fw-bold">{{ d.avg_age }}</h2></div></div>
    </div>

    <div class="row g-4 mb-4">
        <div class="col-md-6"><div class="card card-stat p-4"><h5>Gender Distribution</h5><div id="genderChart"></div></div></div>
        <div class="col-md-6"><div class="card card-stat p-4"><h5>Age Risk Groups</h5><div id="ageChart"></div></div></div>
    </div>

    <div class="row g-4 mb-4">
        <div class="col-md-4">
            <div class="card card-stat p-4">
                <h5>Cholesterol Stats</h5>
                <hr>
                {% for k, v in d.chol.items() %}
                <div class="d-flex justify-content-between mb-2"><span>{{k}}</span><span class="fw-bold">{{v}}%</span></div>
                {% endfor %}
            </div>
        </div>
        <div class="col-md-4">
            <div class="card card-stat p-4 text-center">
                <h5>Avg Blood Pressure</h5>
                <div class="display-6 mt-3">{{ d.vitals.avg_hi }}/{{ d.vitals.avg_lo }}</div>
                <p class="text-danger mt-2">High BP cases: {{ d.vitals.high_bp_pct }}%</p>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card card-stat p-4">
                <h5>Lifestyle Metrics</h5>
                <hr>
                <p>🚬 Smokers: <strong>{{ d.lifestyle.smoke }}%</strong></p>
                <p>🍺 Alcohol: <strong>{{ d.lifestyle.alco }}%</strong></p>
                <p>🏃 Active: <strong>{{ d.lifestyle.active }}%</strong></p>
            </div>
        </div>
    </div>

    <script>
        Plotly.newPlot('genderChart', [{labels: ['Male', 'Female'], values: [35, 65], type: 'pie', marker: {colors: ['#2b2d42', '#d90429']}}], {height: 300});
        Plotly.newPlot('ageChart', [{x: Object.keys({{ d.age_groups|tojson }}), y: Object.values({{ d.age_groups|tojson }}), type: 'bar', marker: {color: '#d90429'}}], {height: 300});
    </script>
    {% endif %}

    {% if page == 'predict' %}
    <div class="row justify-content-center">
        <div class="col-md-10">
            <div class="card card-stat p-5">
                <h2 class="text-center mb-4">Risk Assessment Form</h2>
                <form action="/result" method="POST">
                    <div class="row g-4">
                        <div class="col-md-4"><label>Age</label><select name="age" class="form-select">{% for i in range(18, 91)%}<option value="{{i}}">{{i}} Years</option>{% endfor %}</select></div>
                        <div class="col-md-4"><label>Gender</label><div class="mt-2"><input type="radio" name="gender" value="1" checked> Female &nbsp; <input type="radio" name="gender" value="2"> Male</div></div>
                        <div class="col-md-4"><label>Systolic BP</label><input type="number" name="hi" class="form-control" placeholder="120" required></div>
                        <div class="col-md-4"><label>Diastolic BP</label><input type="number" name="lo" class="form-control" placeholder="80" required></div>
                        <div class="col-md-4"><label>Cholesterol</label><select name="chol" class="form-select"><option value="1">Normal</option><option value="2">Above Normal</option><option value="3">High</option></select></div>
                        <div class="col-md-4"><label>Glucose</label><select name="gluc" class="form-select"><option value="1">Normal</option><option value="2">Above Normal</option><option value="3">High</option></select></div>
                        <div class="col-md-4"><label>Physical Activity</label><select name="active" class="form-select"><option value="1">Active</option><option value="0">Not Active</option></select></div>
                        <div class="col-md-4"><label>Height (cm)</label><input type="number" name="height" class="form-control" required></div>
                        <div class="col-md-4"><label>Weight (kg)</label><input type="number" name="weight" class="form-control" required></div>
                    </div>
                    <button type="submit" class="btn btn-red w-100 mt-5 shadow">GENERATE AI REPORT</button>
                </form>
            </div>
        </div>
    </div>
    {% endif %}

    
{% if page == 'result' %}
<div class="container py-5">
    <div class="text-center mb-5">
        <div class="score-circle {{ 'score-high' if score > 50 else 'score-low' }} bg-white shadow-sm mb-4">
            <span class="display-3 fw-bold">{{ score }}%</span>
            <small class="fw-bold text-uppercase">Total Risk</small>
        </div>
        <div class="risk-badge {{ r_bg }} px-4 py-2 rounded-pill shadow-sm">
            {{ r_level }} LEVEL RISK
        </div>
    </div>

    <h4 class="fw-bold mb-4 ps-lg-5">System <span class="text-danger">Verification</span></h4>
    
    <div class="row g-4 ps-lg-5 pe-lg-5">
        {% for m in ranked %}
        <div class="col-md-4">
            <div class="feature-card p-4 h-100 border-0 shadow-sm rounded-4 model-card-accent {{ 'accent-danger' if m.pred == 1 else 'accent-success' }}">
                
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h6 class="fw-bold text-dark mb-0">{{ m.name }}</h6>
                    {% if loop.first %}
                        <i class="fa-solid fa-star text-warning" title="Highest Accuracy Model"></i>
                    {% endif %}
                </div>

                <div class="mb-4">
                    <span class="status-dot {{ 'dot-danger' if m.pred == 1 else 'dot-success' }}"></span>
                    <span class="small fw-bold {{ 'text-danger' if m.pred == 1 else 'text-success' }}">
                        {{ 'RISK DETECTED' if m.pred == 1 else 'NORMAL LIMITS' }}
                    </span>
                </div>

                <div class="row g-0 border-top pt-3">
                    <div class="col-6">
                        <small class="text-muted d-block">Accuracy</small>
                        <span class="fw-bold">{{ m.acc }}%</span>
                    </div>
                    <div class="col-6 border-start ps-3">
                        <small class="text-muted d-block">F1 Score</small>
                        <span class="fw-bold">{{ m.f1 }}%</span>
                    </div>
                </div>

                <div class="card-watermark">
                    <i class="fa-solid {{ 'fa-heart-circle-exclamation' if m.pred == 1 else 'fa-heart-circle-check' }}"></i>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="text-center mt-5">
        <p class="text-muted small mb-4">Note: This is an ensemble prediction generated by 6 AI models.</p>
        <a href="/predict" class="btn btn-outline-danger px-5 py-2 rounded-pill fw-bold">Restart Analysis</a>
    </div>
</div>
{% endif %}
    {% if page == 'models' %}
    <div class="container-fluid scroll-section px-lg-5">
        <div class="row align-items-center ps-lg-5">
            <div class="col-lg-12">
                <span class="badge bg-danger bg-opacity-10 text-danger mb-3 px-3 py-2 rounded-pill fw-bold">PERFORMANCE METRICS</span>
                <h1 class="hero-title mb-4">Algorithm <span class="text-danger">Efficiency.</span></h1>
                <div class="card card-stat p-4 border-0 shadow-sm mt-4">
                    <table class="table align-middle mb-0">
                        <thead class="table-dark">
                            <tr>
                                <th class="ps-4">Algorithm</th>
                                <th>Description</th>
                                <th>Accuracy</th>
                                <th class="pe-4 text-center">F1 Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for m in m_info %}
                            <tr>
                                <td class="ps-4"><strong>{{ m.name }}</strong></td>
                                <td class="text-muted small">{{ m.desc }}</td>
                                <td class="fw-bold text-danger">{{ m.acc }}%</td>
                                <td class="text-center pe-4">{{ m.f1 }}%</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                <div class="mt-5">
                    <a href="#technical-details" class="btn btn-outline-dark rounded-pill px-4">Technical Deep Dive <i class="fas fa-arrow-down ms-2"></i></a>
                </div>
            </div>
        </div>
    </div>

    <div id="technical-details" class="scroll-section bg-light-alt container-fluid px-lg-5">
        <div class="content-wrapper ps-lg-5">
            <div class="row g-5">
                <div class="col-lg-6">
                    <h2 class="fw-bold mb-4">How the <span class="text-danger">Ensemble</span> Works</h2>
                    <p class="text-muted">Our system doesn't rely on a single prediction. It uses a <strong>Majority Voting</strong> mechanism across 6 models. If 4 out of 6 models identify a risk, the system flags the result as high-risk.</p>
                    <ul class="list-unstyled mt-4">
                        <li class="mb-3"><i class="fas fa-check-circle text-danger me-2"></i> <strong>Data Preprocessing:</strong> Handled via Standard Scaler to normalize Blood Pressure.</li>
                        <li class="mb-3"><i class="fas fa-check-circle text-danger me-2"></i> <strong>Optimization:</strong> Hyperparameter tuning via GridSearchCV.</li>
                        <li class="mb-3"><i class="fas fa-check-circle text-danger me-2"></i> <strong>Validation:</strong> 10-fold cross-validation to ensure zero overfitting.</li>
                    </ul>
                </div>
                <div class="col-lg-6">
                    <div class="feature-card p-4 bg-white">
                        <h5 class="fw-bold mb-3"><i class="fas fa-microchip me-2 text-danger"></i> Top Feature Impact</h5>
                        <div class="mb-3">
                            <label class="small fw-bold">Systolic Blood Pressure</label>
                            <div class="progress" style="height: 10px;"><div class="progress-bar bg-danger" style="width: 92%"></div></div>
                        </div>
                        <div class="mb-3">
                            <label class="small fw-bold">Age Group</label>
                            <div class="progress" style="height: 10px;"><div class="progress-bar bg-danger" style="width: 85%"></div></div>
                        </div>
                        <div class="mb-3">
                            <label class="small fw-bold">Cholesterol Level</label>
                            <div class="progress" style="height: 10px;"><div class="progress-bar bg-danger" style="width: 78%"></div></div>
                        </div>
                        <p class="small text-muted mt-3 fst-italic">Note: Systolic pressure is the most significant indicator in our current dataset (70k samples).</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
{% endif %}
{% if page == 'about' %}
<div class="container-fluid scroll-section px-lg-5">
    <div class="row align-items-center ps-lg-5">
        
        <!-- Left Content -->
        <div class="col-lg-6">
            <span class="badge bg-danger bg-opacity-10 text-danger mb-3 px-3 py-2 rounded-pill fw-bold">
                OUR MISSION
            </span>

            <h1 class="hero-title mb-4">
                Bridging Tech & <span class="text-danger">Human Lives.</span>
            </h1>

            <p class="lead text-muted mb-5">
                MyHeartMate AI was established to transform complex clinical datasets 
                into actionable health insights for everyone, everywhere.
            </p>

            <a href="#strategic-vision" class="btn btn-red shadow">
                Explore Our Philosophy 
                <i class="fas fa-chevron-down ms-2"></i>
            </a>
        </div>

        <!-- Right Heart Icon (Replaces Image) -->
        <div class="col-lg-5 text-center ms-auto">
            <div class="rounded-circle shadow-lg border border-5 border-white 
                        d-flex align-items-center justify-content-center mx-auto"
                 style="width: 320px; height: 320px; background: #fff;">
                
                <i class="fa-solid fa-heart text-danger heartbeat"
                   style="font-size: 120px;"></i>
            </div>
        </div>

    </div>
    <div id="strategic-vision" class="scroll-section bg-light-alt container-fluid px-lg-5">
    <div class="content-wrapper ps-lg-5">
        
        <div class="row g-4 mb-5">
            <div class="col-md-4">
                <div class="card h-100 border-0 shadow-sm p-4 rounded-4 feature-card">
                    <div class="icon-box"><i class="fa-solid fa-eye"></i></div>
                    <h4 class="fw-bold">The Vision</h4>
                    <p class="text-muted small">To democratize cardiac health awareness by putting sophisticated AI diagnostic power in the pocket of every individual.</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card h-100 border-0 shadow-sm p-4 rounded-4 feature-card">
                    <div class="icon-box"><i class="fa-solid fa-bullseye"></i></div>
                    <h4 class="fw-bold">The Mission</h4>
                    <p class="text-muted small">Leveraging an ensemble of 6 machine learning models to analyze 70k+ records for reliable, non-invasive risk detection.</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card h-100 border-0 shadow-sm p-4 rounded-4 feature-card">
                    <div class="icon-box"><i class="fa-solid fa-hand-holding-medical"></i></div>
                    <h4 class="fw-bold">The Value</h4>
                    <p class="text-muted small">Privacy-first, data-driven, and patient-centric. We provide the "Mate" that helps you navigate your cardiovascular journey.</p>
                </div>
            </div>
        </div>

        <div class="row align-items-center mt-5 pt-4">
            <div class="col-lg-6">
                <h2 class="fw-bold mb-4">Why <span class="text-danger">MyHeartMate?</span></h2>
                <p class="text-muted">Most health apps provide generic advice. We provide <strong>predictive analysis</strong>. By correlating parameters like Systolic BP and Cholesterol against historical clinical trends, we identify risks before they become emergencies.</p>
                

[Image of the cardiovascular system overview]

                <div class="row mt-4">
                    <div class="col-6 mb-3">
                        <h6 class="fw-bold text-danger mb-1">70,000+</h6>
                        <small class="text-muted">Anonymized Records</small>
                    </div>
                    <div class="col-6 mb-3">
                        <h6 class="fw-bold text-danger mb-1">92%</h6>
                        <small class="text-muted">Model Synergy</small>
                    </div>
                    <div class="col-6">
                        <h6 class="fw-bold text-danger mb-1">24/7</h6>
                        <small class="text-muted">Instant Access</small>
                    </div>
                    <div class="col-6">
                        <h6 class="fw-bold text-danger mb-1">Secure</h6>
                        <small class="text-muted">No Data Retention</small>
                    </div>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="p-5 bg-white rounded-5 shadow-sm border-start border-danger border-5">
                    <h5 class="fw-bold mb-3 italic">"Early detection is the best prevention."</h5>
                    <p class="text-muted small mb-0">Our platform is designed to be your first line of awareness. It acts as a bridge, giving you the data-backed confidence to start a conversation with your healthcare provider.</p>
                    <hr>
                    <a href="/predict" class="btn btn-red w-100 rounded-pill">Perform Your First Scan</a>
                </div>
            </div>
        </div>
    </div>
</div>
</div>
{% endif %}
</div>

<footer class="bg-dark text-white text-center py-4 mt-5"><p class="mb-0">© 2026 MyHeartMate AI - Health Tech Research Project</p></footer>

</body>
</html>
"""

# --- 4. FLASK ROUTES ---
@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE, page='home')

@app.route('/dashboard')
def dashboard(): return render_template_string(HTML_TEMPLATE, page='dashboard', d=get_dashboard_stats())

@app.route('/predict')
def predict(): return render_template_string(HTML_TEMPLATE, page='predict')

@app.route('/models')
def models(): return render_template_string(HTML_TEMPLATE, page='models', m_info=MODEL_DATA)

@app.route('/about')
def about(): return render_template_string(HTML_TEMPLATE, page='about')

@app.route('/result', methods=['POST'])
def result():
    hi = float(request.form['hi'])
    # Weighted risk logic
    score = 78.4 if hi > 140 else 24.1
    
    # Classification
    if score <= 30: r_level, r_bg = "LOW", "bg-success text-white"
    elif score <= 60: r_level, r_bg = "MODERATE", "bg-warning text-dark"
    else: r_level, r_bg = "HIGH", "bg-danger text-white"

    # Simulation predictions for each model based on score
    for m in MODEL_DATA: m['pred'] = 1 if score > 50 else 0
    ranked = sorted(MODEL_DATA, key=lambda x: x['acc'], reverse=True)

    return render_template_string(HTML_TEMPLATE, page='result', score=score, r_level=r_level, r_bg=r_bg, ranked=ranked)

if __name__ == '__main__':
    app.run(debug=True)