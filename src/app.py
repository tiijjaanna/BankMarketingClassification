"""
Deployment – Flask UI
======================
Pokretanje:
    python app.py

Otvori browser na: http://127.0.0.1:5000
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "tuned_model.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="sr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Marketing – Predikcija pretplate</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #F8FAFC;
               color: #1E293B; min-height: 100vh; padding: 2rem; }
        .container { max-width: 700px; margin: 0 auto; }
        h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.5rem; }
        p.subtitle { color: #64748B; margin-bottom: 2rem; font-size: 0.95rem; }
        .card { background: white; border-radius: 12px; padding: 2rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
        .card h2 { font-size: 0.9rem; font-weight: 600; color: #475569;
                   text-transform: uppercase; letter-spacing: 0.05em;
                   margin-bottom: 1.2rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .field { display: flex; flex-direction: column; gap: 0.3rem; }
        label { font-size: 0.85rem; font-weight: 500; color: #374151; }
        input, select {
            padding: 0.5rem 0.75rem; border: 1px solid #D1D5DB;
            border-radius: 8px; font-size: 0.9rem; color: #1E293B;
            background: white; width: 100%;
        }
        input:focus, select:focus {
            outline: none; border-color: #2563EB;
            box-shadow: 0 0 0 2px rgba(37,99,235,0.15);
        }
        button {
            width: 100%; padding: 0.85rem; background: #2563EB; color: white;
            border: none; border-radius: 10px; font-size: 1rem; font-weight: 600;
            cursor: pointer; margin-top: 0.5rem;
        }
        button:hover { background: #1D4ED8; }

        /* Rezultat */
        .result-yes {
            margin-top: 1.5rem; padding: 1.5rem; border-radius: 10px;
            background: #ECFDF5; border: 2px solid #10B981; text-align: center;
        }
        .result-no {
            margin-top: 1.5rem; padding: 1.5rem; border-radius: 10px;
            background: #FEF2F2; border: 2px solid #EF4444; text-align: center;
        }
        .result-yes h3 { font-size: 1.3rem; color: #065F46; margin-bottom: 0.4rem; }
        .result-no  h3 { font-size: 1.3rem; color: #991B1B; margin-bottom: 0.4rem; }
        .result-yes p, .result-no p { font-size: 0.95rem; color: #374151; }
    </style>
</head>
<body>
<div class="container">
    <h1>Bank Marketing – Predikcija pretplate</h1>
    <p class="subtitle">Unesite podatke o klijentu da predvidite da li ce se pretplatiti na oroceni depozit.</p>

    <form method="POST" action="/">

        <div class="card">
            <h2>Demografski podaci</h2>
            <div class="grid">
                <div class="field">
                    <label>Starost (age)</label>
                    <input type="number" name="age" value="{{ form.get('age', 35) }}" min="18" max="100">
                </div>
                <div class="field">
                    <label>Zanimanje (job)</label>
                    <select name="job">
                        {% for j in ['admin.','blue-collar','entrepreneur','housemaid',
                                     'management','retired','self-employed','services',
                                     'student','technician','unemployed','unknown'] %}
                        <option value="{{ j }}" {{ 'selected' if form.get('job') == j }}>{{ j }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label>Bracni status (marital)</label>
                    <select name="marital">
                        {% for m in ['married','single','divorced','unknown'] %}
                        <option value="{{ m }}" {{ 'selected' if form.get('marital') == m }}>{{ m }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label>Obrazovanje (education)</label>
                    <select name="education">
                        {% for e in ['university.degree','high.school','professional.course',
                                     'basic.9y','basic.6y','basic.4y','illiterate','unknown'] %}
                        <option value="{{ e }}" {{ 'selected' if form.get('education') == e }}>{{ e }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Finansijski podaci</h2>
            <div class="grid">
                <div class="field">
                    <label>Kreditno zaduzenje (default)</label>
                    <select name="default">
                        {% for d in ['no','yes','unknown'] %}
                        <option value="{{ d }}" {{ 'selected' if form.get('default') == d }}>{{ d }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label>Stambeni kredit (housing)</label>
                    <select name="housing">
                        {% for h in ['yes','no','unknown'] %}
                        <option value="{{ h }}" {{ 'selected' if form.get('housing') == h }}>{{ h }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label>Licni kredit (loan)</label>
                    <select name="loan">
                        {% for l in ['no','yes','unknown'] %}
                        <option value="{{ l }}" {{ 'selected' if form.get('loan') == l }}>{{ l }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Podaci o kampanji</h2>
            <div class="grid">
                <div class="field">
                    <label>Tip kontakta (contact)</label>
                    <select name="contact">
                        {% for c in ['cellular','telephone'] %}
                        <option value="{{ c }}" {{ 'selected' if form.get('contact') == c }}>{{ c }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label>Mesec (month)</label>
                    <select name="month">
                        {% for mo in ['jan','feb','mar','apr','may','jun',
                                      'jul','aug','sep','oct','nov','dec'] %}
                        <option value="{{ mo }}" {{ 'selected' if form.get('month') == mo }}>{{ mo }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label>Dan (day_of_week)</label>
                    <select name="day_of_week">
                        {% for day in ['mon','tue','wed','thu','fri'] %}
                        <option value="{{ day }}" {{ 'selected' if form.get('day_of_week') == day }}>{{ day }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label>Broj kontakata u kampanji (campaign)</label>
                    <input type="number" name="campaign" value="{{ form.get('campaign', 1) }}" min="1">
                </div>
                <div class="field">
                    <label>pdays (999 = nije prethodno kontaktiran)</label>
                    <input type="number" name="pdays" value="{{ form.get('pdays', 999) }}">
                </div>
                <div class="field">
                    <label>Broj prethodnih kontakata (previous)</label>
                    <input type="number" name="previous" value="{{ form.get('previous', 0) }}" min="0">
                </div>
                <div class="field">
                    <label>Ishod prethodne kampanje (poutcome)</label>
                    <select name="poutcome">
                        {% for po in ['nonexistent','success','failure'] %}
                        <option value="{{ po }}" {{ 'selected' if form.get('poutcome') == po }}>{{ po }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Makroekonomski pokazatelji</h2>
            <div class="grid">
                <div class="field">
                    <label>emp.var.rate</label>
                    <input type="number" name="emp_var_rate" value="{{ form.get('emp_var_rate', 1.1) }}" step="0.1">
                </div>
                <div class="field">
                    <label>cons.price.idx</label>
                    <input type="number" name="cons_price_idx" value="{{ form.get('cons_price_idx', 93.994) }}" step="0.001">
                </div>
                <div class="field">
                    <label>cons.conf.idx</label>
                    <input type="number" name="cons_conf_idx" value="{{ form.get('cons_conf_idx', -36.4) }}" step="0.1">
                </div>
                <div class="field">
                    <label>euribor3m</label>
                    <input type="number" name="euribor3m" value="{{ form.get('euribor3m', 4.857) }}" step="0.001">
                </div>
                <div class="field">
                    <label>nr.employed</label>
                    <input type="number" name="nr_employed" value="{{ form.get('nr_employed', 5191.0) }}" step="0.1">
                </div>
            </div>
        </div>

        <button type="submit">Predvidi pretplatu</button>
    </form>

    {% if result is not none %}
        {% if result.prediction == 1 %}
        <div class="result-yes">
            <h3>✅ Klijent CE se pretplatiti (yes)</h3>
            <p>Verovatnoca pretplate: <strong>{{ "%.1f"|format(result.probability * 100) }}%</strong></p>
        </div>
        {% else %}
        <div class="result-no">
            <h3>❌ Klijent NECE se pretplatiti (no)</h3>
            <p>Verovatnoca pretplate: <strong>{{ "%.1f"|format(result.probability * 100) }}%</strong></p>
        </div>
        {% endif %}
    {% endif %}

</div>
</body>
</html>
"""


def preprocess_input(form) -> pd.DataFrame:
    education_order = {
        'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3,
        'high.school': 4, 'professional.course': 5,
        'university.degree': 6, 'unknown': 7,
    }
    three_val = {'no': 0, 'yes': 1, 'unknown': 2}
    contact_map = {'cellular': 1, 'telephone': 0}
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}

    row = {
        'age':            int(form['age']),
        'education':      education_order.get(form['education'], 7),
        'default':        three_val.get(form['default'], 2),
        'housing':        three_val.get(form['housing'], 2),
        'loan':           three_val.get(form['loan'], 2),
        'contact':        contact_map.get(form['contact'], 1),
        'campaign':       int(form['campaign']),
        'pdays':          int(form['pdays']),
        'previous':       int(form['previous']),
        'emp.var.rate':   float(form['emp_var_rate']),
        'cons.price.idx': float(form['cons_price_idx']),
        'cons.conf.idx':  float(form['cons_conf_idx']),
        'euribor3m':      float(form['euribor3m']),
        'nr.employed':    float(form['nr_employed']),
    }

    # Ciklicno enkodiranje
    m = month_map.get(form['month'], 5)
    d = day_map.get(form['day_of_week'], 1)
    row['month_sin'] = np.sin(2 * np.pi * m / 12)
    row['month_cos'] = np.cos(2 * np.pi * m / 12)
    row['day_sin']   = np.sin(2 * np.pi * d / 5)
    row['day_cos']   = np.cos(2 * np.pi * d / 5)

    # One-Hot za job
    for j in ['blue-collar', 'entrepreneur', 'housemaid', 'management',
               'retired', 'self-employed', 'services', 'student',
               'technician', 'unemployed', 'unknown']:
        row[f'job_{j}'] = 1 if form['job'] == j else 0

    # One-Hot za marital
    for mv in ['married', 'single', 'unknown']:
        row[f'marital_{mv}'] = 1 if form['marital'] == mv else 0

    # One-Hot za poutcome
    for po in ['nonexistent', 'success']:
        row[f'poutcome_{po}'] = 1 if form['poutcome'] == po else 0

    return pd.DataFrame([row])


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    form = {}

    if request.method == 'POST':
        form = request.form
        X = preprocess_input(form)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        result = {'prediction': prediction, 'probability': probability}

    return render_template_string(HTML_TEMPLATE, result=result, form=form)


if __name__ == '__main__':
    print(f"Model ucitan iz: {MODEL_PATH}")
    print("Otvori browser na: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)