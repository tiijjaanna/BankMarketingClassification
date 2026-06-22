"""
Bank Marketing Classification — Streamlit aplikacija
Učitava best_tuned_model.joblib (ceo pipeline: preprocessor + SMOTE + classifier)
i vrši predikciju za novog klijenta.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ── Putanje ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_tuned_model.joblib"

# ── Učitavanje modela ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Zaglavlje ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bank Marketing Prediction", page_icon="🏦", layout="centered")
st.title("🏦 Bank Marketing Prediction")
st.write(
    "Predikcija da li će klijent prihvatiti ponudu oročenog depozita. "
    "Unesite podatke o klijentu i kliknite **Predict**."
)

# ── Sidebar: opis projekta ────────────────────────────────────────────────────
with st.sidebar:
    st.header("O projektu")
    st.write(
        "U pitanju je studentski projekat koji vrši predikciju pretplate klijenta  "
        "na bankovni rok depozit na osnovu demografskih, profesionalnih i  "
        "promotivnih karakteristika."
    )
    st.write("**Dataset:** Bank Marketing (UCI), 41 188 klijenata.")
    st.write("**Najbolji model:** Logistic Regression.")


# ── Input widgeti ─────────────────────────────────────────────────────────────
st.subheader("Lični podaci")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Starost", min_value=18, max_value=100, value=40, step=1)
    job = st.selectbox("Zanimanje", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ])
    marital = st.selectbox("Bračni status", ["divorced", "married", "single", "unknown"])

with col2:
    education = st.selectbox("Nivo obrazovanja", [
        "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
        "professional.course", "university.degree", "unknown",
    ])
    default = st.selectbox("Kreditno zaduženje", ["no", "unknown", "yes"])

st.subheader("Finansijski podaci")
col3, col4 = st.columns(2)

with col3:
    housing = st.selectbox("Stambeni kredit", ["no", "unknown", "yes"])

with col4:
    loan = st.selectbox("Lični kredit", ["no", "unknown", "yes"])

st.subheader("Podaci o kampanji")
col5, col6 = st.columns(2)

with col5:
    contact = st.selectbox("Tip kontakta", ["cellular", "telephone"])
    month = st.selectbox("Mesec kontakta", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ])
    day_of_week = st.selectbox("Dan u nedelji", ["mon", "tue", "wed", "thu", "fri"])

with col6:
    poutcome = st.selectbox("Ishod prethodne kampanje", ["failure", "nonexistent", "success"])
    campaign = st.number_input("Broj kontakata u ovoj kampanji", min_value=1, max_value=50, value=1, step=1)
    previous = st.number_input("Broj prethodnih kontakata", min_value=0, max_value=50, value=0, step=1)

pdays = st.number_input(
    "Broj dana od prethodnog kontakta (-1 = nije prethodno kontaktiran)",
    min_value=-1, max_value=999, value=-1, step=1,
)

st.subheader("Ekonomski pokazatelji")
col7, col8 = st.columns(2)

with col7:
    cons_price_idx = st.number_input(
        "Indeks potrošačkih cena", min_value=90.0, max_value=100.0,
        value=93.994, step=0.001, format="%.3f",
    )
    euribor3m = st.number_input(
        "Euribor 3-mesečna stopa (%)", min_value=0.0, max_value=6.0,
        value=4.857, step=0.001, format="%.3f",
    )

with col8:
    cons_conf_idx = st.number_input(
        "Indeks poverenja potrošača", min_value=-60.0, max_value=-20.0,
        value=-40.5, step=0.1, format="%.1f",
    )

# ── Predikcija ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    input_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "poutcome": poutcome,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    st.divider()
    st.subheader("📊 Rezultat predikcije")

    if prediction == 1:
        st.success(f"✅ **Klijent će se verovatno pretplatiti** (verovatnoća: {probability * 100:.1f}%)")
    else:
        st.error(f"❌ **Klijent se verovatno neće pretplatiti** (verovatnoća: {probability * 100:.1f}%)")

    st.progress(float(probability))
    st.caption(f"Verovatnoća pretplate: {probability * 100:.1f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Verovatnoća YES", f"{probability * 100:.1f}%")
    with col_b:
        st.metric("Verovatnoća NO", f"{(1 - probability) * 100:.1f}%")