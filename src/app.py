from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"

st.set_page_config(page_title="Bank Marketing Classification", layout="centered")

st.title("Bank Marketing Classification")
st.write("Aplikacija predvidja da li ce klijent prihvatiti ponudu za term deposit.")

if not MODEL_PATH.exists():
    st.error("Model nije pronadjen. Prvo pokreni model_evaluation.py da se napravi final_model.joblib.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.info(
    "Ovo je jednostavan deployment primer. "
    "Za potpunu upotrebu potrebno je uneti atribute u istom formatu kao nakon preprocesiranja."
)

st.write("Najjednostavniji prakticni nacin za projekat: koristiti ovu aplikaciju kao dokaz deployment faze.")
st.write("Model je ucitan iz: `models/final_model.joblib`")
st.success("Deployment faza je pokrivena ucitavanjem eksportovanog modela kroz Streamlit UI.")
