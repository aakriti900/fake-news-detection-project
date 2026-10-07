"""Fake News Detector - web app. Run: streamlit run app.py"""
import os
import joblib
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

MODEL_PATH = "outputs/model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("📰 Fake News Detector")
st.write("Paste any news headline or text and check whether it is REAL or FAKE.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. First run `python train.py` in the terminal, then reopen the app.")
    st.stop()

model = load_model()

examples = {
    "Real example": "Government releases inflation figures for the quarter, showing a modest increase",
    "Fake example": "SHOCKING: Government secretly hiding the truth about vaccination, insiders reveal!!!",
}

if "text" not in st.session_state:
    st.session_state.text = ""

c1, c2 = st.columns(2)
if c1.button("Try real example"):
    st.session_state.text = examples["Real example"]
if c2.button("Try fake example"):
    st.session_state.text = examples["Fake example"]

text = st.text_area("News text", key="text", height=150, placeholder="Paste a headline here...")

if st.button("Check News", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        label = model.predict([text])[0]
        if hasattr(model, "predict_proba"):
            proba = dict(zip(model.classes_, model.predict_proba([text])[0]))
            conf = max(proba.values())
        else:
            proba, conf = None, None

        if label == "FAKE":
            st.error("🚨 This news looks FAKE")
        else:
            st.success("✅ This news looks REAL")

        if proba:
            st.write(f"Confidence: **{conf:.1%}**")
            st.progress(float(proba.get("FAKE", 0)), text=f"Fake probability: {proba.get('FAKE', 0):.1%}")
            st.progress(float(proba.get("REAL", 0)), text=f"Real probability: {proba.get('REAL', 0):.1%}")

st.caption("Note: This is an ML model and is not 100% accurate. Always verify important news with trusted sources.")
