import os
import streamlit as st
import pandas as pd
from datetime import datetime
from emotion_engine import EmotionEngine
from guidance import generate_guidance

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

@st.cache_resource
def load_engine():
    return EmotionEngine()

engine = load_engine()
LOG_FILE = "data/interactions.csv"

def save_interaction(text, result):
    os.makedirs("data", exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "student_input": text,
        "bilstm_emotion": result["bilstm"]["emotion"],
        "bilstm_confidence": round(result["bilstm"]["confidence"], 4),
        "bert_emotion": result["bert"]["emotion"],
        "bert_confidence": round(result["bert"]["confidence"], 4),
        "final_emotion": result["final_emotion"],
        "mixed_emotions": ", ".join(
            f"{x['emotion']}:{x['score']:.2f}" for x in result["mixed_emotions"]
        ),
    }])
    if os.path.exists(LOG_FILE):
        row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_FILE, index=False)

st.title("🎓 AI-Driven Emotion Detection & Personalized Learning Support")
st.caption("Emotion-aware academic support using BiLSTM-style sequence scoring, BERT, rule enhancement, and Gemini guidance.")

with st.sidebar:
    st.header("⚙️ Controls")
    use_gemini = st.toggle("Use Gemini AI response", value=False)
    st.info("Set GEMINI_API_KEY in your environment or .streamlit/secrets.toml to enable Gemini.")
    page = st.radio("Navigate", ["Learning Assistant", "Analytics Dashboard", "Model Comparison", "About"])

if page == "Learning Assistant":
    st.subheader("Describe your study challenge")
    text = st.text_area(
        "Student input",
        placeholder="Example: I tried recursion many times but I still don't understand the base case.",
        height=150,
    )
    if st.button("🔍 Analyze & Support", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter a study challenge.")
        else:
            with st.spinner("Analyzing emotional context..."):
                result = engine.predict(text)
                guidance = generate_guidance(text, result["final_emotion"], use_gemini)

            c1, c2, c3 = st.columns(3)
            c1.metric("Final Emotion", result["final_emotion"])
            c2.metric("BiLSTM Prediction", f"{result['bilstm']['emotion']} ({result['bilstm']['confidence']:.0%})")
            c3.metric("BERT Prediction", f"{result['bert']['emotion']} ({result['bert']['confidence']:.0%})")

            st.subheader("Mixed-emotion breakdown")
            chart_df = pd.DataFrame(result["all_scores"]).set_index("emotion")
            st.bar_chart(chart_df["score"])

            st.subheader("💡 Personalized Learning Support")
            st.markdown(guidance)
            save_interaction(text, result)
            st.success("Interaction saved to CSV.")

elif page == "Analytics Dashboard":
    st.subheader("📊 Emotion Analytics Dashboard")
    if not os.path.exists(LOG_FILE):
        st.info("No interactions yet. Analyze a study challenge first.")
    else:
        df = pd.read_csv(LOG_FILE)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Interactions", len(df))
        c2.metric("Most Common Emotion", df["final_emotion"].mode().iloc[0])
        c3.metric("Emotion Types Seen", df["final_emotion"].nunique())
        st.subheader("Emotion distribution")
        st.bar_chart(df["final_emotion"].value_counts())
        st.subheader("Recent interactions")
        st.dataframe(df.tail(20).sort_index(ascending=False), use_container_width=True)

elif page == "Model Comparison":
    st.subheader("⚖️ BiLSTM vs BERT Comparison")
    if not os.path.exists(LOG_FILE):
        st.info("No model comparison data yet.")
    else:
        df = pd.read_csv(LOG_FILE)
        agreement = (df["bilstm_emotion"] == df["bert_emotion"]).mean()
        st.metric("Model Agreement", f"{agreement:.1%}")
        comparison = pd.crosstab(df["bilstm_emotion"], df["bert_emotion"])
        st.dataframe(comparison, use_container_width=True)
        st.caption("Rows: BiLSTM-style sequence prediction. Columns: BERT prediction.")

else:
    st.subheader("About the Project")
    st.markdown("""
This platform converts a learner's free-text study challenge into emotion-aware academic support.

**Emotion classes:** Bored, Confident, Confused, Curious, Frustrated.

**Pipeline:** Student text → preprocessing → BiLSTM-style sequence scoring + BERT → keyword enhancement → mixed-emotion analysis → final emotion → Gemini/fallback guidance → CSV logging → analytics.

**Responsible AI:** The application is designed for learning support, not medical or psychological diagnosis. Predictions are contextual signals and may be imperfect.
""")
