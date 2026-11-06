import random
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Multilingual BERT Sentiment", page_icon="🔍", layout="wide")

BERT_ID = "TahiraAbbas/bert-multilingual-imdb-sentiment"
BERT_URL = f"https://huggingface.co/{BERT_ID}"

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1100px;}
.hero {background: linear-gradient(120deg, #0f172a 0%, #0e7490 60%, #06b6d4 130%);
    padding: 2rem 2.2rem; border-radius: 22px; color: #ffffff; margin-bottom: 1rem;
    box-shadow: 0 12px 32px rgba(14,116,144,0.35);}
.hero h1 {margin: 0; font-size: 2.3rem; color: #ffffff;}
.hero p {margin: 0.4rem 0 0 0; font-size: 1.05rem; color: #e2e8f0;}
.pill {display: inline-block; padding: 0.25rem 0.8rem; margin: 0.8rem 0.4rem 0 0;
    border-radius: 999px; background: rgba(255,255,255,0.18); font-size: 0.85rem; color: #ffffff;}
.mcard {border-radius: 16px; padding: 1.1rem 1.3rem; border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.06); border-top: 6px solid #06b6d4; line-height: 1.7;}
.result {border-radius: 16px; padding: 1.3rem; text-align: center; color: #ffffff;
    font-size: 1.8rem; font-weight: 800; margin-bottom: 0.8rem;}
.pos {background: linear-gradient(135deg, #047857, #10b981);}
.neg {background: linear-gradient(135deg, #be123c, #f43f5e);}
.small {font-size: 0.85rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(max_entries=1)
def load_model():
    return pipeline("sentiment-analysis", model=BERT_ID)

@st.cache_data
def load_samples():
    try:
        return pd.read_csv("sample_reviews.csv")["review"].tolist()
    except Exception:
        return []

samples = load_samples()
LABELS = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}

st.session_state.setdefault("bert_text", "The movie was excellent.")
st.session_state.setdefault("bert_history", [])

with st.sidebar:
    st.header("Model")
    st.markdown(f"""
**🔍 Multilingual BERT** (encoder-only)

- Base: `bert-base-multilingual-cased`
- Languages: 104
- **Dataset:** IMDB movie reviews
- [Open on Hugging Face]({BERT_URL})
""")
    st.caption("The model runs on CPU, so the first request can take a while.")
    if st.button("Clear history", use_container_width=True):
        st.session_state["bert_history"] = []
        st.rerun()

st.markdown("""
<div class="hero">
  <h1>🔍 Multilingual BERT: Sentiment Classification</h1>
  <p>An encoder-only Transformer that reads the whole text and predicts POSITIVE or NEGATIVE.</p>
  <span class="pill">Encoder-only</span><span class="pill">Multilingual</span>
  <span class="pill">IMDB dataset</span><span class="pill">Hugging Face</span>
</div>
""", unsafe_allow_html=True)

tab_run, tab_info = st.tabs(["🔍 Classify", "📚 Model & Dataset"])

with tab_run:
    st.caption("Fine-tuned on English IMDB reviews. Try other languages to see cross-lingual transfer.")
    st.write("**Try an example:**")
    ex = st.columns(4)
    examples = ["The movie was excellent.", "La película fue excelente.",
                "Ce film était terrible.", "یہ فلم بہت اچھی تھی۔"]
    for col, e in zip(ex, examples):
        if col.button(e, key="ex_" + e, use_container_width=True):
            st.session_state["bert_text"] = e
    if samples and st.button("🎲 Load a random IMDB review"):
        st.session_state["bert_text"] = random.choice(samples)

    text = st.text_area("Input text", key="bert_text", height=150)
    st.markdown(f"<span class='small'>{len(text.split())} words</span>", unsafe_allow_html=True)

    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Loading BERT and classifying..."):
                clf = load_model()
                res = clf(text, truncation=True, max_length=512, top_k=None)
                if res and isinstance(res[0], list):
                    res = res[0]
            scores = {LABELS.get(r["label"], r["label"]): r["score"] for r in res}
            best = max(scores, key=scores.get)
            css = "pos" if best == "POSITIVE" else "neg"
            icon = "😊" if best == "POSITIVE" else "😞"
            st.markdown(f"<div class='result {css}'>{icon} {best}</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{scores[best]:.1%}")
            c2.metric("Words analysed", min(len(text.split()), 512))
            for lab in ["POSITIVE", "NEGATIVE"]:
                if lab in scores:
                    st.write(f"{lab}: {scores[lab]:.1%}")
                    st.progress(float(scores[lab]))
            with st.expander("Raw model output"):
                st.json(res)
            st.session_state["bert_history"].insert(0, {
                "Input": text[:80] + ("..." if len(text) > 80 else ""),
                "Output": best, "Confidence": f"{scores[best]:.1%}"})

    if st.session_state["bert_history"]:
        st.markdown("#### History")
        st.dataframe(pd.DataFrame(st.session_state["bert_history"]), use_container_width=True, hide_index=True)

with tab_info:
    st.markdown(f"""
<div class="mcard">
<h3>🔍 Multilingual BERT</h3>
<b>Architecture:</b> encoder-only (bidirectional attention)<br>
<b>Base model:</b> bert-base-multilingual-cased (104 languages)<br>
<b>Fine-tuned model:</b> <a href="{BERT_URL}" target="_blank">{BERT_ID}</a><br>
<b>Task:</b> sentiment classification (POSITIVE / NEGATIVE)<br>
<b>Dataset:</b> IMDB movie reviews (labelled positive / negative)<br>
<b>Why this dataset:</b> classification needs labelled examples.<br>
<b>Pre-training:</b> masked word prediction<br>
<b>Best for:</b> classification, NER, question answering
</div>""", unsafe_allow_html=True)
    st.success("The encoder reads the whole input at once, using words on both the left and the right.")
