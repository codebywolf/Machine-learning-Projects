import os
import sys
import pandas as pd
import streamlit as st
import torch
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # one level up from src
sys.path.append(BASE_DIR)

from predict import ProductionInferenceEngine

# Set Page Configurations
st.set_page_config(
    page_title="Fake News Detection Portal",
    page_icon="🛡️",
    layout="wide"
)
# Initialize Session State values so they persist across page reruns
if "sandbox_news" not in st.session_state:
    st.session_state.sandbox_news = None
if "sandbox_label" not in st.session_state:
    st.session_state.sandbox_label = None

# Initialize Inference Backend directly using st.cache_resource
@st.cache_resource
def get_inference_engine():
    return ProductionInferenceEngine(
        roberta_dir=os.path.join(PROJECT_ROOT, 'artifacts', 'roberta_model', 'checkpoint-8232'),
        classical_model_path=os.path.join(PROJECT_ROOT, 'artifacts', 'lr_model.joblib'),
        vertorizer_path=os.path.join(PROJECT_ROOT, 'artifacts', 'tfidf_vectorizer.joblib')
    )
engine = get_inference_engine()

st.title('🛡️ FidesAI: Dual-Engine Fake News Classification Portal')

st.markdown(
    "Input any raw news article text block below. The framework passes the data through "
    "both the statistical **Logistic Regression Baseline** and the contextual **RoBERTa Deep Learning Transformer** simultaneously."
)
st.write("---")

# SUGGESTIONS SANDBOX SELECTION
st.subheader("💡 Quick-Test Suggestions Sandbox")
st.markdown("<span style='font-size: 14px;'>**Purpose:** Provides instant, one-click article text for quick testing without needing to browse the web for test cases.\n**Dataset Context:** Extracted directly from a combined (WALFake + ISOT_True + ISOT_Fake) benchmark framework.\n**Model Integrity:** Every sample is drawn from the held-out validation pool and remains strictly unseen by both models to ensure completely unbiased testing.</span>", unsafe_allow_html=True)

suggestions_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sample_suggestions.csv')
if os.path.exists(suggestions_path):
    def sample_generator(suggestions_df_path: str):
        df = pd.read_csv(suggestions_df_path)
        example = df.sample(1)
        news = example.text.values[0]
        label = example.label.values[0]
        return news, label

    # When clicked, update state variables so they survive the script execution rerun
    if st.button('Show News Sample', type='primary'):
        news, news_label = sample_generator(suggestions_path)
        st.session_state.sandbox_news = news
        st.session_state.sandbox_label = news_label

    # Always render the text container if data exists in the persistent session state
    if st.session_state.sandbox_news is not None:
        st.write('Category:', str(st.session_state.sandbox_label))
        st.code(
            body=st.session_state.sandbox_news,
            language="text",
            wrap_lines=True,
            height=200
        )
else:
    st.info("Sample suggestions dataset sheet not found at destination path.")
# ============================================================================
st.write("---")

col_input, col_clear = st.columns([0.95, 0.05], vertical_alignment="center")
with col_clear:
    if st.button('Clear', type='primary'):
        st.session_state['news_text_input'] = ''
        # 2. Evict cached models out of Streamlit resource memory structures
        # st.cache_resource.clear()

        # 3. Explicitly force Python's garbage collector to reclaim the freed memory space
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()


with col_input:
    # Input block section
    user_input = st.text_area(
        label='Paste news article text here:',
        placeholder='Type or paste text sequence...',
        height=250,
        key='news_text_input'
    )

if st.button('Execute Verification Analysis', type='primary'):
    if not user_input.strip():
        st.write('ACTION BLOCKED: Please input text content before executing an inference pass.')
    else:
        with st.spinner('Processing sequences through parallel network topologies...'):
            metrics = engine.predict_test(user_input)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Statistical Baseline (Logistic Regression)')
            if 'classical_baseline' in metrics:
                data = metrics['classical_baseline']
                pred = data['prediction']
                conf = data['confidence']

                if "FAKE" in pred:
                    st.error(f"**Classification:** {pred}")
                else:
                    st.success(f"**Classification:** {pred}")

                st.metric(label="Baseline Pipeline Confidence", value=f"{conf:.2%}")
            else:
                st.info("Logistic Regression components are currently unavailable.")

        with col2:
            st.subheader('Transformer Context Engine (RoBERTa)')
            if 'roberta_transformer' in metrics:
                data = metrics['roberta_transformer']
                pred = data['prediction']
                conf = data['confidence']

                if "FAKE" in pred:
                    st.error(f"**Classification:** {pred}")
                else:
                    st.success(f"**Classification:** {pred}")

                st.metric(label="Transformer Pipeline Confidence", value=f"{conf:.2%}")

                with st.expander("View Softmax Probabilities Details"):
                    prob_data = data.get("raw_probabilities", data.get("raw_probability"))
                    st.json(prob_data)
            else:
                st.info("Transformer Model Weights are currently unavailable.")
