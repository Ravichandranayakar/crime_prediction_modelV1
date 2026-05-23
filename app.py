import streamlit as st
import joblib
import json
import numpy as np
import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
MODEL_DIR = Path(__file__).parent / "crime_prediction_modelV1/model_trainingV1/artifacts"
INFERENCE_DIR = Path(__file__).parent / "crime_prediction_modelV1/inferenceV1"

# Page config
st.set_page_config(
    page_title="Crime Type Prediction",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom Styling
# ============================================================================
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .title-container {
            text-align: center;
            margin-bottom: 2rem;
        }
        .warning-box {
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            border-left: 5px solid #ff6b6b;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .warning-title {
            font-weight: bold;
            color: #333;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        .warning-text {
            color: #333;
            line-height: 1.6;
        }
        .input-label {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            color: #333;
        }
        .result-card {
            background: #f8f9fa;
            border-left: 5px solid #0066cc;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .crime-label {
            display: inline-block;
            background: #0066cc;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            margin: 0.3rem;
            font-weight: 500;
        }
        .confidence {
            font-size: 1.2rem;
            font-weight: 600;
            color: #0066cc;
            margin-top: 1rem;
        }
        .error-box {
            background: #fee;
            border-left: 5px solid #c33;
            padding: 1rem;
            border-radius: 0.5rem;
            color: #c33;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# Load Model
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained crime prediction model and associated artifacts."""
    try:
        model = joblib.load(MODEL_DIR / "crime_model_v1.pkl")
        tfidf = joblib.load(MODEL_DIR / "tfidf.pkl")
        mlb = joblib.load(MODEL_DIR / "mlb.pkl")
        
        with open(MODEL_DIR / "label_thresholds.json", "r") as f:
            thresholds = json.load(f)
        
        return {
            "model": model,
            "tfidf": tfidf,
            "mlb": mlb,
            "thresholds": thresholds,
            "label_names": list(mlb.classes_)
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ============================================================================
# Prediction Functions
# ============================================================================
def apply_thresholds(probs, label_names, thresholds):
    """Apply custom thresholds to prediction probabilities."""
    y_pred = np.zeros_like(probs, dtype=int)
    for j, label in enumerate(label_names):
        th = thresholds.get(label, 0.5)
        y_pred[:, j] = (probs[:, j] >= th).astype(int)
    return y_pred

def predict_crime(text, model_dict, mode="citizen"):
    """
    Predict crime type from narrative text.
    
    Args:
        text: FIR narrative or complaint text
        model_dict: Dictionary containing model artifacts
        mode: "citizen" for citizen view, "lawyer" for detailed view
    
    Returns:
        Dictionary with predictions and confidence
    """
    MIN_WORDS = 8
    
    # Validate input
    if len(text.split()) < MIN_WORDS:
        return {
            "error": f"Insufficient information. Please provide at least {MIN_WORDS} words."
        }
    
    # Vectorize and predict
    X_vec = model_dict["tfidf"].transform([text])
    proba = model_dict["model"].predict_proba(X_vec)[0]
    
    # Apply thresholds
    y_pred = apply_thresholds(
        proba.reshape(1, -1),
        model_dict["label_names"],
        model_dict["thresholds"]
    )[0]
    
    # Get predicted labels
    if y_pred.any():
        predicted_labels = model_dict["mlb"].inverse_transform(y_pred.reshape(1, -1))[0]
        predicted_labels = list(predicted_labels)
    else:
        predicted_labels = []
    
    # Get confidence (max probability among positive predictions)
    if predicted_labels:
        conf = float(proba[y_pred == 1].max())
    else:
        conf = 0.0
    
    # Filter for citizen view
    if mode == "citizen":
        citizen_visible = {
            "homicide", "sexual_offence", "kidnapping",
            "domestic_violence", "cyber_crime", "fraud",
            "robbery", "cheating"
        }
        predicted_labels = [l for l in predicted_labels if l in citizen_visible]
    
    return {
        "labels": predicted_labels,
        "confidence": conf,
        "all_probabilities": dict(zip(model_dict["label_names"], proba))
    }

# ============================================================================
# Main UI
# ============================================================================
def main():
    # Load model
    model_dict = load_model()
    if model_dict is None:
        st.error("Failed to load the model. Please check the model files.")
        return
    
    # Header
    st.markdown("""
        <div class="title-container">
            <h1>⚖️ Crime Type Prediction</h1>
            <p style="font-size: 1.1rem; color: #666;">AI-powered crime classification from FIR narrative text</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Warning Box
    st.markdown("""
        <div class="warning-box">
            <div class="warning-title">⚠️ Important Notice</div>
            <div class="warning-text">
                This is an AI-powered prediction model. Results are for reference only and should not be considered 
                as legal advice. Always consult qualified legal professionals for official classification and legal proceedings.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        mode = st.selectbox(
            "View Mode",
            options=["citizen", "lawyer"],
            format_func=lambda x: "Citizen View" if x == "citizen" else "Lawyer/Detailed View",
            help="Citizen view shows common crimes, Lawyer view shows all crime categories"
        )
        
        st.divider()
        st.markdown("### Model Information")
        st.info(f"""
        - **Crime Categories**: {len(model_dict['label_names'])}
        - **Model Type**: TF-IDF + Logistic Regression
        - **Min. Text Length**: 8 words
        - **Max. Text Length**: 5000 characters
        """)
    
    # Input Section
    st.markdown('<div class="input-label">📝 Enter FIR Narrative or Complaint Text</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        max_chars = 5000
        narrative_text = st.text_area(
            label="FIR Narrative or Complaint Text",
            placeholder="Paste FIR narrative or complaint text here... (minimum 20 characters, maximum 5000 characters)",
            height=250,
            max_chars=max_chars,
            label_visibility="collapsed"
        )
        
        char_count = len(narrative_text)
        word_count = len(narrative_text.split()) if narrative_text else 0
        
        st.caption(f"📊 Characters: {char_count} / {max_chars} | Words: {word_count}")
    
    with col2:
        st.markdown("### Instructions")
        st.markdown("""
        1. **Paste or type** the FIR narrative or complaint text
        2. **Minimum requirement**: 20 characters or 8 words
        3. **Click** "Predict Crime Types" button
        4. **Review** the predicted crime categories
        5. **Note**: AI predictions are for reference only
        
        ### Crime Categories Covered
        - Homicide
        - Sexual Offence
        - Kidnapping
        - Domestic Violence
        - Cyber Crime
        - Fraud & Cheating
        - Robbery & Theft
        - Assault & Weapons
        - And more...
        """)
    
    st.divider()
    
    # Prediction Button
    if st.button(
        "🔍 Predict Crime Types",
        use_container_width=True,
        type="primary",
        key="predict_btn"
    ):
        if not narrative_text or len(narrative_text) < 20:
            st.error("❌ Please enter at least 20 characters of text.")
        elif word_count < 8:
            st.error("❌ Please provide at least 8 words.")
        else:
            with st.spinner("🤔 Analyzing text..."):
                result = predict_crime(narrative_text, model_dict, mode=mode)
            
            st.divider()
            st.markdown("### Prediction Results")
            
            if "error" in result:
                st.markdown(f'<div class="error-box">{result["error"]}</div>', unsafe_allow_html=True)
            else:
                if result["labels"]:
                    st.success(f"✅ Predicted Crime Types")
                    
                    # Display crime labels
                    cols = st.columns(len(result["labels"]) if len(result["labels"]) <= 4 else 4)
                    for idx, label in enumerate(result["labels"]):
                        with cols[idx % 4]:
                            st.markdown(
                                f'<div class="crime-label">{label.replace("_", " ").title()}</div>',
                                unsafe_allow_html=True
                            )
                    
                    # Confidence
                    confidence_pct = result["confidence"] * 100
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f'<div class="confidence">Confidence: {confidence_pct:.1f}%</div>', unsafe_allow_html=True)
                    with col2:
                        if confidence_pct >= 80:
                            st.success("🟢 High Confidence")
                        elif confidence_pct >= 60:
                            st.info("🟡 Medium Confidence")
                        else:
                            st.warning("🟠 Low Confidence")
                    
                    # Detailed probabilities
                    with st.expander("📊 View All Crime Probabilities"):
                        prob_data = sorted(
                            result["all_probabilities"].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write("**Crime Category**")
                        with col2:
                            st.write("**Probability**")
                        
                        st.divider()
                        
                        for label, prob in prob_data:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(label.replace("_", " ").title())
                            with col2:
                                st.write(f"{prob*100:.1f}%")
                
                else:
                    st.info("ℹ️ No specific crime types matched with sufficient confidence. The narrative may describe a general offense.")
    
    st.divider()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; color: #999; font-size: 0.9rem; margin-top: 2rem;">
            <p>🔒 This application uses machine learning for classification purposes only.</p>
            <p>For official legal matters, please consult with qualified legal professionals.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
