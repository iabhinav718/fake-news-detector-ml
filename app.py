import streamlit as st
import pickle
import re

# Load trained model and vectorizer
@st.cache_resource
def load_model():
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def clean_text(text):
    """Clean and preprocess text data"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_news(text, model, vectorizer):
    """Predict if news is fake or real"""
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return None, None
    
    # Transform text using TF-IDF
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    return prediction, probability

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fake-news {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .real-news {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
try:
    model, vectorizer = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")

# Header
st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect fake news using AI-powered Natural Language Processing</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This application uses Machine Learning to detect fake news articles.
    
    **How it works:**
    1. Enter news text
    2. Click 'Analyze'
    3. Get instant prediction
    
    **Technology:**
    - TF-IDF Vectorization
    - Logistic Regression
    - Natural Language Processing
    """)
    
    st.header("Examples")
    if st.button("Load Fake News Example"):
        st.session_state.example_text = "BREAKING: Scientists confirm earth will stop rotating tomorrow! Experts warn catastrophic consequences are imminent."
    
    if st.button("Load Real News Example"):
        st.session_state.example_text = "WASHINGTON (Reuters) - The U.S. Federal Reserve signaled that interest rates could remain high as inflation moderates, according to statements released today."

# Main content
st.write("---")

# Text input
default_text = st.session_state.get('example_text', '')
user_input = st.text_area(
    "Enter news article text:",
    value=default_text,
    height=200,
    placeholder="Paste your news article here..."
)

# Clear example from session state after use
if 'example_text' in st.session_state and user_input == st.session_state.example_text:
    del st.session_state.example_text

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

# Prediction
if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    elif not model_loaded:
        st.error("❌ Model not loaded. Please check the error message above.")
    else:
        with st.spinner("Analyzing..."):
            prediction, probability = predict_news(user_input, model, vectorizer)
            
            if prediction is None:
                st.error("❌ Unable to process the text. Please enter valid news content.")
            else:
                st.write("---")
                
                if prediction == 0:  # Fake
                    st.markdown('<div class="fake-news">', unsafe_allow_html=True)
                    st.markdown("### 🚨 Prediction: FAKE NEWS")
                    st.markdown(f"**Confidence:** {probability[0]*100:.2f}%")
                    st.markdown("This article appears to contain characteristics of fake news.")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:  # Real
                    st.markdown('<div class="real-news">', unsafe_allow_html=True)
                    st.markdown("### ✅ Prediction: REAL NEWS")
                    st.markdown(f"**Confidence:** {probability[1]*100:.2f}%")
                    st.markdown("This article appears to be legitimate news content.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show probability breakdown
                st.write("---")
                st.subheader("Confidence Breakdown")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake News Probability", f"{probability[0]*100:.2f}%")
                with col2:
                    st.metric("Real News Probability", f"{probability[1]*100:.2f}%")

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with Streamlit | Powered by Machine Learning</p>
    <p><small>Note: This tool provides predictions based on linguistic patterns and should not be the sole basis for determining news authenticity.</small></p>
</div>
""", unsafe_allow_html=True)