import streamlit as st
import pickle
import re
import html
from pathlib import Path

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import hstack as sp_hstack

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multilingual Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD NLTK
# ============================================================

nltk.download("stopwords", quiet=True)

# ============================================================
# LOAD MODEL FILES
# ============================================================

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "best_model.pkl", "rb") as f:
    best_clf = pickle.load(f)

with open(BASE_DIR / "tfidf_word.pkl", "rb") as f:
    tfidf_word = pickle.load(f)

with open(BASE_DIR / "tfidf_char.pkl", "rb") as f:
    tfidf_char = pickle.load(f)

with open(BASE_DIR / "label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ============================================================
# PREPROCESSING
# ============================================================

ps = PorterStemmer()

english_stops = set(stopwords.words("english"))

hinglish_stops = {
    'hai', 'hain', 'ho', 'tha', 'thi', 'the', 'ka', 'ki', 'ke',
    'ko', 'se', 'me', 'mein', 'pe', 'par', 'aur', 'ya', 'bhi',
    'yeh', 'ye', 'woh', 'wo', 'ek', 'koi', 'kuch', 'sab', 'apna',
    'apni', 'apne', 'uska', 'uski', 'uske', 'mera', 'meri', 'mere',
    'tera', 'teri', 'tere', 'humara', 'tumhara', 'unka', 'kya', 'kyun',
    'kaise', 'kab', 'kahan', 'kaun', 'nahi', 'nahin', 'mat', 'na',
    'ji', 'bhai', 'yaar', 'dost', 'sir', 'madam'
}

all_stops = english_stops | hinglish_stops


def preprocess_multilingual(text):

    text = str(text).lower()

    text = re.sub(
        r'http\S+|www\.\S+',
        ' url ',
        text
    )

    text = re.sub(
        r'\S+@\S+',
        ' email ',
        text
    )

    text = re.sub(
        r'\b\d{10,}\b',
        ' phonenumber ',
        text
    )

    text = re.sub(
        r'₹|rs\.?|inr',
        ' rupees ',
        text
    )

    text = re.sub(
        r'£|\$|€',
        ' currency ',
        text
    )

    text = re.sub(
        r'[^\w\s\u0900-\u097F]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    tokens = text.split()

    cleaned = []

    for tok in tokens:

        if tok in all_stops or len(tok) < 2:
            continue

        if re.match(r'^[a-z]+$', tok):
            tok = ps.stem(tok)

        cleaned.append(tok)

    return ' '.join(cleaned)


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(message):

    try:

        from langdetect import detect

        raw_lang = detect(str(message))

        if raw_lang == "hi":
            return "Hindi"

        elif raw_lang == "en":
            return "English"

        else:
            return "Hinglish"

    except:

        return "Hinglish"


# ============================================================
# PREDICTION
# ============================================================

def predict_message(message):

    language = detect_language(message)

    processed = preprocess_multilingual(message)

    X_w = tfidf_word.transform([processed])

    X_c = tfidf_char.transform([processed])

    X = sp_hstack([X_w, X_c])

    label = best_clf.predict(X)[0]

    probability = (
        best_clf.predict_proba(X)[0]
        if hasattr(best_clf, "predict_proba")
        else None
    )

    if label == 1:

        result = "SPAM / FRAUD"

    else:

        result = "LEGITIMATE"

    if probability is not None:

        confidence = max(probability) * 100

    else:

        confidence = 0

    return language, result, confidence


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 50%,
            #f8fbff 100%
        );
    }

    /* Hide default menu/footer */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */

    .hero {
        padding: 30px 10px 20px 10px;
        text-align: center;
    }

    .hero-icon {
        font-size: 55px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #475569;
        max-width: 850px;
        margin: auto;
    }

    /* Cards */

    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.07);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 23px;
        font-weight: 700;
        color: #172554;
        margin-bottom: 10px;
    }

    /* Result cards */

    .spam-card {
        background: #fff1f2;
        border: 2px solid #fb7185;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
    }

    .ham-card {
        background: #ecfdf5;
        border: 2px solid #34d399;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
    }

    .spam-text {
        color: #be123c;
    }

    .ham-text {
        color: #047857;
    }

    .result-info {
        font-size: 17px;
        color: #334155;
        margin-top: 8px;
    }

    /* Feature cards */

    .feature {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        height: 150px;
        box-shadow: 0 5px 15px rgba(15, 23, 42, 0.05);
    }

    .feature-icon {
        font-size: 30px;
    }

    .feature-title {
        font-size: 17px;
        font-weight: 700;
        color: #172554;
    }

    .feature-text {
        color: #64748b;
        font-size: 14px;
    }

    /* Example boxes */

    .example-box {
        padding: 15px;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 10px;
        color: #334155;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #172554,
            #1e3a8a
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Button */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 17px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:20px 5px;">
            <div style="font-size:60px;">🛡️</div>
            <h2>Multilingual<br>Fraud Detection</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🌐 Supported Languages")

    st.markdown(
        """
        **🇬🇧 English**

        **🇮🇳 Hindi**

        **💬 Hinglish**
        """
    )

    st.divider()

    st.markdown("### 🤖 Machine Learning")

    st.markdown(
        """
        **Final Model**

        Linear SVM

        **Features**

        Word TF-IDF + Character TF-IDF
        """
    )

    st.divider()

    st.markdown("### 💡 How to use")

    st.markdown(
        """
        1. Enter a message.
        2. Click **Detect Message**.
        3. Check the detected language.
        4. View the prediction and confidence.
        """
    )

    st.divider()

    st.caption(
        "Academic Project • Multilingual Spam & Fraud Detection"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">🛡️</div>

        <div class="hero-title">
            Multilingual Phishing & Spam Detection
        </div>

        <div class="hero-subtitle">
            Detect spam, phishing, fraudulent and unwanted messages
            across English, Hindi and Hinglish using Machine Learning
            and Natural Language Processing.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MESSAGE INPUT
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            ✉️ Analyze Your Message
        </div>

        <div style="color:#64748b; margin-bottom:15px;">
            Enter any SMS, WhatsApp message, email text or suspicious
            message below.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

message = st.text_area(
    "Message",
    height=170,
    placeholder=(
        "Example:\n"
        "Congratulations! You have won ₹10,00,000. "
        "Click this link to claim your prize..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# DETECT BUTTON
# ============================================================

detect_clicked = st.button(
    "🔍  Detect Message",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION RESULT
# ============================================================

if detect_clicked:

    if not message.strip():

        st.warning(
            "⚠️ Please enter a message before clicking Detect Message."
        )

    else:

        language, result, confidence = predict_message(message)

        safe_message = html.escape(message)

        if result == "SPAM / FRAUD":

            st.markdown(
                f"""
                <div class="spam-card">

                    <div class="result-title spam-text">
                        🚨 SPAM / FRAUD DETECTED
                    </div>

                    <div class="result-info">
                        This message has been classified as
                        potentially unwanted, fraudulent or suspicious.
                    </div>

                    <br>

                    <b>🌐 Language:</b> {language}<br>

                    <b>📊 Confidence:</b> {confidence:.1f}%

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="ham-card">

                    <div class="result-title ham-text">
                        ✅ LEGITIMATE MESSAGE
                    </div>

                    <div class="result-info">
                        The model classified this message as
                        legitimate (Ham).
                    </div>

                    <br>

                    <b>🌐 Language:</b> {language}<br>

                    <b>📊 Confidence:</b> {confidence:.1f}%

                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(
            min(int(confidence), 100),
            text=f"Model Confidence: {confidence:.1f}%"
        )

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    📋 Analyzed Message
                </div>

                <div class="example-box">
                    {safe_message}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FEATURES
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="card-title">
        ✨ Key Features
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        <div class="feature">

            <div class="feature-icon">🌐</div>

            <div class="feature-title">
                Multilingual
            </div>

            <div class="feature-text">
                English, Hindi and Hinglish messages.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="feature">

            <div class="feature-icon">🤖</div>

            <div class="feature-title">
                Machine Learning
            </div>

            <div class="feature-text">
                Linear SVM classification model.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="feature">

            <div class="feature-icon">🔤</div>

            <div class="feature-title">
                NLP Features
            </div>

            <div class="feature-text">
                Word and character TF-IDF features.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="feature">

            <div class="feature-icon">⚡</div>

            <div class="feature-title">
                Fast Prediction
            </div>

            <div class="feature-text">
                Analyze messages instantly.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# EXAMPLES
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="card-title">
        💬 Try These Example Messages
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="example-box">
        🚨 <b>Suspicious:</b><br>
        Congratulations! You have won a free lottery.
        Claim your prize now!
        </div>

        <div class="example-box">
        🚨 <b>Suspicious:</b><br>
        URGENT: Your bank account is blocked.
        Share OTP to verify.
        </div>

        <div class="example-box">
        🚨 <b>Hinglish:</b><br>
        Yaar FREE iPhone jeetne ke liye is link pe click kar jaldi!
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="example-box">
        ✅ <b>Legitimate:</b><br>
        Hey, are you free for dinner tonight?
        </div>

        <div class="example-box">
        ✅ <b>Hindi:</b><br>
        कल मिलते हैं, ठीक है?
        </div>

        <div class="example-box">
        ✅ <b>Hinglish:</b><br>
        Bhai kal cricket dekhne chalte hain?
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        text-align:center;
        padding:25px;
        color:#64748b;
        border-top:1px solid #e2e8f0;
    ">

        🛡️ <b>Multilingual Phishing & Spam Detection</b>

        <br>

        Machine Learning Based Detection System

        <br><br>

        <small>
        English • Hindi • Hinglish
        </small>

    </div>
    """,
    unsafe_allow_html=True
)
