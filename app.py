import streamlit as st
import pickle
import re
from pathlib import Path

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import hstack as sp_hstack


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ShieldAI | Multilingual Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
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

    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\b\d{10,}\b', ' phonenumber ', text)
    text = re.sub(r'₹|rs\.?|inr', ' rupees ', text)
    text = re.sub(r'£|\$|€', ' currency ', text)

    text = re.sub(
        r'[^\w\s\u0900-\u097F]',
        ' ',
        text
    )

    text = re.sub(r'\s+', ' ', text).strip()

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

        if raw_lang == "en":
            return "English"

        return "Hinglish"

    except Exception:

        return "Hinglish"


# ============================================================
# PREDICTION
# ============================================================

def predict_message(message):

    language = detect_language(message)

    processed = preprocess_multilingual(message)

    X_word = tfidf_word.transform([processed])
    X_char = tfidf_char.transform([processed])

    X = sp_hstack([X_word, X_char])

    label = best_clf.predict(X)[0]

    if hasattr(best_clf, "predict_proba"):
        probability = best_clf.predict_proba(X)[0]
        confidence = float(max(probability) * 100)
    else:
        confidence = 0.0

    if label == 1:
        result = "SPAM / FRAUD"
    else:
        result = "LEGITIMATE"

    return language, result, confidence


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(37, 99, 235, 0.08),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(14, 165, 233, 0.08),
            transparent 30%
        ),
        #f7f9fc;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ==========================================================
   TOP NAV
   ========================================================== */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 4px 22px 4px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: linear-gradient(
        135deg,
        #2563eb,
        #0ea5e9
    );

    color: white;
    font-size: 22px;

    box-shadow:
        0 8px 20px rgba(37, 99, 235, 0.25);
}

.brand-name {
    font-size: 19px;
    font-weight: 800;
    color: #0f172a;
}

.brand-sub {
    font-size: 11px;
    color: #64748b;
    margin-top: 1px;
}

.status {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    color: #047857;

    padding: 7px 13px;
    border-radius: 999px;

    font-size: 12px;
    font-weight: 700;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 55%,
            #1d4ed8 100%
        );

    border-radius: 24px;

    padding: 42px 48px;

    margin-bottom: 24px;

    box-shadow:
        0 20px 50px rgba(15, 23, 42, 0.18);
}

.hero::after {
    content: "";

    position: absolute;

    width: 260px;
    height: 260px;

    right: -70px;
    top: -100px;

    border-radius: 50%;

    background: rgba(56, 189, 248, 0.12);
}

.hero-badge {
    display: inline-block;

    background: rgba(255,255,255,0.10);

    border: 1px solid rgba(255,255,255,0.16);

    color: #bfdbfe;

    padding: 7px 13px;

    border-radius: 999px;

    font-size: 12px;
    font-weight: 700;

    margin-bottom: 16px;
}

.hero-title {
    color: white;

    font-size: 38px;

    font-weight: 850;

    line-height: 1.15;

    margin-bottom: 12px;
}

.hero-description {
    color: #cbd5e1;

    font-size: 16px;

    max-width: 720px;

    line-height: 1.65;
}


/* ==========================================================
   SECTION TITLES
   ========================================================== */

.section-title {
    color: #0f172a;

    font-size: 21px;

    font-weight: 800;

    margin-bottom: 5px;
}

.section-subtitle {
    color: #64748b;

    font-size: 13px;

    margin-bottom: 16px;
}


/* ==========================================================
   INPUT CARD
   ========================================================== */

.input-card {
    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 20px;

    padding: 24px;

    box-shadow:
        0 8px 30px rgba(15, 23, 42, 0.06);

    margin-bottom: 22px;
}


/* ==========================================================
   TEXT AREA
   ========================================================== */

textarea {
    border-radius: 14px !important;

    border: 1px solid #cbd5e1 !important;

    background: #f8fafc !important;

    color: #0f172a !important;

    font-size: 15px !important;
}

textarea:focus {
    border: 2px solid #2563eb !important;

    box-shadow:
        0 0 0 3px rgba(37, 99, 235, 0.10) !important;
}


/* ==========================================================
   BUTTON
   ========================================================== */

.stButton > button {

    border: none !important;

    border-radius: 12px !important;

    min-height: 48px !important;

    font-weight: 750 !important;

    font-size: 15px !important;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        ) !important;

    color: white !important;

    box-shadow:
        0 8px 18px rgba(37, 99, 235, 0.22);

    transition: all 0.2s ease;
}

.stButton > button:hover {

    transform: translateY(-1px);

    box-shadow:
        0 12px 25px rgba(37, 99, 235, 0.28);
}


/* ==========================================================
   RESULT
   ========================================================== */

.result-card {

    border-radius: 20px;

    padding: 25px;

    margin: 18px 0 22px 0;

    border: 1px solid;
}

.result-spam {

    background: linear-gradient(
        135deg,
        #fff1f2,
        #fff7f8
    );

    border-color: #fecdd3;
}

.result-safe {

    background: linear-gradient(
        135deg,
        #ecfdf5,
        #f5fffa
    );

    border-color: #bbf7d0;
}

.result-icon {

    width: 54px;
    height: 54px;

    border-radius: 15px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 26px;

    margin-bottom: 13px;
}

.icon-spam {

    background: #ffe4e6;
}

.icon-safe {

    background: #d1fae5;
}

.result-heading {

    font-size: 25px;

    font-weight: 850;

    margin-bottom: 6px;
}

.result-spam .result-heading {
    color: #be123c;
}

.result-safe .result-heading {
    color: #047857;
}

.result-description {

    color: #475569;

    font-size: 14px;

    margin-bottom: 20px;
}

.meta-grid {

    display: grid;

    grid-template-columns:
        repeat(2, minmax(0, 1fr));

    gap: 12px;
}

.meta {

    background: rgba(255,255,255,0.72);

    border: 1px solid rgba(148,163,184,0.22);

    border-radius: 12px;

    padding: 12px 14px;
}

.meta-label {

    color: #64748b;

    font-size: 11px;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.5px;
}

.meta-value {

    color: #0f172a;

    font-size: 16px;

    font-weight: 800;

    margin-top: 3px;
}


/* ==========================================================
   INFO CARDS
   ========================================================== */

.info-card {

    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 17px;

    padding: 20px;

    height: 100%;

    box-shadow:
        0 6px 22px rgba(15,23,42,0.045);
}

.info-icon {

    font-size: 26px;

    margin-bottom: 10px;
}

.info-title {

    color: #0f172a;

    font-size: 15px;

    font-weight: 800;

    margin-bottom: 5px;
}

.info-text {

    color: #64748b;

    font-size: 13px;

    line-height: 1.5;
}


/* ==========================================================
   EXAMPLE BUTTONS
   ========================================================== */

.example-label {

    color: #475569;

    font-size: 12px;

    font-weight: 700;

    margin-bottom: 5px;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {

    text-align: center;

    padding: 30px 10px 10px 10px;

    margin-top: 35px;

    border-top: 1px solid #e2e8f0;

    color: #64748b;

    font-size: 12px;
}

.footer strong {
    color: #334155;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
<div class="topbar">

    <div class="brand">

        <div class="brand-icon">
            🛡️
        </div>

        <div>

            <div class="brand-name">
                ShieldAI
            </div>

            <div class="brand-sub">
                Multilingual Message Security
            </div>

        </div>

    </div>

    <div class="status">
        ● MODEL ONLINE
    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

    <div class="hero-badge">
        🧠 MACHINE LEARNING • NLP • MULTILINGUAL
    </div>

    <div class="hero-title">
        Detect suspicious messages<br>
        before they become a threat.
    </div>

    <div class="hero-description">
        Analyze English, Hindi and Hinglish messages for spam,
        phishing and fraudulent content using a trained
        Linear SVM classification model.
    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# ANALYZER
# ============================================================

st.markdown(
    """
<div class="section-title">
    🔍 Message Analyzer
</div>

<div class="section-subtitle">
    Paste a message below and let the model analyze it.
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

message = st.text_area(
    "Message",
    height=155,
    placeholder=(
        "Example: Congratulations! You have won ₹10,00,000. "
        "Click the link to claim your prize..."
    ),
    label_visibility="collapsed"
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# DETECT
# ============================================================

detect_clicked = st.button(
    "🔎  Analyze Message",
    use_container_width=True
)


# ============================================================
# RESULT
# ============================================================

if detect_clicked:

    if not message.strip():

        st.warning("Please enter a message to analyze.")

    else:

        language, result, confidence = predict_message(message)

        if result == "SPAM / FRAUD":

            st.markdown(
                f"""
<div class="result-card result-spam">

    <div class="result-icon icon-spam">
        🚨
    </div>

    <div class="result-heading">
        SPAM / FRAUD DETECTED
    </div>

    <div class="result-description">
        This message contains patterns associated with
        potentially unwanted, fraudulent or suspicious content.
    </div>

    <div class="meta-grid">

        <div class="meta">

            <div class="meta-label">
                Detected Language
            </div>

            <div class="meta-value">
                🌐 {language}
            </div>

        </div>

        <div class="meta">

            <div class="meta-label">
                Model Confidence
            </div>

            <div class="meta-value">
                {confidence:.1f}%
            </div>

        </div>

    </div>

</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div class="result-card result-safe">

    <div class="result-icon icon-safe">
        ✓
    </div>

    <div class="result-heading">
        LEGITIMATE MESSAGE
    </div>

    <div class="result-description">
        The model classified this message as legitimate
        (Ham) based on the learned message patterns.
    </div>

    <div class="meta-grid">

        <div class="meta">

            <div class="meta-label">
                Detected Language
            </div>

            <div class="meta-value">
                🌐 {language}
            </div>

        </div>

        <div class="meta">

            <div class="meta-label">
                Model Confidence
            </div>

            <div class="meta-value">
                {confidence:.1f}%
            </div>

        </div>

    </div>

</div>
""",
                unsafe_allow_html=True
            )

        st.progress(
            min(confidence / 100, 1.0),
            text=f"Prediction confidence • {confidence:.1f}%"
        )


# ============================================================
# TECHNOLOGY CARDS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
<div class="section-title">
    ⚙️ Detection Pipeline
</div>

<div class="section-subtitle">
    The system combines multilingual preprocessing with
    machine-learning based text classification.
</div>
""",
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

cards = [
    (
        "🌐",
        "3 Languages",
        "English, Hindi and Hinglish message support."
    ),
    (
        "🧹",
        "NLP Preprocessing",
        "Cleaning, normalization, stopword removal and stemming."
    ),
    (
        "🔤",
        "TF-IDF Features",
        "Word-level and character-level text representations."
    ),
    (
        "🧠",
        "Linear SVM",
        "Trained classification model for final prediction."
    )
]

for column, card in zip([c1, c2, c3, c4], cards):

    with column:

        st.markdown(
            f"""
<div class="info-card">

    <div class="info-icon">
        {card[0]}
    </div>

    <div class="info-title">
        {card[1]}
    </div>

    <div class="info-text">
        {card[2]}
    </div>

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# QUICK TESTS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
<div class="section-title">
    💬 Quick Test
</div>

<div class="section-subtitle">
    Try one of these messages by clicking the button.
</div>
""",
    unsafe_allow_html=True
)

q1, q2, q3 = st.columns(3)

with q1:

    if st.button(
        "🚨 Lottery message",
        use_container_width=True
    ):

        st.session_state["sample"] = (
            "Congratulations! You have won a free lottery "
            "ticket worth $1000. Claim now!"
        )

with q2:

    if st.button(
        "🇮🇳 Hindi message",
        use_container_width=True
    ):

        st.session_state["sample"] = (
            "कल मिलते हैं, ठीक है?"
        )

with q3:

    if st.button(
        "💬 Hinglish message",
        use_container_width=True
    ):

        st.session_state["sample"] = (
            "Yaar FREE iPhone jeetne ke liye "
            "is link pe click kar jaldi!"
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📘 About this project"):

    st.markdown(
        """
### Multilingual Phishing & Spam Detection

This system detects spam, phishing, fraudulent and unwanted
messages across **English, Hindi and Hinglish**.

**Machine Learning Model**

- Linear SVM
- Word-level TF-IDF
- Character-level TF-IDF

**Processing**

- Text normalization
- URL and email normalization
- Phone number normalization
- Stopword removal
- Stemming
- TF-IDF feature extraction
- Machine learning classification

The application is designed as an academic demonstration
of multilingual and code-mixed message classification.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

    <strong>🛡️ ShieldAI — Multilingual Message Security</strong>

    <br><br>

    Machine Learning • Natural Language Processing •
    English • Hindi • Hinglish

    <br><br>

    Final Year Academic Project

</div>
""",
    unsafe_allow_html=True
)
