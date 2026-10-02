import streamlit as st
import pickle
import re
import textwrap
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
# HELPER FOR HTML
# ============================================================

def render_html(content):
    st.markdown(
        textwrap.dedent(content),
        unsafe_allow_html=True
    )


# ============================================================
# NLTK
# ============================================================

nltk.download("stopwords", quiet=True)


# ============================================================
# LOAD MODEL FILES
# ============================================================

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_models():

    with open(BASE_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "tfidf_word.pkl", "rb") as f:
        word_vectorizer = pickle.load(f)

    with open(BASE_DIR / "tfidf_char.pkl", "rb") as f:
        char_vectorizer = pickle.load(f)

    with open(BASE_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, word_vectorizer, char_vectorizer, label_encoder


best_clf, tfidf_word, tfidf_char, encoder = load_models()


# ============================================================
# PREPROCESSING
# ============================================================

ps = PorterStemmer()

english_stops = set(stopwords.words("english"))

hinglish_stops = {
    "hai", "hain", "ho", "tha", "thi", "the",
    "ka", "ki", "ke", "ko", "se", "me", "mein",
    "pe", "par", "aur", "ya", "bhi",
    "yeh", "ye", "woh", "wo", "ek", "koi",
    "kuch", "sab", "apna", "apni", "apne",
    "uska", "uski", "uske",
    "mera", "meri", "mere",
    "tera", "teri", "tere",
    "humara", "tumhara", "unka",
    "kya", "kyun", "kaise", "kab",
    "kahan", "kaun",
    "nahi", "nahin", "mat", "na",
    "ji", "bhai", "yaar", "dost",
    "sir", "madam"
}

all_stops = english_stops | hinglish_stops


def preprocess_multilingual(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\.\S+",
        " url ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " email ",
        text
    )

    text = re.sub(
        r"\b\d{10,}\b",
        " phonenumber ",
        text
    )

    text = re.sub(
        r"₹|rs\.?|inr",
        " rupees ",
        text
    )

    text = re.sub(
        r"£|\$|€",
        " currency ",
        text
    )

    text = re.sub(
        r"[^\w\s\u0900-\u097F]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = text.split()

    cleaned = []

    for tok in tokens:

        if tok in all_stops or len(tok) < 2:
            continue

        if re.match(r"^[a-z]+$", tok):
            tok = ps.stem(tok)

        cleaned.append(tok)

    return " ".join(cleaned)


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

        probabilities = best_clf.predict_proba(X)[0]

        confidence = float(max(probabilities) * 100)

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

render_html("""
<style>

.stApp {
    background-color: #f5f7fb;
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


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: white;
}


/* HERO */

.header-box {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );

    padding: 38px;

    border-radius: 22px;

    margin-bottom: 28px;

    color: white;

    box-shadow:
        0 12px 35px rgba(15, 23, 42, 0.18);
}

.header-title {
    font-size: 36px;
    font-weight: 800;
    line-height: 1.2;
}

.header-text {
    font-size: 16px;
    color: #cbd5e1;
    margin-top: 12px;
    line-height: 1.6;
}


/* SECTION TITLES */

.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
}

.subtitle {
    font-size: 16px;
    color: #64748b;
    margin-bottom: 20px;
}


/* RESULT CARDS */

.spam-result {
    background-color: #fff1f2;

    border: 2px solid #fb7185;

    border-radius: 18px;

    padding: 25px;

    margin-top: 20px;
}

.safe-result {
    background-color: #ecfdf5;

    border: 2px solid #34d399;

    border-radius: 18px;

    padding: 25px;

    margin-top: 20px;
}

.spam-title {
    color: #be123c;

    font-size: 27px;

    font-weight: 800;
}

.safe-title {
    color: #047857;

    font-size: 27px;

    font-weight: 800;
}

.result-text {
    color: #475569;

    font-size: 15px;

    margin-top: 8px;

    line-height: 1.5;
}


/* INFO CARDS */

.info-card {
    background: white;

    padding: 22px;

    border-radius: 16px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 5px 20px rgba(15, 23, 42, 0.05);

    min-height: 145px;
}

.info-icon {
    font-size: 28px;
}

.info-title {
    font-size: 17px;

    font-weight: 700;

    color: #0f172a;

    margin-top: 8px;
}

.info-text {
    font-size: 13px;

    color: #64748b;

    margin-top: 7px;

    line-height: 1.5;
}


/* TEXT AREA */

textarea {
    border-radius: 12px !important;

    border: 1px solid #cbd5e1 !important;

    background-color: white !important;

    font-size: 16px !important;
}


/* BUTTONS */

.stButton > button {
    border-radius: 10px;

    font-weight: 700;

    min-height: 45px;
}


/* FOOTER */

.footer-text {
    text-align: center;

    color: #64748b;

    font-size: 13px;

    padding: 30px;

    margin-top: 35px;

    border-top: 1px solid #e2e8f0;
}

</style>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🛡️ ShieldAI")

    st.markdown(
        "### Multilingual Fraud Detection"
    )

    st.divider()

    st.markdown(
        "### 🌐 Supported Languages"
    )

    st.markdown(
        """
        🇬🇧 **English**

        🇮🇳 **Hindi**

        💬 **Hinglish**
        """
    )

    st.divider()

    st.markdown(
        "### 🤖 Machine Learning"
    )

    st.markdown(
        """
        **Final Model**

        Linear SVM

        **Features**

        Word TF-IDF + Character TF-IDF
        """
    )

    st.divider()

    st.markdown(
        "### 💡 How to Use"
    )

    st.markdown(
        """
        **1.** Enter a message.

        **2.** Click Analyze Message.

        **3.** Check the language.

        **4.** View the prediction.

        **5.** Check the confidence.
        """
    )

    st.divider()

    st.caption(
        "Final Year Academic Project"
    )


# ============================================================
# HEADER
# ============================================================

render_html("""
<div class="header-box">

    <div class="header-title">
        🛡️ Multilingual Phishing & Spam Detection
    </div>

    <div class="header-text">
        Detect spam, phishing, fraudulent and unwanted messages
        in English, Hindi and Hinglish using Machine Learning
        and Natural Language Processing.
    </div>

</div>
""")


# ============================================================
# MESSAGE ANALYZER
# ============================================================

render_html("""
<div class="main-title">
    🔍 Message Analyzer
</div>

<div class="subtitle">
    Enter a message below and analyze it instantly.
</div>
""")


# ============================================================
# SAMPLE MESSAGE BUTTONS
# ============================================================

if "sample_message" not in st.session_state:
    st.session_state.sample_message = ""


sample1, sample2, sample3 = st.columns(3)


with sample1:

    if st.button(
        "🚨 Spam Example",
        use_container_width=True
    ):

        st.session_state.sample_message = (
            "Congratulations! You have won a free lottery "
            "ticket worth $1000. Claim now!"
        )


with sample2:

    if st.button(
        "🇮🇳 Hindi Example",
        use_container_width=True
    ):

        st.session_state.sample_message = (
            "कल मिलते हैं, ठीक है?"
        )


with sample3:

    if st.button(
        "💬 Hinglish Example",
        use_container_width=True
    ):

        st.session_state.sample_message = (
            "Yaar FREE iPhone jeetne ke liye "
            "is link pe click kar jaldi!"
        )


# ============================================================
# MESSAGE INPUT
# ============================================================

message = st.text_area(
    "Enter your message",
    value=st.session_state.sample_message,
    height=160,
    placeholder=(
        "Example: Congratulations! You have won ₹10,00,000..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔎 Analyze Message",
    type="primary",
    use_container_width=True
)


# ============================================================
# RESULT
# ============================================================

if analyze:

    if not message.strip():

        st.warning(
            "⚠️ Please enter a message before analyzing."
        )

    else:

        language, result, confidence = predict_message(
            message
        )

        st.markdown("---")

        st.subheader(
            "📊 Prediction Result"
        )

        if result == "SPAM / FRAUD":

            render_html("""
            <div class="spam-result">

                <div class="spam-title">
                    🚨 SPAM / FRAUD DETECTED
                </div>

                <div class="result-text">
                    The model detected patterns associated
                    with suspicious or fraudulent messages.
                </div>

            </div>
            """)

        else:

            render_html("""
            <div class="safe-result">

                <div class="safe-title">
                    ✅ LEGITIMATE MESSAGE
                </div>

                <div class="result-text">
                    The model classified this message as
                    legitimate (Ham).
                </div>

            </div>
            """)

        st.write("")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "🌐 Detected Language",
                language
            )

        with result_col2:

            st.metric(
                "📊 Confidence",
                f"{confidence:.1f}%"
            )

        st.progress(
            min(confidence / 100, 1.0),
            text=f"Model confidence: {confidence:.1f}%"
        )


# ============================================================
# DETECTION PIPELINE
# ============================================================

st.markdown("---")

st.subheader(
    "⚙️ Detection Pipeline"
)

st.caption(
    "The system combines multilingual preprocessing "
    "with machine-learning based classification."
)


feature1, feature2, feature3, feature4 = st.columns(4)


with feature1:

    render_html("""
    <div class="info-card">

        <div class="info-icon">
            🌐
        </div>

        <div class="info-title">
            Multilingual
        </div>

        <div class="info-text">
            English, Hindi and Hinglish
            message detection.
        </div>

    </div>
    """)


with feature2:

    render_html("""
    <div class="info-card">

        <div class="info-icon">
            🧹
        </div>

        <div class="info-title">
            NLP Processing
        </div>

        <div class="info-text">
            Text cleaning, normalization,
            stopword removal and stemming.
        </div>

    </div>
    """)


with feature3:

    render_html("""
    <div class="info-card">

        <div class="info-icon">
            🔤
        </div>

        <div class="info-title">
            TF-IDF Features
        </div>

        <div class="info-text">
            Word-level and character-level
            text features.
        </div>

    </div>
    """)


with feature4:

    render_html("""
    <div class="info-card">

        <div class="info-icon">
            🧠
        </div>

        <div class="info-title">
            Linear SVM
        </div>

        <div class="info-text">
            Machine learning classification
            for message detection.
        </div>

    </div>
    """)


# ============================================================
# INFORMATION TABS
# ============================================================

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "📘 About Project",
        "🧪 Example Messages",
        "🔧 Technology"
    ]
)


with tab1:

    st.markdown(
        """
        ### Multilingual Phishing & Spam Detection

        This application detects spam, phishing, fraudulent
        and unwanted messages across:

        - 🇬🇧 English
        - 🇮🇳 Hindi
        - 💬 Hinglish

        The system uses machine learning and NLP techniques
        to process and classify messages.
        """
    )


with tab2:

    st.markdown(
        "### 🚨 Suspicious Examples"
    )

    st.code(
        "Congratulations! You have won a free lottery ticket. Claim now!",
        language=None
    )

    st.code(
        "URGENT: Your SBI account is blocked. Share OTP to verify.",
        language=None
    )

    st.code(
        "Yaar FREE iPhone jeetne ke liye is link pe click kar jaldi!",
        language=None
    )

    st.markdown(
        "### ✅ Legitimate Examples"
    )

    st.code(
        "Hey, are you free for dinner tonight?",
        language=None
    )

    st.code(
        "कल मिलते हैं, ठीक है?",
        language=None
    )

    st.code(
        "Bhai kal cricket dekhne chalte hain?",
        language=None
    )


with tab3:

    st.markdown(
        """
        ### Technology Stack

        **Machine Learning**

        Linear SVM

        **Feature Extraction**

        Word TF-IDF + Character TF-IDF

        **Natural Language Processing**

        Text normalization, stopword removal
        and stemming.

        **Deployment**

        Streamlit Community Cloud
        """
    )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer-text">

    🛡️ <b>Multilingual Phishing & Spam Detection</b>

    <br><br>

    Machine Learning • Natural Language Processing

    <br>

    English • Hindi • Hinglish

    <br><br>

    Final Year Academic Project

</div>
""")
