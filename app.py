import streamlit as st
import pickle
import re
import textwrap
from pathlib import Path
import random

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

st.markdown("""
<style>

/* =========================================================
   MAIN CYBERSECURITY BACKGROUND
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 8% 10%,
            rgba(99, 102, 241, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 92% 12%,
            rgba(14, 165, 233, 0.22),
            transparent 30%
        ),
        radial-gradient(
            circle at 80% 90%,
            rgba(168, 85, 247, 0.16),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f8faff 0%,
            #eef4ff 45%,
            #f5f1ff 100%
        );

    min-height: 100vh;
}


/* =========================================================
   REMOVE DEFAULT STREAMLIT ELEMENTS
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* =========================================================
   MAIN CONTENT
   ========================================================= */

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* =========================================================
   TITLE
   ========================================================= */

.main-title {
    color: #111827 !important;

    font-size: 42px !important;

    font-weight: 850 !important;

    letter-spacing: -1.2px;

    line-height: 1.15;
}


.subtitle {
    color: #475569 !important;

    font-size: 16px !important;

    line-height: 1.6;

    margin-bottom: 25px;
}


/* =========================================================
   HORIZONTAL LINES
   ========================================================= */

hr {
    border: none !important;

    border-top: 1px solid rgba(148,163,184,0.28) !important;

    margin: 25px 0 !important;
}


/* =========================================================
   TEXT AREA
   ========================================================= */

textarea {
    background: rgba(255,255,255,0.92) !important;

    border: 1px solid #cbd5e1 !important;

    border-radius: 15px !important;

    color: #0f172a !important;

    font-size: 16px !important;

    box-shadow:
        0 8px 25px rgba(30,64,175,0.06) !important;
}

textarea:focus {
    border: 2px solid #6366f1 !important;

    box-shadow:
        0 0 0 4px rgba(99,102,241,0.12) !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {

    border: 1px solid rgba(148,163,184,0.35) !important;

    border-radius: 13px !important;

    background: rgba(255,255,255,0.90) !important;

    color: #1e293b !important;

    font-weight: 700 !important;

    min-height: 48px !important;

    transition: all 0.2s ease !important;

    box-shadow:
        0 5px 18px rgba(30,64,175,0.06) !important;
}

.stButton > button:hover {

    border-color: #6366f1 !important;

    background: #eef2ff !important;

    transform: translateY(-2px);

    box-shadow:
        0 10px 25px rgba(99,102,241,0.16) !important;
}


/* =========================================================
   ANALYZE BUTTON
   ========================================================= */

.stButton > button[kind="primary"] {

    background:
        linear-gradient(
            90deg,
            #0ea5e9,
            #2563eb,
            #7c3aed
        ) !important;

    border: none !important;

    color: white !important;

    font-size: 17px !important;

    min-height: 52px !important;

    box-shadow:
        0 10px 28px rgba(79,70,229,0.28) !important;
}

.stButton > button[kind="primary"]:hover {

    background:
        linear-gradient(
            90deg,
            #0284c7,
            #1d4ed8,
            #6d28d9
        ) !important;

    transform: translateY(-2px);
}


/* =========================================================
   METRICS
   ========================================================= */

div[data-testid="stMetric"] {

    background: rgba(255,255,255,0.80);

    border: 1px solid rgba(148,163,184,0.25);

    border-radius: 15px;

    padding: 15px;

    box-shadow:
        0 8px 25px rgba(30,64,175,0.06);
}


/* =========================================================
   INFO BOXES
   ========================================================= */

div[data-testid="stAlert"] {

    border-radius: 14px !important;
}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {

    font-weight: 700 !important;

    color: #475569 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {

    color: #4f46e5 !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #0b1120 0%,
            #111c3a 55%,
            #172554 100%
        );
}

section[data-testid="stSidebar"] * {

    color: #f8fafc;
}


/* =========================================================
   SIDEBAR BUTTON
   ========================================================= */

section[data-testid="stSidebar"] .stButton > button {

    background: rgba(255,255,255,0.08) !important;

    border: 1px solid rgba(255,255,255,0.12) !important;

    color: white !important;
}


/* =========================================================
   EXPANDER
   ========================================================= */

div[data-testid="stExpander"] {

    background: rgba(255,255,255,0.72);

    border: 1px solid rgba(148,163,184,0.25);

    border-radius: 15px;
}


/* =========================================================
   CODE EXAMPLES
   ========================================================= */

div[data-testid="stCode"] {

    border-radius: 12px !important;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .main-title {
        font-size: 30px !important;
    }

    .subtitle {
        font-size: 14px !important;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🛡️ ShieldAI"
    )

    st.caption(
        "Multilingual Fraud Detection"
    )

    st.divider()

    st.success(
        "🟢 MODEL ONLINE"
    )

    with st.expander(
        "🌐 Supported Languages",
        expanded=False
    ):

        st.markdown("🇬🇧 **English**")
        st.markdown("🇮🇳 **Hindi**")
        st.markdown("💬 **Hinglish**")

    with st.expander(
        "🤖 Machine Learning",
        expanded=True
    ):

        st.markdown("**Final Model**")

        st.info(
            "🧠 Linear SVM"
        )

        st.markdown("**Features**")

        st.info(
            "🔤 Word TF-IDF + Character TF-IDF"
        )

    with st.expander(
        "💡 How to Use",
        expanded=False
    ):

        st.markdown(
            """
            **1️⃣** Enter a message.

            **2️⃣** Click **Analyze Message**.

            **3️⃣** Check the detected language.

            **4️⃣** View the prediction.

            **5️⃣** Check the confidence.
            """
        )

    with st.expander(
        "📚 Project",
        expanded=False
    ):

        st.markdown(
            """
            **Type**

            Final Year Academic Project

            **Domain**

            Machine Learning + NLP

            **Languages**

            English • Hindi • Hinglish
            """
        )

    st.divider()

    st.caption(
        "🛡️ Multilingual Phishing & Spam Detection"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "# 🕵️‍♂️ Multilingual Phishing & Spam Detection"
)

st.markdown(
    """
    Detect spam, phishing, fraudulent and unwanted messages
    in **English, Hindi and Hinglish** using Machine Learning
    and Natural Language Processing.
    """
)

st.divider()


# ============================================================
# MESSAGE ANALYZER
# ============================================================

st.subheader("🔍 Message Analyzer")

st.caption(
    "Enter a message below and analyze it instantly."
)


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

        st.rerun()


with sample2:

    if st.button(
        "🇮🇳 Hindi Example",
        use_container_width=True
    ):

        st.session_state.sample_message = (
            "कल मिलते हैं, ठीक है?"
        )

        st.rerun()


with sample3:

    if st.button(
        "💬 Hinglish Example",
        use_container_width=True
    ):

        st.session_state.sample_message = (
            "Yaar FREE iPhone jeetne ke liye "
            "is link pe click kar jaldi!"
        )

        st.rerun()


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

            st.error(
                "🚨 SPAM / FRAUD DETECTED"
            )

            st.markdown(
                "**Warning:** The model detected patterns associated "
                "with suspicious or fraudulent messages."
            )

        else:

            st.success(
                "✅ LEGITIMATE MESSAGE"
            )

            st.markdown(
                "**Safe classification:** The model classified this "
                "message as legitimate (Ham)."
            )

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


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.info(
        "🌐 **Multilingual**\n\n"
        "English, Hindi and Hinglish message detection."
    )


with col2:

    st.info(
        "🧹 **NLP Processing**\n\n"
        "Text cleaning, normalization, stopword removal and stemming."
    )


with col3:

    st.info(
        "🔤 **TF-IDF Features**\n\n"
        "Word-level and character-level text features."
    )


with col4:

    st.info(
        "🧠 **Linear SVM**\n\n"
        "Machine learning classification for message detection."
    )


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


# ============================================================
# ABOUT PROJECT
# ============================================================

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


# ============================================================
# SAMPLE MESSAGE BUTTONS
# ============================================================

if "sample_message" not in st.session_state:
    st.session_state.sample_message = ""


spam_examples = [
    "Congratulations! You have won a free lottery ticket worth $1000. Claim now!",
    "URGENT: Your bank account has been blocked. Click the link to verify your account.",
    "You are selected for a ₹50,000 cash prize. Claim your reward immediately!",
    "Congratulations! You won an iPhone 15. Pay ₹999 to receive your prize.",
    "Your KYC has expired. Update your details now to avoid account suspension.",
    "URGENT! You have received a cashback of ₹10,000. Click here to claim.",
    "You have won a lucky draw prize of ₹5,00,000. Send your details to claim.",
    "Your mobile number has won a special reward. Claim it before midnight!"
]


hindi_examples = [
    "कल मिलते हैं, ठीक है?",
    "मुझे आज कॉलेज जाना है।",
    "क्या तुम शाम को मेरे साथ बाजार चलोगे?",
    "आज मौसम बहुत अच्छा है।",
    "माँ ने कहा है कि जल्दी घर आ जाना।",
    "कल हमारी क्लास सुबह दस बजे है।",
    "तुमने खाना खा लिया क्या?",
    "आज शाम को क्रिकेट खेलने चलें?"
]


hinglish_examples = [
    "Yaar FREE iPhone jeetne ke liye is link pe click kar jaldi!",
    "Bhai kal cricket dekhne chalte hain?",
    "Yaar aaj college kab jana hai?",
    "Bhai mujhe kal assignment submit karna hai.",
    "Tum free ho kya? Aaj movie dekhne chalte hain.",
    "Yaar ye offer bahut amazing hai, jaldi check kar!",
    "Bhai kal exam ke liye preparation kiya?",
    "Aaj evening mein chai peene chalte hain?"
]


sample1, sample2, sample3 = st.columns(3)


with sample1:

    if st.button(
        "🚨 Spam Example",
        use_container_width=True
    ):

        st.session_state.sample_message = random.choice(
            spam_examples
        )

        st.rerun()


with sample2:

    if st.button(
        "🇮🇳 Hindi Example",
        use_container_width=True
    ):

        st.session_state.sample_message = random.choice(
            hindi_examples
        )

        st.rerun()


with sample3:

    if st.button(
        "💬 Hinglish Example",
        use_container_width=True
    ):

        st.session_state.sample_message = random.choice(
            hinglish_examples
        )

        st.rerun()

# ============================================================
# TECHNOLOGY
# ============================================================

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

st.divider()

st.caption(
    "🛡️ Multilingual Phishing & Spam Detection"
)

st.caption(
    "Machine Learning • Natural Language Processing"
)

st.caption(
    "English • Hindi • Hinglish"
)

st.caption(
    "Final Year Academic Project"
)
