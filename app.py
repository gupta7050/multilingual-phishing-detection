import streamlit as st
import pickle
import re
import random
import inspect
from pathlib import Path
import requests

import numpy as np
from scipy.sparse import hstack
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multilingual Phishing & Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BACKEND API CONFIGURATION
# ============================================================

try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = "http://127.0.0.1:8000"

API_URL = API_URL.rstrip("/")


# ============================================================
# COMPATIBILITY: button width (old and new Streamlit versions)
# ============================================================

if "width" in inspect.signature(st.button).parameters:
    STRETCH = {"width": "stretch"}
else:
    STRETCH = {"use_container_width": True}


# ============================================================
# SETTINGS
# ============================================================

# IMPORTANT: this must match how your TRAINING notebook preprocessed text.
# Keep False if your saved .pkl files were trained with the original
# preprocessing. Set True only after retraining with the fixed version.
USE_FIXED_PREPROCESSING = False


# ============================================================
# NLTK
# ============================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

ps = PorterStemmer()

english_stops = set(
    stopwords.words("english")
)

hinglish_stops = {
    "hai", "hain", "ho", "tha", "thi", "the",
    "ka", "ki", "ke", "ko", "se", "me", "mein",
    "pe", "par", "aur", "ya", "bhi", "yeh", "ye",
    "woh", "wo", "ek", "koi", "kuch", "sab",
    "apna", "apni", "apne", "uska", "uski", "uske",
    "mera", "meri", "mere", "tera", "teri", "tere",
    "humara", "tumhara", "unka", "kya", "kyun",
    "kaise", "kab", "kahan", "kaun", "nahi", "nahin",
    "mat", "na", "ji", "bhai", "yaar", "dost",
    "sir", "madam"
}

all_stops = english_stops | hinglish_stops

# Romanized Hindi words used to detect Hinglish written in Latin script
HINGLISH_MARKERS = (hinglish_stops - english_stops) | {
    "karo", "kare", "bhejo", "jeeta", "jeete", "abhi", "tumhara",
    "tumhare", "wala", "rahe", "mila", "batao", "kijiye", "tumne",
    "tumhe", "naam", "liye", "jaldi", "sirf", "aaj", "pehle"
}


# ============================================================
# SESSION STATE
# ============================================================

if "message" not in st.session_state:
    st.session_state.message = ""

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_email" not in st.session_state:
    st.session_state.user_email = ""


# ============================================================
# LOAD MODELS (cached, loaded only once)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():

    with open(BASE_DIR / "tfidf_word.pkl", "rb") as f:
        tfidf_word = pickle.load(f)

    with open(BASE_DIR / "tfidf_char.pkl", "rb") as f:
        tfidf_char = pickle.load(f)

    with open(BASE_DIR / "best_model.pkl", "rb") as f:
        best_clf = pickle.load(f)

    with open(BASE_DIR / "label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return tfidf_word, tfidf_char, best_clf, encoder


try:

    tfidf_word, tfidf_char, best_clf, encoder = load_models()

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# ============================================================
# PREPROCESSING
# ============================================================

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

    if USE_FIXED_PREPROCESSING:
        # Word boundaries so "hours", "users", "first" are not corrupted
        text = re.sub(
            r"₹|\brs\b\.?|\binr\b",
            " rupees ",
            text
        )
    else:
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

    if USE_FIXED_PREPROCESSING:
        # Remove the Devanagari danda / double danda
        text = re.sub(
            r"[\u0964\u0965]",
            " ",
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

        if re.match(
            r"^[a-z]+$",
            tok
        ):
            tok = ps.stem(tok)

        cleaned.append(tok)

    return " ".join(cleaned)


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    text = str(text)

    hindi = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))

    tokens = re.findall(r"[a-z]+", text.lower())
    marker_hits = sum(t in HINGLISH_MARKERS for t in tokens)

    # Devanagari mixed with a lot of Latin text
    if hindi and latin and hindi / (hindi + latin) < 0.8:
        return "Hinglish"

    # Mostly Devanagari (a few words like KYC / OTP are still Hindi)
    if hindi:
        return "Hindi"

    # Romanized Hindi mixed with English
    if latin and marker_hits >= 2:
        return "Hinglish"

    if latin:
        return "English"

    return "Unknown"


# ============================================================
# LABEL HELPERS
# ============================================================

SPAM_LABELS = {"spam", "phishing", "fraud", "scam", "smishing", "1"}
HAM_LABELS = {
    "ham", "legitimate", "legit", "safe", "normal",
    "not_spam", "not spam", "non-spam", "non_spam", "nonspam", "0"
}


def label_to_result(label_text):

    label_text = str(label_text).strip().lower()

    if label_text in HAM_LABELS:
        return "LEGITIMATE"

    if label_text in SPAM_LABELS:
        return "SPAM / FRAUD"

    # Unknown label name: negated forms are legitimate
    if label_text.startswith(("not", "non", "no_", "no ")):
        return "LEGITIMATE"

    if any(k in label_text for k in ("spam", "fraud", "phish", "scam")):
        return "SPAM / FRAUD"

    return "LEGITIMATE"


# ============================================================
# BACKEND API FUNCTIONS
# ============================================================

def login_user(email, password):

    try:
        response = requests.post(
            f"{API_URL}/login",
            json={
                "email": email,
                "password": password
            },
            timeout=30
        )

        if response.status_code == 200:
            return True, response.json()

        try:
            return False, response.json().get(
                "detail",
                "Invalid email or password."
            )
        except Exception:
            return False, "Invalid email or password."

    except requests.exceptions.RequestException as e:
        return False, f"Backend connection error: {e}"


def register_user(name, email, password):

    try:
        response = requests.post(
            f"{API_URL}/register",
            json={
                "name": name,
                "email": email,
                "password": password
            },
            timeout=30
        )

        if response.status_code in (200, 201):
            return True, response.json()

        try:
            return False, response.json().get(
                "detail",
                "Registration failed."
            )
        except Exception:
            return False, "Registration failed."

    except requests.exceptions.RequestException as e:
        return False, f"Backend connection error: {e}"


def get_history(user_id):

    try:
        response = requests.get(
            f"{API_URL}/history/{user_id}",
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        return []

    except requests.exceptions.RequestException:
        return []


def predict_from_backend(message, user_id):

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "message": message,
                "user_id": user_id
            },
            timeout=60
        )

        if response.status_code != 200:
            try:
                detail = response.json().get(
                    "detail",
                    "Prediction failed."
                )
            except Exception:
                detail = "Prediction failed."

            st.error(f"❌ {detail}")
            return None, 0.0, "Unknown"

        data = response.json()

        return (
            data.get("prediction"),
            float(data.get("confidence", 0)),
            data.get("language", "Unknown")
        )

    except requests.exceptions.RequestException as e:
        st.error(
            "❌ Could not connect to the FastAPI backend. "
            f"Please make sure the backend is running.\n\n{e}"
        )
        return None, 0.0, "Unknown"


# ============================================================
# PREDICTION
# ============================================================
def predict_message(message):

    language = detect_language(message)

    if not message.strip():
        return language, None, 0.0

    if not st.session_state.logged_in:
        return language, None, 0.0

    result, confidence, backend_language = predict_from_backend(
        message,
        st.session_state.user_id
    )

    if backend_language and backend_language != "Unknown":
        language = backend_language

    return language, result, confidence


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       MAIN HACKER / CYBERSECURITY BACKGROUND
       ====================================================== */

    .stApp {

        background:
        radial-gradient(
            circle at 15% 20%,
            rgba(0, 255, 136, 0.10),
            transparent 28%
        ),

        radial-gradient(
            circle at 85% 80%,
            rgba(0, 255, 136, 0.08),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #020806 0%,
            #06130d 45%,
            #020807 100%
        ) !important;

        color:
        #e8fff4 !important;
    }


    /* ======================================================
       CYBER GRID
       ====================================================== */

    .stApp::before {

        content: "";

        position: fixed;

        inset: 0;

        background-image:

        linear-gradient(
            rgba(0, 255, 136, 0.035) 1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(0, 255, 136, 0.035) 1px,
            transparent 1px
        );

        background-size:
        40px 40px;

        pointer-events:
        none;

        z-index:
        0;
    }


    /* ======================================================
       MAIN CONTENT
       ====================================================== */

    .main .block-container,
    [data-testid="stMainBlockContainer"] {

        position:
        relative;

        z-index:
        1;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {

        background:
        linear-gradient(
            180deg,
            #010504 0%,
            #03150d 50%,
            #020a06 100%
        ) !important;

        border-right:
        1px solid
        rgba(0, 255, 136, 0.20);
    }

    section[data-testid="stSidebar"] * {

        color:
        #d9ffeb;
    }


    /* ======================================================
       SIDEBAR EXPANDERS
       ====================================================== */

    section[data-testid="stSidebar"]
    div[data-testid="stExpander"] {

        background:
        rgba(3, 25, 16, 0.90) !important;

        border:
        1px solid
        rgba(0, 255, 136, 0.25) !important;

        border-radius:
        15px !important;

        margin-bottom:
        10px !important;

        box-shadow:
        0 0 12px
        rgba(0, 255, 136, 0.06) !important;
    }


    section[data-testid="stSidebar"]
    div[data-testid="stExpander"]
    summary {

        color:
        #00ff88 !important;

        font-weight:
        600 !important;
    }


    section[data-testid="stSidebar"]
    div[data-testid="stExpander"]:hover {

        background:
        rgba(5, 45, 27, 0.95) !important;

        border-color:
        rgba(0, 255, 136, 0.65) !important;

        box-shadow:
        0 0 15px
        rgba(0, 255, 136, 0.15) !important;
    }


    /* ======================================================
       SIDEBAR INFO BOX
       ====================================================== */

    section[data-testid="stSidebar"]
    div[data-testid="stAlert"] {

        background:
        rgba(0, 45, 25, 0.75) !important;

        border:
        1px solid
        rgba(0, 255, 136, 0.30) !important;
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1 {

        color:
        #00ff88 !important;

        font-weight:
        800 !important;

        text-shadow:
        0 0 8px
        rgba(0, 255, 136, 0.45),

        0 0 20px
        rgba(0, 255, 136, 0.20) !important;
    }


    h2,
    h3 {

        color:
        #e8fff4 !important;

        font-weight:
        700 !important;
    }


    /* ======================================================
       NORMAL TEXT
       ====================================================== */

    p {

        color:
        #b7d9c8;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {

        border-radius:
        12px !important;

        border:
        1px solid
        rgba(0, 255, 136, 0.45) !important;

        background:
        linear-gradient(
            135deg,
            #071b12,
            #0b281a
        ) !important;

        color:
        #00ff88 !important;

        font-weight:
        600 !important;

        min-height:
        44px !important;

        box-shadow:
        0 0 10px
        rgba(0, 255, 136, 0.08) !important;

        transition:
        all 0.2s ease !important;
    }


    .stButton > button:hover {

        background:
        #0b2f1d !important;

        color:
        #ffffff !important;

        border-color:
        #00ff88 !important;

        box-shadow:
        0 0 18px
        rgba(0, 255, 136, 0.30) !important;

        transform:
        translateY(-1px);
    }


    /* ======================================================
       TEXT AREA
       ====================================================== */

    textarea {

        border-radius:
        14px !important;

        border:
        1px solid
        rgba(0, 255, 136, 0.45) !important;

        background:
        #020b07 !important;

        color:
        #eafff3 !important;

        font-size:
        16px !important;

        box-shadow:
        0 0 12px
        rgba(0, 255, 136, 0.08) !important;
    }


    textarea:focus {

        border-color:
        #00ff88 !important;

        box-shadow:
        0 0 15px
        rgba(0, 255, 136, 0.25) !important;
    }


    /* ======================================================
       TEXT AREA LABEL
       ====================================================== */

    label {

        color:
        #9de8bd !important;

        font-weight:
        600 !important;
    }


    /* ======================================================
       ALERTS
       ====================================================== */

    div[data-testid="stAlert"] {

        background:
        rgba(3, 25, 16, 0.90) !important;

        border:
        1px solid
        rgba(0, 255, 136, 0.25) !important;

        color:
        #d9ffeb !important;

        border-radius:
        12px !important;
    }


    /* ======================================================
       METRICS
       ====================================================== */

    div[data-testid="stMetric"] {

        background:
        rgba(3, 20, 13, 0.85) !important;

        border:
        1px solid
        rgba(0, 255, 136, 0.25) !important;

        border-radius:
        14px !important;

        padding:
        12px !important;

        box-shadow:
        0 0 15px
        rgba(0, 255, 136, 0.08) !important;
    }


    div[data-testid="stMetricLabel"] {

        color:
        #8fbda5 !important;
    }


    div[data-testid="stMetricValue"] {

        color:
        #00ff88 !important;
    }


    /* ======================================================
       PROGRESS BAR
       ====================================================== */

    div[data-testid="stProgressBar"] {

        background:
        #092016 !important;
    }


    div[data-testid="stProgressBar"]
    div[role="progressbar"] {

        background:
        #00ff88 !important;

        box-shadow:
        0 0 10px
        rgba(0, 255, 136, 0.35);
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {

        color:
        #9de8bd !important;

        font-weight:
        600 !important;
    }


    button[data-baseweb="tab"][aria-selected="true"] {

        color:
        #00ff88 !important;
    }


    /* ======================================================
       DIVIDERS
       ====================================================== */

    hr {

        border-color:
        rgba(0, 255, 136, 0.15) !important;
    }


    /* ======================================================
       STREAMLIT CLOUD TOP BAR
       Keep ONLY Share and the three-dot menu.
       Hide Star, Edit (pen) and GitHub icons.
       ====================================================== */

    /* GitHub icon */
    #GithubIcon,
    header a[href*="github.com"],
    [data-testid="stToolbar"] a[href*="github.com"],
    [data-testid="stToolbarActions"] a[href*="github.com"],
    .stAppToolbar a[href*="github.com"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Star / Favorite icon */
    header [aria-label*="star" i],
    header [title*="star" i],
    header [data-testid*="star" i],
    header [data-testid*="favorite" i],
    [data-testid="stToolbar"] [aria-label*="star" i],
    [data-testid="stToolbar"] [title*="star" i],
    [data-testid="stToolbar"] [data-testid*="star" i],
    [data-testid="stToolbar"] [data-testid*="favorite" i],
    [data-testid="stToolbarActions"] [aria-label*="star" i],
    [data-testid="stToolbarActions"] [title*="star" i],
    .stAppToolbar [aria-label*="star" i],
    .stAppToolbar [title*="star" i] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Edit / pen icon */
    header [aria-label*="edit" i],
    header [title*="edit" i],
    header [data-testid*="edit" i],
    [data-testid="stToolbar"] [aria-label*="edit" i],
    [data-testid="stToolbar"] [title*="edit" i],
    [data-testid="stToolbar"] [data-testid*="edit" i],
    [data-testid="stToolbarActions"] [aria-label*="edit" i],
    [data-testid="stToolbarActions"] [title*="edit" i],
    .stAppToolbar [aria-label*="edit" i],
    .stAppToolbar [title*="edit" i] {
        display: none !important;
        visibility: hidden !important;
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
        "## 🛡️ ShieldAI"
    )

    st.caption(
        "Multilingual Fraud Detection"
    )

    st.divider()

    st.success(
        "MODEL ONLINE"
    )


    # --------------------------------------------------------
    # SUPPORTED LANGUAGES
    # --------------------------------------------------------

    with st.expander(
        "Supported Languages",
        expanded=False
    ):

        st.markdown(
            "**English**"
        )

        st.markdown(
            "**हिंदी**"
        )

        st.markdown(
            "**Hinglish**"
        )


    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    with st.expander(
        "Machine Learning",
        expanded=True
    ):

        st.markdown(
            "**Final Model**"
        )

        st.info(
            "Linear SVM"
        )

        st.markdown(
            "**Features**"
        )

        st.info(
            "Word TF-IDF + Character TF-IDF"
        )


    # --------------------------------------------------------
    # HOW TO USE
    # --------------------------------------------------------

    with st.expander(
        "How to Use",
        expanded=False
    ):

        st.markdown(
            """
            **1.** Enter a message.

            **2.** Click **Analyze Message**.

            **3.** Check the detected language.

            **4.** View the prediction.

            **5.** Check the confidence.
            """
        )


    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    with st.expander(
        "Project",
        expanded=False
    ):

        st.markdown(
            """
            **Type**

            Final Year Academic Project

            **Domain**

            AI-Driven Cybersecurity & Multilingual NLP

            **Languages**

            English • Hindi • Hinglish
            """
        )


    st.divider()

    st.caption(
        "Multilingual Phishing & Spam Detection"
    )


# ============================================================
# HEADER + LOGIN
# ============================================================

header_col, login_col = st.columns([8.2, 1.8], vertical_alignment="top")

with header_col:

    st.markdown(
        "# 🛡️ Multilingual Phishing & Spam Detection"
    )

    st.caption(
        "AI-powered detection of suspicious messages "
        "across English, Hindi and Hinglish."
    )

with login_col:

    if not st.session_state.logged_in:

        with st.popover("🔐 Login", use_container_width=True):

            login_tab, register_tab = st.tabs(
                ["Login", "Create Account"]
            )

            with login_tab:

                st.markdown("### 🔐 Login")

                login_email = st.text_input(
                    "Email",
                    key="login_email"
                )

                login_password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password"
                )

                if st.button(
                    "Login",
                    type="primary",
                    key="login_submit",
                    use_container_width=True
                ):

                    if not login_email or not login_password:

                        st.warning(
                            "Please enter email and password."
                        )

                    else:

                        success, data = login_user(
                            login_email,
                            login_password
                        )

                        if success:

                            st.session_state.logged_in = True
                            st.session_state.user_id = data["user_id"]
                            st.session_state.user_name = data["name"]
                            st.session_state.user_email = data["email"]

                            st.rerun()

                        else:

                            st.error(str(data))

            with register_tab:

                st.markdown("### 📝 Create Account")

                register_name = st.text_input(
                    "Full Name",
                    key="register_name"
                )

                register_email = st.text_input(
                    "Email",
                    key="register_email"
                )

                register_password = st.text_input(
                    "Password",
                    type="password",
                    key="register_password"
                )

                register_confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    key="register_confirm"
                )

                if st.button(
                    "Create Account",
                    type="primary",
                    key="register_submit",
                    use_container_width=True
                ):

                    if not register_name or not register_email or not register_password:

                        st.warning(
                            "Please fill all fields."
                        )

                    elif register_password != register_confirm:

                        st.error(
                            "Passwords do not match."
                        )

                    else:

                        success, data = register_user(
                            register_name,
                            register_email,
                            register_password
                        )

                        if success:

                            st.success(
                                "Account created successfully. "
                                "You can now login."
                            )

                        else:

                            st.error(str(data))

    else:

        with st.popover(
            f"👤 {st.session_state.user_name}",
            use_container_width=True
        ):

            st.markdown("### 👤 Account")
            st.caption(st.session_state.user_email)

            if st.button(
                "🚪 Logout",
                key="logout_button",
                use_container_width=True
            ):

                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.user_name = ""
                st.session_state.user_email = ""

                st.rerun()


# ============================================================
# MESSAGE ANALYZER
# ============================================================

st.subheader(
    "🔍 Message Analyzer"
)

st.caption(
    "Enter a message below and analyze it instantly."
)


# ============================================================
# SAMPLE MESSAGES
# ============================================================

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


legit_examples = [

    "Hi, are we still meeting for lunch tomorrow at 1 pm?",

    "Your order has been shipped and will arrive by Friday. Thank you for shopping with us.",

    "Reminder: your dentist appointment is scheduled for Monday at 10:30 AM.",

    "Please find the meeting notes attached. Let me know if I missed anything.",

    "Happy birthday! Hope you have a wonderful day with family and friends.",

    "Bhai kal shaam ko cricket khelne chalna hai, tum aa rahe ho?",

    "Mummy ne kaha hai ki aaj dinner ghar par karna hai.",

    "कल की मीटिंग शाम पाँच बजे है, कृपया समय पर पहुँचें।"

]


hindi_examples = [

    "आपका बैंक खाता बंद होने वाला है। तुरंत KYC अपडेट करने के लिए इस लिंक पर क्लिक करें।",

    "बधाई हो! आपने ₹5,00,000 की लॉटरी जीती है। इनाम पाने के लिए अपना OTP भेजें।",

    "आपके मोबाइल नंबर पर ₹10,000 का कैशबैक मिला है। अभी लिंक पर क्लिक करके दावा करें।",

    "आपका बिजली बिल बकाया है। कनेक्शन कटने से बचने के लिए तुरंत भुगतान करें।",

    "आपका बैंक अकाउंट सस्पेंड कर दिया गया है। सत्यापन के लिए अपना ATM PIN और OTP साझा करें।",

    "आपके नाम पर एक पार्सल आया है। डिलीवरी पूरी करने के लिए ₹50 का शुल्क जमा करें।",

    "आपका KYC समाप्त हो गया है। बैंक खाता बंद होने से बचाने के लिए अभी अपनी जानकारी अपडेट करें।",

    "आपको सरकारी योजना के तहत ₹25,000 की सहायता राशि मिली है। प्राप्त करने के लिए बैंक विवरण भेजें।",

    "आपका क्रेडिट कार्ड ब्लॉक होने वाला है। इसे सक्रिय करने के लिए इस लिंक पर क्लिक करें।",

    "आपने ऑनलाइन प्रतियोगिता में पहला पुरस्कार जीता है। पुरस्कार पाने के लिए अपना बैंक विवरण दें।",

    "आपके खाते में संदिग्ध गतिविधि पाई गई है। तुरंत OTP बताकर अपना खाता सत्यापित करें।",

    "आज आखिरी मौका है! ₹1 लाख का इनाम पाने के लिए अभी अपना मोबाइल नंबर और बैंक विवरण भेजें।",

    "आपका SIM कार्ड बंद होने वाला है। KYC सत्यापन के लिए इस लिंक पर अपनी जानकारी भरें।",

    "बधाई हो! आपका नंबर lucky draw में चुना गया है। ₹2 लाख का इनाम पाने के लिए processing fee जमा करें।",

    "आपके बैंक खाते में ₹15,000 जमा किए गए हैं। राशि प्राप्त करने के लिए पहले ₹500 का शुल्क दें।",

    "आपका PAN कार्ड अपडेट नहीं है। खाता बंद होने से बचाने के लिए तुरंत अपना PAN और OTP साझा करें।",

    "आपके नाम से एक loan मंजूर हुआ है। पैसे प्राप्त करने के लिए पहले registration fee जमा करें।",

    "आपका UPI बंद होने वाला है। दोबारा सक्रिय करने के लिए अपना UPI PIN दर्ज करें।",

    "आपके खाते में cashback pending है। उसे प्राप्त करने के लिए दिए गए लिंक पर क्लिक करें।",

    "आपको मुफ्त मोबाइल फोन मिला है। डिलीवरी के लिए ₹999 का शुल्क जमा करें।"

]


hinglish_examples = [

    "Bhai tumne ₹50,000 ka lottery prize jeeta hai, claim karne ke liye OTP bhejo!",

    "Your bank account KYC expire ho gaya hai, account block hone se bachne ke liye link pe click karo.",

    "Congratulations! Tumhare number par ₹10,000 cashback mila hai, abhi claim karo.",

    "Bhai tumhara bank account suspend hone wala hai, verification ke liye OTP share karo.",

    "Yaar tumne lucky draw mein iPhone jeeta hai, delivery ke liye ₹999 payment karo.",

    "Tumhara electricity bill pending hai, connection cut hone se bachne ke liye abhi payment karo.",

    "Bhai tumhare naam pe ek parcel aaya hai, delivery complete karne ke liye ₹50 fee pay karo.",

    "Tumhara KYC update nahi hai, account close hone se pehle apna Aadhaar aur OTP send karo.",

    "Congratulations bhai! Tumhe ₹2 lakh ka reward mila hai, processing fee pay karke claim karo.",

    "Yaar tumhara SIM band hone wala hai, KYC verify karne ke liye is link pe click karo.",

    "Bhai tumhare account mein suspicious activity mili hai, verification ke liye OTP batao.",

    "Tumhe government scheme ke through ₹25,000 mil rahe hain, bank details send karo.",

    "Bhai tumhara credit card block hone wala hai, activate karne ke liye link open karo.",

    "FREE cashback jeetne ka chance hai! Bas apna UPI PIN enter karo aur reward claim karo.",

    "Yaar tumhara loan approve ho gaya hai, amount receive karne ke liye registration fee pay karo.",

    "Bhai tumhare mobile number ne lucky draw jeeta hai, prize lene ke liye processing charges do.",

    "Tumhara PAN update nahi hai, bank account safe rakhne ke liye PAN aur OTP share karo.",

    "Bhai ₹1 lakh ka special offer mila hai, sirf aaj claim karna hai. Jaldi link pe click karo.",

    "Your UPI account suspend hone wala hai, reactivate karne ke liye UPI PIN enter karo.",

    "Yaar tumhare account mein cashback pending hai, receive karne ke liye verification complete karo."

]


# ============================================================
# SAMPLE BUTTON CALLBACKS
# (callbacks run before the text area is drawn, so no st.rerun()
#  is needed and the text box always gets replaced)
# ============================================================

def set_sample(examples):
    st.session_state.message = random.choice(examples)


# ============================================================
# SAMPLE BUTTONS
# ============================================================

sample1, sample2, sample3, sample4 = st.columns(4)


with sample1:

    st.button(
        "🚨 Spam Example",
        on_click=set_sample,
        args=(spam_examples,),
        **STRETCH
    )


with sample2:

    st.button(
        "🇮🇳 Hindi Example",
        on_click=set_sample,
        args=(hindi_examples,),
        **STRETCH
    )


with sample3:

    st.button(
        "💬 Hinglish Example",
        on_click=set_sample,
        args=(hinglish_examples,),
        **STRETCH
    )


with sample4:

    st.button(
        "✅ Legitimate Example",
        on_click=set_sample,
        args=(legit_examples,),
        **STRETCH
    )


# ============================================================
# MESSAGE INPUT
# ============================================================

message = st.text_area(
    "Message",
    key="message",
    height=160,
    placeholder=(
        "Example: Congratulations! You have won a prize. "
        "Click the link to claim."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔎 Analyze Message",
    type="primary",
    **STRETCH
)


# ============================================================
# RESULT
# ============================================================

if analyze:

    if not st.session_state.logged_in:

        st.warning(
            "🔐 Please login before analyzing a message."
        )

    elif not message.strip():

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

        if result is None:

            st.warning(
                "⚠️ The message is too short or contains only common "
                "words, so there is not enough content to analyze. "
                "Please enter a longer message."
            )

        else:

            if result == "SPAM / FRAUD":

                st.error(
                    "🚨 SPAM / FRAUD DETECTED"
                )

                st.markdown(
                    "**Warning:** The model detected patterns "
                    "associated with suspicious or fraudulent messages."
                )

            else:

                st.success(
                    "✅ LEGITIMATE MESSAGE"
                )

                st.markdown(
                    "**Result:** No fraud patterns were detected, "
                    "but always stay cautious with links and requests "
                    "for personal information."
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
                    "🎯 Confidence",
                    f"{confidence:.1f}%"
                )

            st.progress(
                min(confidence / 100, 1.0),
                text=(
                    f"Model confidence: "
                    f"{confidence:.1f}%"
                )
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
        "🌍 **Multilingual**\n\n"
        "English, Hindi and Hinglish message detection."
    )


with col2:

    st.info(
        "🧹 **NLP Processing**\n\n"
        "Text cleaning, normalization, stopword removal "
        "and stemming."
    )


with col3:

    st.info(
        "📐 **TF-IDF Features**\n\n"
        "Word-level and character-level text features."
    )


with col4:

    st.info(
        "🤖 **Linear SVM**\n\n"
        "Machine learning classification for message detection."
    )


# ============================================================
# INFORMATION TABS
# ============================================================

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "About the Project",
        "How It Works",
        "Supported Languages"
    ]
)


with tab1:

    st.markdown(
        """
        ### Multilingual Phishing & Spam Detection

        This project uses Natural Language Processing
        and Machine Learning to detect suspicious messages.

        The system is designed for:

        - Spam detection
        - Phishing detection
        - Fraudulent message detection
        - Multilingual text classification
        - Code-mixed Hinglish messages

        **Final Model:** Linear SVM

        **Feature Engineering:**
        Word TF-IDF + Character TF-IDF
        """
    )


with tab2:

    st.markdown(
        """
        ### Detection Process

        **1. Text Input**

        The user enters a message.

        **2. Language Detection**

        The system identifies English, Hindi or Hinglish.

        **3. Text Preprocessing**

        URLs, emails, phone numbers, currencies and
        unnecessary characters are normalized.

        **4. Feature Extraction**

        Word-level and character-level TF-IDF features
        are generated.

        **5. Classification**

        Linear SVM predicts whether the message is
        legitimate or suspicious.

        **6. Result**

        The application displays the prediction,
        detected language and confidence.
        """
    )


with tab3:

    st.markdown(
        """
        ### Supported Languages

        **English**

        English-language messages.

        **हिंदी**

        Hindi-language messages written in Devanagari.

        **Hinglish**

        Hindi and English words used together.
        """
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.markdown("---")

st.subheader("📜 Prediction History")

if not st.session_state.logged_in:

    st.info(
        "🔐 Login to view your prediction history."
    )

else:

    history = get_history(
        st.session_state.user_id
    )

    if not history:

        st.info(
            "No prediction history available yet."
        )

    else:

        for item in history:

            prediction = item.get(
                "prediction",
                "Unknown"
            )

            confidence = float(
                item.get("confidence", 0) or 0
            )

            message_text = item.get(
                "message",
                ""
            )

            language = item.get(
                "language",
                "Unknown"
            )

            created_at = item.get(
                "created_at",
                ""
            )

            if prediction == "SPAM / FRAUD":

                st.error(
                    f"🚨 **{prediction}**  |  "
                    f"Confidence: {confidence:.1f}%"
                )

            else:

                st.success(
                    f"✅ **{prediction}**  |  "
                    f"Confidence: {confidence:.1f}%"
                )

            st.write(
                f"**Message:** {message_text}"
            )

            st.caption(
                f"🌐 Language: {language}  •  🕒 {created_at}"
            )

            st.divider()


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
