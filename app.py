import streamlit as st
import pickle
import re
import random
from pathlib import Path
from scipy.sparse import hstack
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multilingual Phishing & Spam Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


# ============================================================
# SESSION STATE
# ============================================================

if "sample_message" not in st.session_state:
    st.session_state.sample_message = ""


# ============================================================
# LOAD MODELS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

try:

    with open(
        BASE_DIR / "tfidf_word.pkl",
        "rb"
    ) as f:
        tfidf_word = pickle.load(f)

    with open(
        BASE_DIR / "tfidf_char.pkl",
        "rb"
    ) as f:
        tfidf_char = pickle.load(f)

    with open(
        BASE_DIR / "best_model.pkl",
        "rb"
    ) as f:
        best_clf = pickle.load(f)

    with open(
        BASE_DIR / "label_encoder.pkl",
        "rb"
    ) as f:
        encoder = pickle.load(f)

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

    hindi_chars = len(
        re.findall(
            r"[\u0900-\u097F]",
            text
        )
    )

    english_chars = len(
        re.findall(
            r"[A-Za-z]",
            text
        )
    )

    if hindi_chars > 0 and english_chars > 0:
        return "Hinglish"

    elif hindi_chars > 0:
        return "Hindi"

    elif english_chars > 0:
        return "English"

    return "Unknown"


# ============================================================
# PREDICTION
# ============================================================

def predict_message(message):

    language = detect_language(message)

    processed = preprocess_multilingual(
        message
    )

    X_w = tfidf_word.transform(
        [processed]
    )

    X_c = tfidf_char.transform(
        [processed]
    )

    X = hstack(
        [X_w, X_c]
    )

    predicted_label = best_clf.predict(X)[0]

    try:

        decoded_label = encoder.inverse_transform(
            [predicted_label]
        )[0]

        label_text = str(
            decoded_label
        ).lower()

        if (
            "spam" in label_text
            or "fraud" in label_text
            or "phishing" in label_text
        ):

            result = "SPAM / FRAUD"

        else:

            result = "LEGITIMATE"

    except Exception:

        if predicted_label == 1:
            result = "SPAM / FRAUD"
        else:
            result = "LEGITIMATE"

    if hasattr(
        best_clf,
        "predict_proba"
    ):

        probability = best_clf.predict_proba(X)[0]

        confidence = (
            max(probability) * 100
        )

    else:

        confidence = 0.0

    return (
        language,
        result,
        confidence
    )


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

    .main .block-container {

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
       Hide Star, Edit and GitHub.
       ====================================================== */

    /* Hide Star / Favorite */
    [data-testid="stToolbar"] button[aria-label*="Star"],
    [data-testid="stToolbar"] button[title*="Star"],
    [data-testid="stToolbar"] [aria-label*="star"],
    [data-testid="stToolbar"] [title*="star"],
    [data-testid="stToolbar"] a[aria-label*="Star"],
    [data-testid="stToolbar"] a[title*="Star"] {
        display: none !important;
    }

    /* Hide Edit / pencil */
    [data-testid="stToolbar"] button[aria-label*="Edit"],
    [data-testid="stToolbar"] button[title*="Edit"],
    [data-testid="stToolbar"] [aria-label*="edit"],
    [data-testid="stToolbar"] [title*="edit"],
    [data-testid="stToolbar"] a[aria-label*="Edit"],
    [data-testid="stToolbar"] a[title*="Edit"] {
        display: none !important;
    }

    /* Hide GitHub */
    #GithubIcon,
    [data-testid="stToolbar"] a[href*="github"],
    [data-testid="stToolbar"] a[aria-label*="GitHub"],
    [data-testid="stToolbar"] a[title*="GitHub"],
    [data-testid="stToolbar"] [aria-label*="github"],
    [data-testid="stToolbar"] [title*="github"] {
        display: none !important;
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
        "##  ShieldAI"
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
# HEADER
# ============================================================

st.markdown(
    "#  Multilingual Phishing & Spam Detection"
)

st.caption(
    "AI-powered detection of suspicious messages "
    "across English, Hindi and Hinglish."
)


# ============================================================
# MESSAGE ANALYZER
# ============================================================

st.subheader(
    " Message Analyzer"
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
# SAMPLE BUTTONS
# ============================================================

sample1, sample2, sample3 = st.columns(3)


with sample1:

    if st.button(
        " Spam Example",
        use_container_width=True
    ):

        st.session_state.sample_message = random.choice(
            spam_examples
        )

        st.rerun()


with sample2:

    if st.button(
        " Hindi Example",
        use_container_width=True
    ):

        st.session_state.sample_message = random.choice(
            hindi_examples
        )

        st.rerun()


with sample3:

    if st.button(
        " Hinglish Example",
        use_container_width=True
    ):

        st.session_state.sample_message = random.choice(
            hinglish_examples
        )

        st.rerun()


# ============================================================
# MESSAGE INPUT
# ============================================================

message = st.text_area(
    "Message",
    value=st.session_state.sample_message,
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
    " Analyze Message",
    use_container_width=True,
    type="primary"
)


# ============================================================
# RESULT
# ============================================================

if analyze:

    if not message.strip():

        st.warning(
            " Please enter a message before analyzing."
        )

    else:

        language, result, confidence = predict_message(
            message
        )

        st.markdown("---")

        st.subheader(
            " Prediction Result"
        )

        if result == "SPAM / FRAUD":

            st.error(
                " SPAM / FRAUD DETECTED"
            )

            st.markdown(
                "**Warning:** The model detected patterns "
                "associated with suspicious or fraudulent messages."
            )

        else:

            st.success(
                " LEGITIMATE MESSAGE"
            )

            st.markdown(
                "**Safe classification:** The model classified "
                "this message as legitimate (Ham)."
            )

        st.write("")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                " Detected Language",
                language
            )

        with result_col2:

            st.metric(
                " Confidence",
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
    " Detection Pipeline"
)

st.caption(
    "The system combines multilingual preprocessing "
    "with machine-learning based classification."
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.info(
        " **Multilingual**\n\n"
        "English, Hindi and Hinglish message detection."
    )


with col2:

    st.info(
        " **NLP Processing**\n\n"
        "Text cleaning, normalization, stopword removal "
        "and stemming."
    )


with col3:

    st.info(
        " **TF-IDF Features**\n\n"
        "Word-level and character-level text features."
    )


with col4:

    st.info(
        " **Linear SVM**\n\n"
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
# FOOTER
# ============================================================

st.divider()

st.caption(
    " Multilingual Phishing & Spam Detection"
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
