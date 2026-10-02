import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import hstack as sp_hstack
import nltk

# Download NLTK resources
nltk.download("stopwords", quiet=True)

# -----------------------------
# Load trained model and files
# -----------------------------

with open("best_model.pkl", "rb") as f:
    best_clf = pickle.load(f)

with open("tfidf_word.pkl", "rb") as f:
    tfidf_word = pickle.load(f)

with open("tfidf_char.pkl", "rb") as f:
    tfidf_char = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# Preprocessing
# -----------------------------

ps = PorterStemmer()

english_stops = set(stopwords.words("english"))

hinglish_stops = {
    'hai', 'hain', 'ho', 'tha', 'thi', 'the', 'ka', 'ki', 'ke',
    'ko', 'se', 'me', 'mein', 'pe', 'par', 'aur', 'ya', 'bhi',
    'yeh', 'ye', 'woh', 'wo', 'ek', 'koi', 'kuch', 'sab', 'apna',
    'apni', 'apne', 'uska', 'uski', 'uske', 'mera', 'meri', 'mere',
    'tera', 'teri', 'tere', 'humara', 'tumhara', 'unka', 'kya', 'kyun',
    'kaise', 'kab', 'kahan', 'kaun', 'nahi', 'nahin', 'mat', 'na',
    'ji', 'bhai', 'yaar', 'dost', 'sir', 'madam',
}

all_stops = english_stops | hinglish_stops


def preprocess_multilingual(text):
    text = str(text).lower()

    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\b\d{10,}\b', ' phonenumber ', text)
    text = re.sub(r'₹|rs\.?|inr', ' rupees ', text)
    text = re.sub(r'£|\$|€', ' currency ', text)

    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
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


# -----------------------------
# Language Detection
# -----------------------------

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
        return "Unknown"


# -----------------------------
# Prediction
# -----------------------------

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
        result = "🚨 SPAM / FRAUD"
    else:
        result = "✅ LEGITIMATE (Ham)"

    confidence = (
        f"{max(probability) * 100:.1f}%"
        if probability is not None
        else "N/A"
    )

    return language, result, confidence


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Multilingual Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Multilingual Fraud Detection")
st.write(
    "Detect spam, phishing and fraudulent messages "
    "in English, Hindi and Hinglish."
)

message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You have won a lottery..."
)

if st.button("🔍 Detect Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        language, result, confidence = predict_message(message)

        st.subheader("Prediction")

        st.write(f"**Language:** {language}")
        st.write(f"**Result:** {result}")
        st.write(f"**Confidence:** {confidence}")