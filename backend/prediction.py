import os
import re
import pickle
from scipy.sparse import hstack
from nltk.stem import PorterStemmer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tfidf_word = pickle.load(
    open(os.path.join(BASE_DIR, "tfidf_word.pkl"), "rb")
)

tfidf_char = pickle.load(
    open(os.path.join(BASE_DIR, "tfidf_char.pkl"), "rb")
)

best_clf = pickle.load(
    open(os.path.join(BASE_DIR, "best_model.pkl"), "rb")
)

encoder = pickle.load(
    open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb")
)

ps = PorterStemmer()

hinglish_stops = {
    'hai', 'hain', 'ho', 'tha', 'thi', 'the', 'ka', 'ki', 'ke',
    'ko', 'se', 'me', 'mein', 'pe', 'par', 'aur', 'ya', 'bhi',
    'yeh', 'ye', 'woh', 'wo', 'ek', 'koi', 'kuch', 'sab', 'apna',
    'apni', 'apne', 'uska', 'uski', 'uske', 'mera', 'meri', 'mere',
    'tera', 'teri', 'tere', 'humara', 'tumhara', 'unka', 'kya', 'kyun',
    'kaise', 'kab', 'kahan', 'kaun', 'nahi', 'nahin', 'mat', 'na',
    'ji', 'bhai', 'yaar', 'dost', 'sir', 'madam'
}


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
        if tok in hinglish_stops or len(tok) < 2:
            continue

        if re.match(r'^[a-z]+$', tok):
            tok = ps.stem(tok)

        cleaned.append(tok)

    return ' '.join(cleaned)


def predict_message(message):

    processed = preprocess_multilingual(message)

    X_word = tfidf_word.transform([processed])
    X_char = tfidf_char.transform([processed])

    X = hstack([X_word, X_char])

    prediction = best_clf.predict(X)[0]

    if hasattr(best_clf, "predict_proba"):
        probabilities = best_clf.predict_proba(X)[0]
        confidence = float(max(probabilities))
    else:
        confidence = 0.0

    if prediction == 1:
        result = "SPAM / FRAUD"
    else:
        result = "LEGITIMATE"

    return {
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    }