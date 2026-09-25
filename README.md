# Multilingual Phishing Detection Using Machine Learning

## 📌 Project Overview

This project presents a machine-learning-based system for detecting **spam, phishing, fraudulent, and unwanted messages** in multilingual and code-mixed environments.

The system focuses on three types of text:

- English
- Hindi
- Hinglish (Hindi + English code-mixed text)

The main goal is to address the limitations of traditional spam and phishing detection systems that primarily focus on English-language messages.

---

## 🎯 Objectives

The main objectives of this project are:

- Detect spam, phishing, and fraudulent messages.
- Support English, Hindi, and Hinglish messages.
- Apply Natural Language Processing techniques for text preprocessing.
- Extract textual features using **TF-IDF**.
- Train and compare multiple machine learning models.
- Explore **Multilingual BERT (mBERT)** for multilingual contextual understanding.
- Evaluate the models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.

---

## 📊 Dataset

The project uses three datasets:

| Language | Dataset | Number of Samples |
|----------|---------|------------------:|
| English | SMS Spam Collection Dataset | 5,572 |
| Hindi | Custom Dataset | 5,000 |
| Hinglish | Custom Dataset | 1,102 |

### English Dataset

The English dataset is the **SMS Spam Collection Dataset**, obtained from UCI/Kaggle.

It contains messages classified into:

- Spam
- Ham (legitimate)

### Hindi Dataset

A custom Hindi dataset containing legitimate and fraudulent messages was created for this project.

It includes examples related to:

- Banking scams
- Phishing attempts
- Fake lottery messages
- Loan offers
- Fraudulent advertisements

### Hinglish Dataset

A custom Hinglish dataset was created for code-mixed communication.

Hinglish contains a combination of Hindi and English, commonly written using Roman script.

The dataset contains:

- Genuine messages
- Phishing messages
- Financial scams
- Promotional messages
- Advertisements

---

## 🔄 System Workflow

```text
Dataset Collection
        ↓
Data Preprocessing
        ↓
Tokenization
        ↓
Stopword Removal
        ↓
Stemming
        ↓
TF-IDF Feature Extraction
        ↓
Machine Learning Models
        ↓
Multilingual Processing using mBERT
        ↓
Message Classification
        ↓
Performance Evaluation
