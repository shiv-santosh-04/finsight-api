# 🧠 Finsight API

A Flask-based REST API that classifies SMS messages as *financial* or *non-financial* using a trained Machine Learning model.

## 🚀 Project Description

This API takes an SMS message as input and uses:
- A vectorizer to preprocess the text
- A trained ML model to classify the message
- A label encoder to decode the prediction

It is built to serve as the backend for the Finsight app — a smart AI-powered finance tracker.

---

## 🛠 Tech Stack

- Python
- Flask
- scikit-learn
- joblib
- pandas

---

## 🔍 API Endpoint

### POST /predict

*Request Body:*
```json
{
  "message": "INR 5000 debited from your HDFC Bank account"
}