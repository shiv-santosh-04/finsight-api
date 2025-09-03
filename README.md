# 🧠 Finsight API

A Flask-based REST API that classifies SMS messages as **financial** or **non-financial** using a trained Machine Learning model.

## 🚀 Project Description

This API takes an SMS message as input and uses:
- A vectorizer to preprocess the text.
- A trained ML model to classify the message.
- A label encoder to decode the prediction.

It is built to serve as the backend for the Finsight app — a smart AI-powered finance tracker.

---

## 🛠️ Tech Stack

- Python
- Flask
- scikit-learn
- joblib
- pandas

---

## 🔍 API Endpoints

### `POST /predict`

**Request Body:**
```json
{
  "message": "INR 5000 debited from your HDFC Bank account"
}
{
  "prediction": "financial"
}
GET /
Returns a simple status message to confirm the API is running.
"ML Model API is running!"
## 🧪 How to Run Locally
Clone the repo:
git clone [https://github.com/shiv-santosh-04/finsight-api.git](https://github.com/shiv-santosh-04/finsight-api.git)
cd finsight-api

Install dependencies:
pip install -r requirements.txt

Run the API:
python app.py

The API will run locally at: http://127.0.0.1:5000/

🐍 Requirements
Make sure you have Python installed.

Dependencies are listed in requirements.txt:

Flask

joblib

scikit-learn

pandas

🌐 Deployment
This API can be hosted on platforms like Render or Railway. Make sure to point your frontend (e.g., Flutter or a Web app) to the correct hosted URL.

👤 Author
 Shiv Santosh
 GitHub