# train_model.py

import pandas as pd
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib # Used for saving the model and vectorizer

print("Starting model training...")

# The same all-in-one data from before
csv_data = """message,label
"Your account XX1234 has been debited for INR 550.00 on 26-08-2025. Your available balance is INR 12345.67.",financial
"Hey, are we still on for dinner tonight at 8?",not_financial
"OTP for your transaction of Rs. 2,500 at AMAZON is 987654. Do not share with anyone.",financial
"Your Amazon package with order ID 404-1234567-8901234 will be delivered today.",not_financial
"Your credit card bill of INR 8,950 is due on 05-09-2025. Please pay to avoid late fees.",financial
"FLAT 50% OFF on all items at MegaStore! Sale ends Sunday. Show this message to avail.",not_financial
"Congratulations! Your personal loan of Rs. 50,000 has been approved. Visit our app to proceed.",financial
"REMINDER: You have a dentist appointment tomorrow at 11 AM.",not_financial
"INR 4,500 has been credited to your account XX5678 via UPI from user@upi.",financial
"Your mobile number has been successfully recharged with the 2GB/day pack.",not_financial
"Reminder: Your EMI of Rs. 7,820 for loan AC/123 is due on 30-08-2025.",financial
"Happy Birthday! Hope you have a wonderful day filled with joy and laughter.",not_financial
"The NAV of your mutual fund 'BlueChip Growth' is now 154.23. Your portfolio value is up by 1.2%.",financial
"Team meeting has been rescheduled to 4 PM today in Conference Room B.",not_financial
"Salary of INR 65,000 has been credited to your account. Your new balance is INR 98,765.",financial
"Your Swiggy order is out for delivery. Your delivery executive is Rahul.",not_financial
"Use your SUPERBANK credit card to get 10% off on movie tickets this weekend! T&C apply.",financial
"Can you please send me the report by EOD? Thanks.",not_financial
"Payment of Rs. 350 to Zomato successful. Your UPI transaction ID is YBL123456789.",financial
"Flight UK820 from Delhi to Mumbai is delayed by 30 minutes. New departure time is 17:00.",not_financial
"ALERT: A transaction of INR 5000 was attempted on your card ending 9876. If this was not you, please call 1800-XXX-XXXX.",financial
"Your verification code for Facebook is FB-12345.",not_financial
"Thank you for paying your electricity bill of Rs. 1,240.",financial
"Don't forget to pick up milk on your way home.",not_financial
"Your stock order for RELIANCE has been executed. 10 shares @ Rs. 2850.",financial
"Your Uber is arriving now. Look for a white Swift Dzire, KA 01 AB 1234.",not_financial
"Cash withdrawal of INR 2000 from ATM detected on your debit card.",financial
"The school will remain closed tomorrow due to heavy rains. Please stay safe.",not_financial
"Interest of Rs. 345 has been credited to your Savings Account.",financial
"Your latest lab reports are ready. Please log in to our portal to view them.",not_financial
"You have received a refund of INR 499 from Flipkart.",financial
"Please approve the login attempt from a new device for your Google Account.",not_financial
"Rs.100 cashback has been credited to your wallet.",financial
"Your subscription to 'Cool Magazine' is expiring soon. Renew now!",not_financial
"Your auto-debit for Netflix subscription of Rs. 649 has been processed.",financial
"Did you see the latest episode last night? It was amazing!",not_financial
"Your insurance premium of INR 12,000 is due. Pay now to keep your policy active.",financial
"URGENT: Your electricity will be disconnected due to a pending update. Click here to verify: [suspicious link]",not_financial
"Transfer of INR 3,000 to John Doe was successful. Ref ID: AX1234.",financial
"Your table for 2 at 'The Great Eatery' is confirmed for 8:30 PM tonight.",not_financial
"You have spent Rs. 5000 on your credit card today. Current outstanding is Rs. 15000.",financial
"Your OTP for login to HealthApp is 554433. Valid for 10 mins.",not_financial
"Your Demat account has been successfully opened.",financial
"Hi! Just wanted to check in. How are you doing?",not_financial
"Your request for a new cheque book has been processed and will be delivered shortly.",financial
"You have used 80% of your daily high-speed data. Recharge now to continue browsing.",not_financial
"A charge of $50 has been made on your international credit card.",financial
"New message from Priya on WhatsApp.",not_financial
"Your Fixed Deposit of Rs. 1,00,000 will mature on 15-09-2025.",financial
"Missed call alert: You have a missed call from +919876543210.",not_financial
"""

df = pd.read_csv(io.StringIO(csv_data))

# Preprocess the data
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label_encoded']

# Train the final model on ALL data
model = MultinomialNB()
model.fit(X, y)
print("Model training complete.")

# --- CRUCIAL STEP: SAVE THE OBJECTS ---
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("Model, vectorizer, and label encoder saved to disk.")