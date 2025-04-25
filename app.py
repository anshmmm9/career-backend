from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["https://career-frontend-azure.vercel.app"])

# Load model and encoders
model = load_model("career_model.h5")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")  # list of feature columns from training

@app.route('/')
def home():
    return "✅ Career Recommendation Backend is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_dict = {
            "Gender": data.get("Gender", ""),
            "UG_Course": data.get("UG_Course", ""),
            "UG_Specialization": data.get("UG_Specialization", ""),
            "Interests": data.get("Interests", ""),
            "Skills": data.get("Skills", ""),
            "CGPA": float(data.get("CGPA", 0)),
            "Has_Certification": data.get("Has_Certification", ""),
            "Certification_Title": data.get("Certification_Title", ""),
            "Is_Working": data.get("Is_Working", ""),
            "Job_Title": data.get("Job_Title", ""),
            "Has_Masters": data.get("Has_Masters", ""),
            "Masters_Field": data.get("Masters_Field", "")
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)

        missing_cols = set(feature_names) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0

        input_encoded = input_encoded[feature_names].copy()

        if 'CGPA' in input_encoded.columns:
            input_encoded[['CGPA']] = scaler.transform(input_encoded[['CGPA']])

        prediction = model.predict(input_encoded)
        predicted_index = np.argmax(prediction)
        predicted_career = label_encoder.inverse_transform([predicted_index])[0]

        return jsonify({'career': predicted_career})

    except Exception as e:
        return jsonify({'error': str(e)}
                       )
    

# Run app
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))  # Default to 10000 if PORT not set
    app.run(host="0.0.0.0", port=port)
