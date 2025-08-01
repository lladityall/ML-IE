
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the trained model and encoders
try:
    with open("advising_action_predictor.pkl", "rb") as f:
        model_bundle = pickle.load(f)

    model = model_bundle["model"]
    le_action = model_bundle["label_encoder_action"]
    le_major = model_bundle["label_encoder_major"]
    feature_columns = model_bundle["feature_columns"]
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model file not found. Please ensure advising_action_predictor.pkl is in the same directory.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get form data
        data = request.get_json() if request.is_json else request.form

        # Extract input features
        student_data = {
            'student_id': int(data['student_id']),
            'age': int(data['age']),
            'major': data['major'],
            'previous_gpa': float(data['previous_gpa']),
            'scholarship': data['scholarship'].lower() == 'true',
            'weekly_study_hours': int(data['weekly_study_hours']),
            'lms_logins_per_week': int(data['lms_logins_per_week']),
            'video_watched_pct': float(data['video_watched_pct']),
            'forum_posts': int(data['forum_posts']),
            'assignment_ontime_pct': float(data['assignment_ontime_pct']),
            'quiz_avg': float(data['quiz_avg']),
            'midterm_grade': float(data['midterm_grade'])
        }

        # Create DataFrame
        df = pd.DataFrame([student_data])

        # Encode categorical features
        df['major_encoded'] = le_major.transform(df['major'])
        df['scholarship'] = df['scholarship'].astype(int)

        # Select features for prediction
        X = df[feature_columns]

        # Make prediction
        predicted_action = model.predict(X)[0]
        predicted_label = le_action.inverse_transform([predicted_action])[0]

        # Get prediction probability
        prediction_proba = model.predict_proba(X)[0]
        confidence = np.max(prediction_proba) * 100

        # Prepare result
        result = {
            'student_id': student_data['student_id'],
            'predicted_action': predicted_label,
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/majors')
def get_majors():
    """Return available majors for the dropdown"""
    try:
        majors = le_major.classes_.tolist()
        return jsonify(majors)
    except:
        return jsonify(['Biology', 'CompSci', 'English', 'History', 'Physics'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
