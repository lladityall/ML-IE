
# Sample script to create a mock pickle file for testing
# This should be replaced with the actual trained model from your Colab notebook

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Create sample data for training (this would be your actual dataset)
np.random.seed(42)
n_samples = 1000

# Generate sample data
sample_data = {
    'age': np.random.randint(18, 25, n_samples),
    'previous_gpa': np.random.uniform(2.0, 4.0, n_samples),
    'scholarship': np.random.choice([0, 1], n_samples),
    'weekly_study_hours': np.random.randint(5, 30, n_samples),
    'lms_logins_per_week': np.random.randint(1, 15, n_samples),
    'video_watched_pct': np.random.uniform(20, 100, n_samples),
    'forum_posts': np.random.randint(0, 10, n_samples),
    'assignment_ontime_pct': np.random.uniform(50, 100, n_samples),
    'quiz_avg': np.random.uniform(40, 100, n_samples),
    'midterm_grade': np.random.uniform(40, 100, n_samples),
    'major': np.random.choice(['Biology', 'CompSci', 'English', 'History', 'Physics'], n_samples)
}

# Create target variable
advising_actions = [
    'Check_In_Engagement', 'Mandatory_Advising_Meeting', 'Monitor_Progress', 
    'No_Action_Needed', 'Recommend_Tutoring', 'Wellness_Referral'
]
sample_data['advising_action'] = np.random.choice(advising_actions, n_samples)

df = pd.DataFrame(sample_data)

# Encode categorical features
le_major = LabelEncoder()
le_action = LabelEncoder()

df['major_encoded'] = le_major.fit_transform(df['major'])
df['advising_action_encoded'] = le_action.fit_transform(df['advising_action'])

# Features for training
features = [
    'age', 'previous_gpa', 'scholarship', 'weekly_study_hours',
    'lms_logins_per_week', 'video_watched_pct', 'forum_posts',
    'assignment_ontime_pct', 'quiz_avg', 'midterm_grade', 'major_encoded'
]

X = df[features]
y = df['advising_action_encoded']

# Train model
model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
model.fit(X, y)

# Create model bundle
model_bundle = {
    "model": model,
    "label_encoder_action": le_action,
    "label_encoder_major": le_major,
    "feature_columns": features
}

# Save to pickle file
with open("advising_action_predictor.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("✅ Sample model created and saved as advising_action_predictor.pkl")
print("📋 Available majors:", le_major.classes_)
print("📋 Available actions:", le_action.classes_)
