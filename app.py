import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load dataset from a relative path
file_path = "Winter Term Stats tracking - All years.csv"  # Ensure this file is in the same directory
df = pd.read_csv(file_path)

# Preprocess dataset
df['Conference'] = df['Conference'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
df['Home/Away'] = df['Home/Away'].astype(str).str.strip().str.lower().map({'home': 1, 'away': 0})
df['Result'] = df['Result'].astype(str).str.strip().str.lower()
df['Win_Indicator'] = df['Result'].map({'win': 1, 'loss': 0, 'draw': 0})

# Define features and target
features = ['Opponent Record', 'Oberlin Shots', 'Opponent Shots', 
            'Oberlin Fouls', 'Opponent Fouls', 
            'Passes Completed', 'Pass percentage', 'Conference', 'Home/Away']
target = 'Win_Indicator'

X = df[features]
y = df[target]

# Handle Missing Values
X = X.fillna(X.mean())

# Split dataset into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=50)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the Random Forest model
best_params = {
    'class_weight': 'balanced',
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 11,
    'n_estimators': 237
}
optimized_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    class_weight=best_params['class_weight'],
    random_state=50
)
optimized_rf.fit(X_train_smote, y_train_smote)

# Streamlit Interface
st.title("Soccer Match Outcome Predictor")
st.write("Input match statistics to predict if Oberlin will win.")

# Input Fields
Opponent_Record = st.slider("Opponent Record (win percentage)", 0.0, 1.0, step=0.01, value=0.5)
Oberlin_Shots = st.slider("Oberlin Shots", 0, 25, step=1, value=12)
Opponent_Shots = st.slider("Opponent Shots", 0, 25, step=1, value=10)
Oberlin_Fouls = st.slider("Oberlin Fouls", 0, 20, step=1, value=5)
Opponent_Fouls = st.slider("Opponent Fouls", 0, 20, step=1, value=7)
Passes_Completed = st.slider("Passes Completed", 0, 700, step=1, value=300)
Pass_Percentage = st.slider("Pass Percentage", 0, 100, step=1, value=75)
Conference = st.selectbox("Conference", ["Conference", "Non Conference"])
Home_Away = st.selectbox("Home/Away", ["Home", "Away"])

# Prediction Button
if st.button("Predict"):
    # Encode categorical inputs
    conference_encoded = 1 if Conference == "Conference" else 0
    home_away_encoded = 1 if Home_Away == "Home" else 0

    # Create input data for prediction
    input_data = pd.DataFrame([{
        'Opponent Record': Opponent_Record,
        'Oberlin Shots': Oberlin_Shots,
        'Opponent Shots': Opponent_Shots,
        'Oberlin Fouls': Oberlin_Fouls,
        'Opponent Fouls': Opponent_Fouls,
        'Passes Completed': Passes_Completed,
        'Pass percentage': Pass_Percentage,
        'Conference': conference_encoded,
        'Home/Away': home_away_encoded
    }])

    # Predict with the trained model
    prediction = optimized_rf.predict(input_data)[0]
    probability = optimized_rf.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 1:
        st.success(f"Prediction: Win ({probability * 100:.2f}% confidence)")
    else:
        st.error(f"Prediction: Loss/Draw ({(1 - probability) * 100:.2f}% confidence)")



# Feature Importance (Optional Visualization)
st.subheader("Feature Importance")
feature_importances = optimized_rf.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(features, feature_importances, color='skyblue')
ax.set_xlabel("Feature Importance Score")
ax.set_ylabel("Features")
ax.set_title("Feature Importance in Random Forest Model")
st.pyplot(fig)
