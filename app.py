# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data
def load_model_components():
    """Loads the trained model, scaler, and column list from pickle files."""
    try:
        model = pickle.load(open('heart_disease_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        model_columns = pickle.load(open('model_columns.pkl', 'rb'))
        return model, scaler, model_columns
    except FileNotFoundError as e:
        st.error(f"Error: Make sure 'heart_disease_model.pkl', 'scaler.pkl', and 'model_columns.pkl' are in the same directory as the app. Missing file: {e.filename}")
        st.stop()

def preprocess_input(input_data, model_columns):
    """
    Transforms user input into a DataFrame that matches the model's training data format.
    This includes one-hot encoding and aligning columns.
    """
    # Create a DataFrame from the user's input dictionary
    input_df = pd.DataFrame([input_data])
    
    # Perform one-hot encoding, just like during training
    input_df_encoded = pd.get_dummies(input_df)
    
    # Align the DataFrame with the model's expected columns.
    # Any missing columns (from one-hot encoding) are filled with 0.
    # Any extra columns are dropped.
    final_input_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    return final_input_df

# --- Load Model Components ---
# This is done once when the app starts
model, scaler, model_columns = load_model_components()

# --- Main Application UI ---
def main():
    st.title("❤️ Heart Disease Prediction App")
    st.write("Enter your health details below to get a risk assessment for heart disease.")
    st.markdown("---")

    # --- Input Form ---
    with st.form("prediction_form"):
        st.header("Your Health Information")
        
        # --- Create columns for better layout ---
        col1, col2, col3 = st.columns(3)

        with col1:
            bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0, 0.1)
            smoking = st.radio("Smoked at least 100 cigarettes?", ('No', 'Yes'))
            alcohol_drinking = st.radio("Heavy alcohol drinker?", ('No', 'Yes'))
            stroke = st.radio("Ever had a stroke?", ('No', 'Yes'))
            physical_health = st.slider("For how many days was your physical health not good? (past 30 days)", 0, 30, 0)
            mental_health = st.slider("For how many days was your mental health not good? (past 30 days)", 0, 30, 0)

        with col2:
            diff_walking = st.radio("Serious difficulty walking or climbing stairs?", ('No', 'Yes'))
            sex = st.selectbox('Sex', ('Female', 'Male'))
            age_category = st.selectbox('Age Category', 
                ('18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'))
            race = st.selectbox('Race', 
                ('White', 'Black', 'American Indian/Alaskan Native', 'Asian', 'Hispanic', 'Other'))
            diabetic = st.selectbox('Diabetic Status', 
                ('No', 'No, borderline diabetes', 'Yes', 'Yes, during pregnancy'))
        
        with col3:
            physical_activity = st.radio("Physical activity in past 30 days?", ('No', 'Yes'))
            gen_health = st.selectbox('General Health', 
                ('Excellent', 'Very good', 'Good', 'Fair', 'Poor'))
            sleep_time = st.slider("Average hours of sleep per day", 1, 24, 7)
            asthma = st.radio("Ever had asthma?", ('No', 'Yes'))
            kidney_disease = st.radio("Ever had kidney disease?", ('No', 'Yes'))
            skin_cancer = st.radio("Ever had skin cancer?", ('No', 'Yes'))

        # --- Prediction Button ---
        submitted = st.form_submit_button("Predict Heart Disease Risk")

    # --- Prediction Logic ---
    if submitted:
        # Map user inputs from the form to the format expected by the model
        input_dict = {
            'BMI': bmi,
            'Smoking': 1 if smoking == 'Yes' else 0,
            'AlcoholDrinking': 1 if alcohol_drinking == 'Yes' else 0,
            'Stroke': 1 if stroke == 'Yes' else 0,
            'PhysicalHealth': physical_health,
            'MentalHealth': mental_health,
            'DiffWalking': 1 if diff_walking == 'Yes' else 0,
            'Sex': sex,
            'AgeCategory': age_category,
            'Race': race,
            'Diabetic': diabetic,
            'PhysicalActivity': 1 if physical_activity == 'Yes' else 0,
            'GenHealth': gen_health,
            'SleepTime': sleep_time,
            'Asthma': 1 if asthma == 'Yes' else 0,
            'KidneyDisease': 1 if kidney_disease == 'Yes' else 0,
            'SkinCancer': 1 if skin_cancer == 'Yes' else 0
        }

        # 1. Preprocess the input to match the training data structure
        processed_input_df = preprocess_input(input_dict, model_columns)
        
        # 2. Scale the input using the loaded scaler
        scaled_input_array = scaler.transform(processed_input_df)

        # 3. Convert the scaled array back to a DataFrame with feature names
        # This is the crucial step to avoid the UserWarning and ensure correct predictions
        scaled_input_df = pd.DataFrame(scaled_input_array, columns=model_columns)

        # 4. Make prediction using the final, correctly formatted DataFrame
        prediction = model.predict(scaled_input_df)
        prediction_proba = model.predict_proba(scaled_input_df)

        # Display Result
        st.markdown("---")
        st.header("Prediction Result")
        
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
        
        st.write(f"Model Confidence: {prediction_proba[0][prediction[0]]:.2%}")
        st.info("Disclaimer: This app is for informational purposes only and is not a substitute for professional medical advice.")

    # --- About Section ---
    st.markdown("---")
    st.header("About This Model")
    st.write("""
    This application uses a Logistic Regression model trained on the CDC's 2020 BRFSS dataset.
    The model was optimized to have a high **Recall** score, meaning it is designed to be 
    very sensitive and correctly identify as many actual cases of heart disease as possible, 
    minimizing the number of missed diagnoses.
    """)
    
    # Display Feature Importance
    st.subheader("Key Predictive Factors")
    try:
        st.image("images/05_feature_importance.png", caption="Top 15 most influential features according to the model.")
    except FileNotFoundError:
        st.warning("Feature importance image not found. Make sure the 'images' folder is in the same directory.")


if __name__ == '__main__':
    main()