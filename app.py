import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Insurance Fraud Detection")

# Define input fields
def user_input():
    # Collect user inputs for a subset of features and set defaults for the rest
    months_as_customer = st.number_input("Months as Customer", min_value=0, max_value=600)
    age = st.number_input("Age", min_value=18, max_value=100)
    policy_state = st.selectbox("Policy State", ["OH", "IN", "IL"])
    policy_csl = st.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
    policy_deductable = st.number_input("Policy Deductible", min_value=500, max_value=2000, step=100)
    policy_annual_premium = st.number_input("Policy Annual Premium", min_value=500.0, max_value=2000.0, step=50.0)
    umbrella_limit = st.number_input("Umbrella Limit", min_value=0, max_value=10000000, step=100000)
    
    # Add more inputs as per your dataset...
    
    # Placeholder/default values for other columns
    data = {
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_state': policy_state,
        'policy_csl': policy_csl,
        'policy_deductable': policy_deductable,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'insured_sex': "MALE",  # Example default
        'insured_education_level': "Associate",  # Example default
        'insured_occupation': "sales",  # Example default
        'insured_hobbies': "board-games",  # Example default
        'insured_relationship': "husband",  # Example default
        'capital-gains': 0,  # Example default
        'capital-loss': 0,  # Example default
        'incident_type': "Single Vehicle Collision",  # Example default
        'collision_type': "Side Collision",  # Example default
        'incident_severity': "Major Damage",  # Example default
        'authorities_contacted': "Police",  # Example default
        'incident_state': "OH",  # Example default
        'incident_city': "Columbus",  # Example default
        'incident_hour_of_the_day': 12,  # Example default
        'number_of_vehicles_involved': 1,  # Example default
        'property_damage': "YES",  # Example default
        'bodily_injuries': 1,  # Example default
        'witnesses': 1,  # Example default
        'police_report_available': "YES",  # Example default
        'total_claim_amount': 10000,  # Example default
        'injury_claim': 5000,  # Example default
        'property_claim': 3000,  # Example default
        'vehicle_claim': 2000,  # Example default
        'auto_make': "Toyota",  # Example default
        'auto_model': "Camry",  # Example default
        'auto_year': 2015,  # Example default
    }
    
    return pd.DataFrame([data])

# User input
input_df = user_input()

# Prediction
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)
    result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
    st.write(f"Prediction: {result}")
