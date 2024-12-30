import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Function to set background image and text color
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: white;
    }}
    .stApp .stMarkdown p {{
        color: white;
    }}
    .stButton button {{
        background-color: #0078D7;
        color: white;
        border-radius: 5px;
        border: none;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the background image
set_background("img.jpg")

# Load the trained model
model = joblib.load("final_xgb_classifier.pkl")

# Function to preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Convert categorical variables to numeric
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    
    # Return preprocessed DataFrame
    return df

# Streamlit UI
st.title("Customer Churn Prediction")

# Sidebar for user inputs
st.sidebar.header("📝Customer Information")

# Gender feature
st.sidebar.subheader("Gender")
gender = st.sidebar.radio("Select Gender", ["Female", "Male"])

# Senior Citizen feature
st.sidebar.subheader("Senior Citizen")
senior_citizen = st.sidebar.radio("Is the customer a Senior Citizen?", ["No", "Yes"])

# Partner feature
st.sidebar.subheader("Partner")
partner = st.sidebar.radio("Does the customer have a partner?", ["No", "Yes"])

# Dependents feature
st.sidebar.subheader("Dependents")
dependents = st.sidebar.radio("Does the customer have dependents?", ["No", "Yes"])

# Phone Service feature
st.sidebar.subheader("Phone Service")
phone_service = st.sidebar.radio("Does the customer have phone service?", ["No", "Yes"])

# Multiple Lines feature
st.sidebar.subheader("Multiple Lines")
multiple_lines = st.sidebar.radio("Does the customer have multiple lines?", ["No", "Yes", "No phone service"])

# Internet Service feature
st.sidebar.subheader("Internet Service")
internet_service = st.sidebar.selectbox("What type of internet service does the customer have?", ['DSL', 'Fiber optic', 'No'])

# Online Security feature
st.sidebar.subheader("Online Security")
online_security = st.sidebar.radio("Does the customer have online security?", ["No", "Yes", "No internet service"])

# Online Backup feature
st.sidebar.subheader("Online Backup")
online_backup = st.sidebar.radio("Does the customer have online backup?", ["No", "Yes", "No internet service"])

# Device Protection feature
st.sidebar.subheader("Device Protection")
device_protection = st.sidebar.radio("Does the customer have device protection?", ["No", "Yes", "No internet service"])

# Tech Support feature
st.sidebar.subheader("Tech Support")
tech_support = st.sidebar.radio("Does the customer have tech support?", ["No", "Yes", "No internet service"])

# Streaming TV feature
st.sidebar.subheader("Streaming TV")
streaming_tv = st.sidebar.radio("Does the customer have streaming TV?", ["No", "Yes", "No internet service"])

# Streaming Movies feature
st.sidebar.subheader("Streaming Movies")
streaming_movies = st.sidebar.radio("Does the customer have streaming movies?", ["No", "Yes", "No internet service"])

# Contract feature
st.sidebar.subheader("Contract")
contract = st.sidebar.selectbox("What type of contract does the customer have?", ['Month-to-month', 'One year', 'Two year'])

# Paperless Billing feature
st.sidebar.subheader("Paperless Billing")
paperless_billing = st.sidebar.radio("Does the customer have paperless billing?", ["No", "Yes"])

# Payment Method feature
st.sidebar.subheader("Payment Method")
payment_method = st.sidebar.selectbox("What is the customer's payment method?", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Monthly Charges feature
st.sidebar.subheader("Monthly Charges")
monthly_charges = st.sidebar.number_input("Enter the customer's monthly charges", value=0.0)

# Total Charges feature
st.sidebar.subheader("Total Charges")
total_charges = st.sidebar.number_input("Enter the customer's total charges", value=0.0)

# Tenure Group feature
st.sidebar.subheader("Tenure Group")
tenure_group = st.sidebar.number_input("Enter the customer's tenure group", value=0)

# Main section for the prediction
st.markdown("📊Prediction Result")
if st.button("📈Predict"):
    # Create dictionary from user inputs
    user_data = {
        'gender': 0 if gender == "Female" else 1,  # Female: 0, Male: 1
        'SeniorCitizen': 0 if senior_citizen == "No" else 1,
        'Partner': 0 if partner == "No" else 1,
        'Dependents': 0 if dependents == "No" else 1,
        'PhoneService': 0 if phone_service == "No" else 1,
        'MultipleLines': 0 if multiple_lines == "No" else 1,  # "No phone service" will be mapped as 0
        'InternetService': internet_service,
        'OnlineSecurity': 0 if online_security == "No" else 1,
        'OnlineBackup': 0 if online_backup == "No" else 1,
        'DeviceProtection': 0 if device_protection == "No" else 1,
        'TechSupport': 0 if tech_support == "No" else 1,
        'StreamingTV': 0 if streaming_tv == "No" else 1,
        'StreamingMovies': 0 if streaming_movies == "No" else 1,
        'Contract': contract,
        'PaperlessBilling': 0 if paperless_billing == "No" else 1,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }
    
    # Preprocess input data
    processed_data = preprocess_input(user_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("⚠️The customer is likely to leave.")
    else:
        st.write("✅ The customer is likely to stay.")
