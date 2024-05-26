import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the model for prediction
model = tf.keras.models.load_model('customer_churn_ANNmodel.h5')

# Streamlit app
st.title('Customer Churn Prediction App ANN')
st.sidebar.header('User Input Parameters')

# User input for prediction
def user_input_features():
    input_features = {
        'gender': 1 if st.sidebar.selectbox('Gender', ['Male', 'Female'], key='gender') == 'Female' else 0,
        'SeniorCitizen': 1 if st.sidebar.selectbox('Senior Citizen', ['Yes', 'No'], key='senior_citizen') == 'Yes' else 0,
        'Partner': 1 if st.sidebar.selectbox('Partner', ['Yes', 'No'], key='partner') == 'Yes' else 0,
        'Dependents': 1 if st.sidebar.selectbox('Dependents', ['Yes', 'No'], key='dependents') == 'Yes' else 0,
        'PhoneService': 1 if st.sidebar.selectbox('Phone Service', ['Yes', 'No'], key='phone_service') == 'Yes' else 0,
        'MultipleLines': 1 if st.sidebar.selectbox('Multiple Lines', ['Yes', 'No'], key='multiple_lines') == 'Yes' else 0,
        'OnlineSecurity': 1 if st.sidebar.selectbox('Online Security', ['Yes', 'No'], key='online_security') == 'Yes' else 0,
        'OnlineBackup': 1 if st.sidebar.selectbox('Online Backup', ['Yes', 'No'], key='online_backup') == 'Yes' else 0,
        'DeviceProtection': 1 if st.sidebar.selectbox('Device Protection', ['Yes', 'No'], key='device_protection') == 'Yes' else 0,
        'TechSupport': 1 if st.sidebar.selectbox('Tech Support', ['Yes', 'No'], key='tech_support') == 'Yes' else 0,
        'StreamingTV': 1 if st.sidebar.selectbox('Streaming TV', ['Yes', 'No'], key='streaming_tv') == 'Yes' else 0,
        'StreamingMovies': 1 if st.sidebar.selectbox('Streaming Movies', ['Yes', 'No'], key='streaming_movies') == 'Yes' else 0,
        'PaperlessBilling': 1 if st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'], key='paperless_billing') == 'Yes' else 0,
        'tenure': st.sidebar.number_input('Tenure', min_value=0, key='tenure'),
        'MonthlyCharges': st.sidebar.number_input('Monthly Charges', min_value=0.0, key='monthly_charges'),
        'TotalCharges': st.sidebar.number_input('Total Charges', min_value=0.0, key='total_charges'),
    }

    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'], key='contract')
    if contract == 'Month-to-month':
        input_features['Contract_Month-to-month'] = 1
        input_features['Contract_One year'] = 0
        input_features['Contract_Two year'] = 0
    elif contract == 'One year':
        input_features['Contract_Month-to-month'] = 0
        input_features['Contract_One year'] = 1
        input_features['Contract_Two year'] = 0
    elif contract == 'Two year':
        input_features['Contract_Month-to-month'] = 0
        input_features['Contract_One year'] = 0
        input_features['Contract_Two year'] = 1

    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], key='payment_method')
    if payment_method == 'Electronic check':
        input_features['PaymentMethod_Electronic check'] = 1
        input_features['PaymentMethod_Mailed check'] = 0
        input_features['PaymentMethod_Bank transfer (automatic)'] = 0
        input_features['PaymentMethod_Credit card (automatic)'] = 0
    elif payment_method == 'Mailed check':
        input_features['PaymentMethod_Electronic check'] = 0
        input_features['PaymentMethod_Mailed check'] = 1
        input_features['PaymentMethod_Bank transfer (automatic)'] = 0
        input_features['PaymentMethod_Credit card (automatic)'] = 0
    elif payment_method == 'Bank transfer (automatic)':
        input_features['PaymentMethod_Electronic check'] = 0
        input_features['PaymentMethod_Mailed check'] = 0
        input_features['PaymentMethod_Bank transfer (automatic)'] = 1
        input_features['PaymentMethod_Credit card (automatic)'] = 0
    elif payment_method == 'Credit card (automatic)':
        input_features['PaymentMethod_Electronic check'] = 0
        input_features['PaymentMethod_Mailed check'] = 0
        input_features['PaymentMethod_Bank transfer (automatic)'] = 0
        input_features['PaymentMethod_Credit card (automatic)'] = 1

    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'], key='internet_service')
    if internet_service == 'DSL':
        input_features['InternetService_DSL'] = 1
        input_features['InternetService_Fiber optic'] = 0
        input_features['InternetService_No'] = 0
    elif internet_service == 'Fiber optic':
        input_features['InternetService_DSL'] = 0
        input_features['InternetService_Fiber optic'] = 1
        input_features['InternetService_No'] = 0
    elif internet_service == 'No':
        input_features['InternetService_DSL'] = 0
        input_features['InternetService_Fiber optic'] = 0
        input_features['InternetService_No'] = 1

    return pd.DataFrame(input_features, index=[0])

# Display user input
user_input = user_input_features()
st.subheader('User Input Features')
st.write(user_input)

# Predict function
def predict_churn(input_data):
    prediction = model.predict(input_data)
    return prediction

# Button to trigger prediction
if st.button('Predict'):
    # Predict
    prediction = predict_churn(user_input.values)

    # Display prediction
    st.subheader('Prediction Probability')
    st.write(prediction[0][0])

    st.subheader('Prediction')
    if prediction > 0.5:
        st.write('Churn: Yes')
    else:
        st.write('Churn: No')
