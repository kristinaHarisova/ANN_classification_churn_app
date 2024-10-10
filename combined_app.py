import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load models
churn_model = tf.keras.models.load_model('model.h5')
salary_model = tf.keras.models.load_model('regression_model.h5')

# Load churn encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load regression encoders and scaler
with open('regression_label_encoder_gender.pkl', 'rb') as file:
    reg_label_encoder_gender = pickle.load(file)

with open('regression_onehot_encoder_geo.pkl', 'rb') as file:
    reg_onehot_encoder_geo = pickle.load(file)

with open('regression_scaler.pkl', 'rb') as file:
    reg_scaler = pickle.load(file)
    
    
# Streamlit app title
st.title('Bank Customer Prediction App')

# Sidebar for task selection (Churn Prediction or Salary Prediction)
task = st.sidebar.selectbox('Choose Prediction Task', ['Churn Prediction', 'Salary Prediction'])

if task == 'Churn Prediction':
    # Inputs for Churn Prediction
    st.header('Customer Churn Prediction')

    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    # Prepare the input data for Churn Prediction
    churn_input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography' for Churn Prediction
    churn_geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    churn_geo_encoded_df = pd.DataFrame(churn_geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data for Churn Prediction
    churn_input_data = pd.concat([churn_input_data.reset_index(drop=True), churn_geo_encoded_df], axis=1)

    # Scale the input data for Churn Prediction
    churn_input_data_scaled = scaler.transform(churn_input_data)

    # Predict churn
    churn_prediction = churn_model.predict(churn_input_data_scaled)
    churn_probability = churn_prediction[0][0]

    # Display the result
    st.write(f'Churn Probability: {churn_probability:.2f}')
    if churn_probability > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

elif task == 'Salary Prediction':
    # Inputs for Salary Prediction
    st.header('Estimated Salary Prediction')

    geography = st.selectbox('Geography', reg_onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', reg_label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    exited = st.selectbox("Exited", [0, 1])
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    # Prepare the input data for Salary Prediction
    salary_input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [reg_label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]  # 'Exited' is only for salary prediction
    })

    # One-hot encode 'Geography' for Salary Prediction
    salary_geo_encoded = reg_onehot_encoder_geo.transform([[geography]]).toarray()
    salary_geo_encoded_df = pd.DataFrame(salary_geo_encoded, columns=reg_onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data for Salary Prediction
    salary_input_data = pd.concat([salary_input_data.reset_index(drop=True), salary_geo_encoded_df], axis=1)

    # Scale the input data for Salary Prediction
    salary_input_data_scaled = reg_scaler.transform(salary_input_data)

    # Predict estimated salary
    salary_prediction = salary_model.predict(salary_input_data_scaled)
    predicted_salary = salary_prediction[0][0]

    # Display the result
    st.write(f"Predicted Estimated Salary: ${predicted_salary: .2f}")
