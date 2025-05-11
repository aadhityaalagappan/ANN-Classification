import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf
import pickle


# Load the ANN model
model = tf.keras.models.load_model("model.h5")

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
    
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)           

print("Model loaded successfully.")

#streamlit app
st.title("Customer Churn Prediction")

#User input
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=  one_hot_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([pd.DataFrame(input_data).reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

st.write("Probability of Churn:", prediction[0][0])
if prediction[0][0] > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")

