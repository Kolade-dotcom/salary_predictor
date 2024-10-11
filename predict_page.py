import streamlit as st
import pickle as pkl
import numpy as np
from data import countries, education, major


def load_model():
    with open("saved_steps.pkl", "rb") as file:
        data = pkl.load(file)
    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]
le_major = data["le_major"]


def show_predict_page():
    st.title("Software Development Salary Prediction")
    st.write("### We need some infomation to predict the salary")

    country_val = st.selectbox("Country", countries)
    education_val = st.selectbox("Education", education)
    major_val = st.selectbox("Undergraduate Major", major)
    age_val = st.slider("Age", min_value=18, max_value=100, value=20)
    age1stcode_val = st.slider(
        "Age of first code", min_value=5, max_value=85, value=16)
    experience_val = st.slider(
        "Years of experience", min_value=0, max_value=50, value=3)

    clicked = st.button("Calculate Salary")
    if clicked:
        X = np.array([[country_val, education_val, experience_val,
                     age_val, age1stcode_val, major_val]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X[:, 5] = le_major.transform(X[:, 5])
        X = X.astype(float)

        pred_salary = regressor.predict(X)
        st.subheader(f"The estimate salary is ${pred_salary[0]:.2f}")
