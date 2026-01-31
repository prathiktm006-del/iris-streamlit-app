import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# Load trained model
model = joblib.load("iris_model.pkl")

# Title
st.markdown(
    "<h1 style='text-align:center; color:green;'>🌸 Iris Flower Prediction App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>ML app using Random Forest Classifier</p>",
    unsafe_allow_html=True
)

st.divider()

# Input layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)

with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

st.divider()

# Prediction button
if st.button("🌼 Predict Iris Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌸 Predicted Species: **{species[prediction[0]]}**")

st.divider()
st.caption("Built using Python, Scikit-learn & Streamlit")
