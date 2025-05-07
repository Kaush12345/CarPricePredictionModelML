import preprocessor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("LinearRegressionModel.pkl")
car = pd.read_csv("cleaned_car.csv")

# App Title
st.title("🚗 Car Price Prediction App")

# Sidebar for user input
st.sidebar.header("Enter Car Details")

# Dropdown inputs
company = st.sidebar.selectbox("Company", sorted(car['company'].unique()))
car_model = st.sidebar.selectbox("Car Model", sorted(car[car['company'] == company]['name'].unique()))
year = st.sidebar.selectbox("Year", sorted(car['year'].unique(), reverse=True))
fuel_type = st.sidebar.selectbox("Fuel Type", sorted(car['fuel_type'].unique()))
kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, 10000)

# Predict Button
if st.button("Predict Price"):
    # Input dataframe
    input_df = pd.DataFrame({
        'name': [car_model],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })

    # Make prediction
    prediction = model.predict(input_df)
    st.success(f"Estimated Selling Price: ₹ {np.round(prediction[0], 2):,.2f}")

    # SHAP Explanation
    st.subheader("SHAP Explanation")

    # Try extracting preprocessor and regressor if model is a pipeline
    try:
        print(model.named_steps)
        regressor = model.named_steps['linearregression']
        X_transformed = preprocessor.transform(input_df)

        explainer = shap.Explainer(regressor, X_transformed)
        shap_values = explainer(X_transformed)

    except AttributeError:
        # If model is not a pipeline
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)

    # Plot SHAP
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(plt)
