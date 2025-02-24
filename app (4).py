import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load('weather_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Load models
model, scaler = load_models()

if model is None or scaler is None:
    st.error("Failed to load models. Please check the model files.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide"
)

# Title
st.title("🌦️ Weather Prediction App")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Temperature Parameters")
    min_temp = st.number_input('Minimum Temperature (°C)', value=15.0)
    max_temp = st.number_input('Maximum Temperature (°C)', value=25.0)
    rainfall = st.number_input('Rainfall (mm)', value=0.0)

with col2:
    st.subheader("Other Parameters")
    humidity = st.number_input('Humidity at 3pm (%)', value=50.0)
    pressure = st.number_input('Pressure at 3pm (hPa)', value=1015.0)
    wind_speed = st.number_input('Wind Speed at 3pm (km/h)', value=20.0)

# Create predict button
if st.button("Predict Rain Tomorrow"):
    try:
        # Create input data
        input_data = pd.DataFrame({
            'MinTemp': [min_temp],
            'MaxTemp': [max_temp],
            'Rainfall': [rainfall],
            'Humidity3pm': [humidity],
            'Pressure3pm': [pressure],
            'WindSpeed3pm': [wind_speed]
        })

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Show results
        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.error("🌧️ Rain is predicted tomorrow!")
        else:
            st.success("☀️ No rain is predicted tomorrow!")

        st.write(f"Probability of rain: {prediction_proba[1]:.2%}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add information about the model
with st.expander("About this model"):
    st.write("""
    This Random Forest model was trained on Australian weather data.
    The model uses various weather parameters to predict the likelihood of rain tomorrow.
    """)
