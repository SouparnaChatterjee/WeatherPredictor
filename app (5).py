import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Debug information
st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# Load the saved model and scaler
@st.cache_resource
def load_models():
    try:
        # Print debug information
        st.write("Attempting to load models...")
        
        # Check if files exist
        if not os.path.exists('weather_model.pkl'):
            st.error("weather_model.pkl not found!")
            return None, None
        if not os.path.exists('scaler.pkl'):
            st.error("scaler.pkl not found!")
            return None, None
            
        # Load models
        model = joblib.load('weather_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        st.write("Models loaded successfully!")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Rest of your app.py code...
