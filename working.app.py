import streamlit as st

# Debug information
st.write("Starting app...")

# Accessing secrets
try:
    st.write("Accessing secrets...")
    st.write(st.secrets)  # Print all available secrets
    api_key = st.secrets["secrets"]["api_key"]
    another_secret = st.secrets["secrets"]["another_secret"]
except Exception as e:
    st.error(f"Error accessing secrets: {e}")
    api_key = None
    another_secret = None

# Debug information
st.write("Secrets accessed")

# Import TensorFlow with error handling
try:
    st.write("Importing TensorFlow...")
    from tensorflow.keras.models import load_model
except Exception as e:
    st.error(f"Error importing TensorFlow: {e}")

# Debug information
st.write("TensorFlow imported")

# Use the secrets in your app
st.title("My Portfolio Optimization App")
if api_key and another_secret:
    st.write(f"API Key: {api_key}")
    st.write(f"Another Secret: {another_secret}")
else:
    st.write("Secrets not available.")

# Your existing Streamlit app code goes here
st.write("Welcome to the portfolio optimization platform.")
# Additional app logic

# Example of how to include some more application logic (e.g., a simple calculation or visualization)
st.write("This is a simple Streamlit app to demonstrate using secrets.")
