import streamlit as st
import joblib
import pandas as pd

# Load models
rf_model = joblib.load('random_forest_model.joblib')
imputer = joblib.load('imputer.joblib')

# Title of the app
st.title('Risk Assessment Using Random Forest Model')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Uploaded:")
    st.write(data.head())

    # Preprocess the data
    data_imputed = imputer.transform(data)

    # Run model predictions
    predictions = rf_model.predict(data_imputed)

    # Display predictions
    st.write("Predictions:")
    st.write(predictions)

    # Visualization of feature importances
    st.write("Feature Importances:")
    importances = rf_model.feature_importances_
    features = data.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df)

# If you want to display the uploaded data as a table
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
