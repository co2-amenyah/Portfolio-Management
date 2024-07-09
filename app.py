import os
import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import torch
import numpy as np
from torch import nn

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css("style.css")

# Define the model directory path
MODEL_DIR = "."

# Define the correct actor model architecture
class ActorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorModel, self).__init__()
        self.latent_pi = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, output_dim)
        self.log_std = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.latent_pi(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        return mu, log_std

# Load models
def load_bi_lstm_model():
    try:
        model = load_model(os.path.join(MODEL_DIR, "bi_lstm_model.keras"), compile=False)
        model.compile(optimizer=Adam(), loss='mse')  # Replace 'mse' with the actual loss function used
        st.write("BI-LSTM Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading BI-LSTM model: {e}")
        return None

def load_random_forest_model():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "random_forest_volatility_model.pkl"))
        st.write("Random Forest Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Random Forest model: {e}")
        return None

def load_sac_model():
    try:
        sac_model_dir = os.path.join(MODEL_DIR, "sac_portfolio_optimisation_model")
        actor_path = os.path.join(sac_model_dir, "policy.pth")
        # Create an instance of the ActorModel
        actor_model = ActorModel(input_dim=10, output_dim=10)
        actor_model.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')), strict=False)
        st.write("SAC Model loaded successfully.")
        return actor_model
    except Exception as e:
        st.error(f"Error loading SAC model: {e}")
        return None

# Check if model files exist
def check_model_file(filename):
    return os.path.isfile(os.path.join(MODEL_DIR, filename))

# Check model files
if not check_model_file("bi_lstm_model.keras"):
    st.error("BI-LSTM model file not found.")
if not check_model_file("random_forest_volatility_model.pkl"):
    st.error("Random Forest model file not found.")
if not check_model_file("sac_portfolio_optimisation_model.zip"):
    st.error("SAC model file not found.")
if not check_model_file("imputer.joblib"):
    st.error("Imputer file not found.")

# Load models and preprocessing objects
bi_lstm_model = load_bi_lstm_model()
random_forest_model = load_random_forest_model()
actor_model = load_sac_model()

# Sample data for testing
sample_data_bi_lstm = np.random.rand(1, 60, 7)  # Adjusted to match expected input shape for BI-LSTM model
sample_data_random_forest = np.random.rand(10, 14)  # Adjusted to match expected input features for Random Forest model
sample_data_sac = torch.tensor(np.random.rand(1, 10), dtype=torch.float32)  # Adjust as needed for SAC model

# Function to handle the user profile form
def user_profile_ui():
    st.title("User Profile Management")

    with st.form("user_profile_form"):
        full_name = st.text_input("Full Name", key="full_name_input")
        age_group = st.selectbox("Age Group", ["20-24", "25-34", "35-44", "45-54", "55-64", "65 and above", "Other"], key="age_group_input")
        gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary", "Other", "Prefer not to say"], key="gender_input")
        investment_experience = st.selectbox("Investment Experience", ["Beginner", "Intermediate", "Advanced", "Expert"], key="investment_experience_input")
        investment_objectives = st.selectbox("Investment Objectives", ["Capital Appreciation", "Income Generation", "Preservation of Capital", "Diversification", "Speculative Gains", "Other"], key="investment_objectives_input")
        risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"], key="risk_tolerance_input")
        benchmark = st.selectbox("Benchmark", ["S&P 500", "FTSE 100", "EURO STOXX 50", "Nikkei 225"], key="benchmark_input")
        investment_horizon = st.number_input("Investment Horizon (years)", min_value=0, key="investment_horizon_input")
        current_allocation = st.multiselect("Current Allocation", ["Bonds", "Equities", "ETF", "Index", "Cash"], key="current_allocation_input")
        preferred_classes = st.multiselect("Preferred Asset Classes", ["Bonds", "Equities", "ETF", "Index", "Cash"], key="preferred_classes_input")
        expected_returns = st.number_input("Expected Returns", min_value=0.0, max_value=1.0, key="expected_returns_input")
        budget = st.number_input("Budget", min_value=0.0, key="budget_input")

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("Information received. Thank you!")

# Portfolio Performance UI
def portfolio_performance_ui():
    st.title("Portfolio Performance")
    st.subheader("Run Portfolio Analysis")

    if st.button("Run Portfolio Analysis"):
        st.write("Running portfolio performance analysis...")
        performance_result = run_portfolio_analysis()
        st.write(performance_result)

# Risk Assessment UI
def risk_assessment_ui():
    st.title("Risk Assessment")
    st.subheader("Run Risk Assessment")

    if st.button("Run Risk Assessment"):
        st.write("Running risk assessment analysis...")
        risk_result = run_risk_assessment()
        st.write(risk_result)

# Market Trends Analysis UI
def market_trends_analysis_ui():
    st.title("Market Trends Analysis")
    st.subheader("Run Market Trends Analysis")

    if st.button("Run Market Trends Analysis"):
        st.write("Running market trends analysis...")
        trends_result = run_market_trends_analysis()
        st.write(trends_result)

# Functions for portfolio analysis, risk assessment, and market trends analysis
def run_portfolio_analysis():
    # Placeholder function for portfolio performance analysis
    # Integrate the SAC model for portfolio optimization
    if actor_model:
        st.write("Using SAC model for portfolio analysis...")
        # Simulating a model output
        try:
            # Example code for SAC model usage (replace with actual code)
            with torch.no_grad():
                mu, log_std = actor_model(sample_data_sac)
            return f"Portfolio Performance Analysis Results: mu={mu}, log_std={log_std}"
        except Exception as e:
            return f"Error during SAC model inference: {e}"
    else:
        return "SAC Model not loaded properly"

def run_risk_assessment():
    # Placeholder function for risk assessment analysis using the loaded Random Forest model
    if random_forest_model:
        st.write("Using Random Forest model for risk assessment...")
        # Simulating a model output
        try:
            # Example code for Random Forest model usage (replace with actual code)
            output = random_forest_model.predict(sample_data_random_forest)
            return f"Risk Assessment Analysis Results: {output}"
        except Exception as e:
            return f"Error during Random Forest model inference: {e}"
    else:
        return "Random Forest Model not loaded properly"

def run_market_trends_analysis():
    # Placeholder function for market trends analysis using the loaded BI-LSTM model
    if bi_lstm_model:
        st.write("Using BI-LSTM model for market trends analysis...")
        # Simulating a model output
        try:
            # Example code for BI-LSTM model usage (replace with actual code)
            output = bi_lstm_model.predict(sample_data_bi_lstm)
            return f"Market Trends Analysis Results: {output}"
        except Exception as e:
            return f"Error during BI-LSTM model inference: {e}"
    else:
        return "BI-LSTM Model not loaded properly"

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Portfolio Performance", "Risk Assessment", "Market Trends Analysis", "Investor News", "Investor Education", "Contact Us"]
page = st.sidebar.radio("Go to", pages, key="navigation_radio")

def render_home():
    st.title("Home")
    st.write("Welcome to the Portfolio Optimisation app.")
    user_profile_ui()

def render_portfolio_performance():
    portfolio_performance_ui()

def render_risk_assessment():
    risk_assessment_ui()

def render_market_trends():
    market_trends_analysis_ui()

def render_investor_news():
    st.title("Investor News")
    st.write("Latest news for investors.")

def render_investor_education():
    st.title("Investor Education")
    st.write("Educational content for investors.")

def render_contact_us():
    st.title("Contact Us")
    st.write("Contact details here.")

# Navigation logic
if page == "Home":
    render_home()
elif page == "Portfolio Performance":
    render_portfolio_performance()
elif page == "Risk Assessment":
    render_risk_assessment()
elif page == "Market Trends Analysis":
    render_market_trends()
elif page == "Investor News":
    render_investor_news()
elif page == "Investor Education":
    render_investor_education()
elif page == "Contact Us":
    render_contact_us()

# Footer with clickable links
st.markdown(
    """
    <div class="footer">
        <div class="marquee">Disclaimer: This is a fictional app for academic purposes only. Investment involves risks, including the loss of principal. Always consult with a qualified financial advisor before making any investment decisions.</div>
        <div class="footer-links">
            <a href="#about-us" onclick="document.querySelector('#about-us').scrollIntoView();">About Us</a> |
            <a href="#services" onclick="document.querySelector('#services').scrollIntoView();">Services</a> |
            <a href="#products" onclick="document.querySelector('#products').scrollIntoView();">Products</a> |
            <a href="#investor-advice" onclick="document.querySelector('#investor-advice').scrollIntoView();">Investor Advice Sessions</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
