import os
import streamlit as st
import pandas as pd
import torch
from torch import nn
import joblib
import matplotlib.pyplot as plt

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

# Load SAC model
def load_sac_model():
    try:
        sac_model_dir = os.path.join(MODEL_DIR, "sac_portfolio_optimisation_model")
        actor_path = os.path.join(sac_model_dir, "policy.pth")
        actor_model = ActorModel(input_dim=10, output_dim=10)  # Adjust input/output dimensions as needed
        actor_model.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')), strict=False)
        st.write("SAC Model loaded successfully.")
        return actor_model
    except Exception as e:
        st.error(f"Error loading SAC model: {e}")
        return None

# Function to handle portfolio optimization
def optimize_portfolio(data, actor_model):
    with torch.no_grad():
        sample_data = torch.tensor(data.values, dtype=torch.float32)
        mu, log_std = actor_model(sample_data)
    portfolio_weights = mu.numpy()
    return portfolio_weights

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Portfolio Performance"]
page = st.sidebar.radio("Go to", pages, key="navigation_radio")

def render_home():
    st.title("Home")
    st.write("Welcome to the Portfolio Optimisation app.")

def render_portfolio_performance():
    st.title("Portfolio Performance")
    st.subheader("Run Portfolio Analysis")

    uploaded_file = st.file_uploader("Upload your CSV file for portfolio analysis", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", data.head())

        actor_model = load_sac_model()
        if actor_model is not None:
            st.write("Running portfolio optimization...")
            portfolio_weights = optimize_portfolio(data, actor_model)
            st.write("Optimized Portfolio Weights:", portfolio_weights)

            # Visualization: Portfolio Allocation by Asset Class
            fig, ax = plt.subplots()
            asset_classes = ['Bonds', 'Equities', 'ETFs', 'Indices']  # Placeholder, replace with actual classes
            weights = portfolio_weights.flatten()
            ax.pie(weights, labels=asset_classes, autopct='%1.1f%%')
            ax.set_title('Portfolio Allocation by Asset Class')
            st.pyplot(fig)

# Navigation logic
if page == "Home":
    render_home()
elif page == "Portfolio Performance":
    render_portfolio_performance()
