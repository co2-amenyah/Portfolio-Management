import os
import streamlit as st
import pandas as pd

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css("style.css")

# Home Page
def render_home():
    st.title("Home")
    st.write("Welcome to the Portfolio Optimisation app.")
    user_profile_ui()

# Portfolio Performance Page
def render_portfolio_performance():
    st.title("Portfolio Performance")
    st.subheader("Run Portfolio Analysis")

    uploaded_file = st.file_uploader("Choose a CSV file for portfolio analysis", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        # Call the SAC model analysis function here (we will add this in the next step)

# Risk Assessment Page
def render_risk_assessment():
    st.title("Risk Assessment")
    st.subheader("Run Risk Assessment")

    uploaded_file = st.file_uploader("Choose a CSV file for risk assessment", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        # Call the Random Forest model analysis function here (we will add this in the next step)

# Market Trends Analysis Page
def render_market_trends():
    st.title("Market Trends Analysis")
    st.subheader("Run Market Trends Analysis")

    uploaded_file = st.file_uploader("Choose a CSV file for market trends analysis", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        # Call the BI-LSTM model analysis function here (we will add this in the next step)

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Portfolio Performance", "Risk Assessment", "Market Trends Analysis", "Investor News", "Investor Education", "Contact Us"]
page = st.sidebar.radio("Go to", pages, key="navigation_radio")

if page == "Home":
    render_home()
elif page == "Portfolio Performance":
    render_portfolio_performance()
elif page == "Risk Assessment":
    render_risk_assessment()
elif page == "Market Trends Analysis":
    render_market_trends()
elif page == "Investor News":
    st.title("Investor News")
    st.write("Latest news for investors.")
elif page == "Investor Education":
    st.title("Investor Education")
    st.write("Educational content for investors.")
elif page == "Contact Us":
    st.title("Contact Us")
    st.write("Contact details here.")

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
