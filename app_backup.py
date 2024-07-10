# Accessing secrets
try:
    api_key = st.secrets["api_key"]
    another_secret = st.secrets["another_secret"]
except Exception as e:
    st.error(f"Error accessing secrets: {e}")

# Import TensorFlow with error handling
try:
    from tensorflow.keras.models import load_model
    st.write("TensorFlow imported successfully")
except Exception as e:
    st.error(f"Error importing TensorFlow: {e}")



import streamlit as st
import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import requests
from bs4 import BeautifulSoup

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css("style.css")

# Define the model directory path
MODEL_DIR = "."

# Function to handle the user profile form
def user_profile_ui():
    st.header("User Profile Management")

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
    # Write-up on how to choose the right optimization objective
    st.markdown(
    """
    ## GET RISK INSIGHTS
    Delve into the intricacies of your investments. With our risk-adjusted returns chart, you can visually see the balance between potential returns and associated risks. Enhance your decision-making with insights that highlight not just returns, but also evaluate them based on the risks taken.
    """
)

# Function to get the correct date column name
def get_date_column(data):
    for col in ['Date', 'Dates', 'ds']:
        if col in data.columns:
            return col
    raise KeyError("No known date column found in the data.")

# Function to get the correct price column name
def get_price_column(data):
    for col in ['Close', 'Adj Close', 'Last Price', 'Price']:
        if col in data.columns:
            return col
    raise KeyError("No known price column found in the data.")

# Function to assign default asset classes
def assign_asset_classes(data):
    if 'Asset Class' not in data.columns:
        st.warning("No 'Asset Class' column found. Assigning default asset classes.")
        data['Asset Class'] = 'Equities'  # Default to Equities for simplicity
    return data

# Placeholder function to calculate performance metrics
def calculate_performance_metrics(data, risk_free_rate):
    price_col = get_price_column(data)
    returns = data[price_col].pct_change().dropna()
    expected_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (expected_return - (risk_free_rate / 100)) / volatility
    return expected_return, volatility, sharpe_ratio

def plot_portfolio_performance(data, portfolio_column):
    fig = go.Figure()
    date_col = get_date_column(data)
    price_col = get_price_column(data)
    
    if portfolio_column not in data.columns:
        st.warning("No 'Portfolio' column found. Calculating portfolio performance.")
        data[portfolio_column] = data[price_col]  # Default to price column for simplicity

    fig.add_trace(go.Scatter(x=data[date_col], y=data[portfolio_column], mode='lines', name='Optimized Portfolio', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data[date_col], y=data[price_col], mode='lines', name='Benchmark', line=dict(color='red')))

    fig.update_layout(title='Portfolio vs Benchmark Performance',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend_title='Legend',
                      hovermode='x unified')
    st.plotly_chart(fig)

def plot_asset_allocation(data):
    if 'Allocation' not in data.columns:
        st.warning("No 'Allocation' column found. Assigning default allocation.")
        data['Allocation'] = np.random.rand(len(data))  # Random allocation for simplicity
    
    allocation = data.groupby('Asset Class').sum(numeric_only=True)['Allocation']
    fig = px.pie(names=allocation.index, values=allocation.values, title='Asset Allocation by Asset Class')
    st.plotly_chart(fig)

# Portfolio Performance UI
def portfolio_performance_ui():
    st.subheader("Portfolio Performance Analysis")

    uploaded_file = st.file_uploader("Upload your portfolio data (CSV, Excel)", type=["csv", "xlsx"], key="portfolio_performance_file")

    sample_portfolios = {
        "Sample Portfolio 1": pd.DataFrame({
            "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
            "Close": np.random.rand(100).cumsum(),
            "Asset Class": ['Equities'] * 100,
            "Allocation": np.random.rand(100)
        }),
        "Sample Portfolio 2": pd.DataFrame({
            "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
            "Close": np.random.rand(100).cumsum(),
            "Asset Class": ['Bonds'] * 100,
            "Allocation": np.random.rand(100)
        }),
        "Sample Portfolio 3": pd.DataFrame({
            "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
            "Close": np.random.rand(100).cumsum(),
            "Asset Class": ['ETF'] * 100,
            "Allocation": np.random.rand(100)
        }),
        "Sample Portfolio 4": pd.DataFrame({
            "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
            "Close": np.random.rand(100).cumsum(),
            "Asset Class": ['Index'] * 100,
            "Allocation": np.random.rand(100)
        }),
    }

    portfolio_choice = st.multiselect("Choose sample portfolios", list(sample_portfolios.keys()), key="portfolio_performance_choice")
    
    if portfolio_choice:
        combined_data = pd.concat([sample_portfolios[choice] for choice in portfolio_choice], ignore_index=True)
    else:
        combined_data = pd.DataFrame()

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

        st.write("Data Preview:")
        st.write(data.head())

        combined_data = pd.concat([combined_data, data], ignore_index=True)

    if not combined_data.empty:
        risk_free_rate = st.number_input("Risk Free Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
        benchmark = st.selectbox("Benchmark", ["HSI Index", "UKX Index", "SX5E Index", "SPX Index"])
        lookback_period = st.selectbox("Lookback Period", ["1 month", "3 months", "6 months", "1 year"])
        optimization_date = st.date_input("Optimization Date")

        if st.button("Run Analysis"):
            st.write("Running portfolio performance analysis...")
            combined_data = assign_asset_classes(combined_data)
            expected_return, volatility, sharpe_ratio = calculate_performance_metrics(combined_data, risk_free_rate)
            st.write(f"Expected Return: {expected_return:.2%}")
            st.write(f"Volatility: {volatility:.2%}")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            plot_portfolio_performance(combined_data, 'Portfolio')
            plot_asset_allocation(combined_data)

    st.markdown("""
    ## HOW TO CHOOSE THE RIGHT OPTIMIZATION OBJECTIVE
    Choosing the right objective for your portfolio optimization depends on your investment goals, risk tolerance, and investment horizon. Consider the following factors when selecting an objective:

    **Risk tolerance:** Assess your level of comfort with fluctuations in the value of your investments. If you're risk-averse, objectives like minimizing volatility or CVaR might be more suitable. Or consider maximizing quadratic utility or Sharpe ratio if you're willing to take on higher risk for potentially higher returns.

    **Investment goals:** Align the objective with your specific financial goals. For example, if you're saving for a long-term goal like retirement, you might focus on maximizing the Sharpe ratio to achieve higher risk-adjusted returns. Minimizing volatility could be a better choice if you're more concerned with preserving capital.

    **Investment horizon:** Consider the time frame of your investments. Longer investment horizons often allow for greater risk-taking, as markets tend to fluctuate over short periods but exhibit long-term growth trends. Shorter horizons may require a more conservative approach, such as minimizing volatility.

    **Diversification:** Ensure your portfolio includes positions diversified across different asset classes, sectors, or regions. This can help mitigate risk, create a more resilient portfolio, and achieve better optimization results.

    Remember, there is no one-size-fits-all approach to choosing an objective. Understanding your financial goals, risk appetite, and investment horizon is essential to make the best decision for your unique situation.
    """)



# Main Risk Assessment UI function
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Helper functions
def get_date_column(data):
    for col in ['Date', 'Dates', 'ds']:
        if col in data.columns:
            return col
    raise KeyError("No known date column found in the data.")

def get_price_column(data):
    for col in ['Close', 'Adj Close', 'Last Price', 'Price']:
        if col in data.columns:
            return col
    raise KeyError("No known price column found in the data.")

def assign_asset_classes(data):
    if 'Asset Class' not in data.columns:
        st.warning("No 'Asset Class' column found. Assigning default asset classes.")
        data['Asset Class'] = 'Equities'  # Default to Equities for simplicity
    return data

# Sample portfolios and beta data
sample_portfolios = {
    "Sample Portfolio 1": pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
        "Close": np.random.rand(100).cumsum(),
        "Asset Class": ['Equities'] * 100,
        "Allocation": np.random.rand(100)
    }),
    "Sample Portfolio 2": pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
        "Close": np.random.rand(100).cumsum(),
        "Asset Class": ['Bonds'] * 100,
        "Allocation": np.random.rand(100)
    }),
    "Sample Portfolio 3": pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
        "Close": np.random.rand(100).cumsum(),
        "Asset Class": ['ETF'] * 100,
        "Allocation": np.random.rand(100)
    }),
    "Sample Portfolio 4": pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
        "Close": np.random.rand(100).cumsum(),
        "Asset Class": ['Index'] * 100,
        "Allocation": np.random.rand(100)
    }),
}

beta_data = {
    "HSI Index": pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
        "Close": np.random.rand(100).cumsum(),
    }),
    "UKX Index": pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
        "Close": np.random.rand(100).cumsum(),
    }),
    "SX5E Index": pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
        "Close": np.random.rand(100).cumsum(),
    }),
    "SPX Index": pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq='D'),
        "Close": np.random.rand(100).cumsum(),
    }),
}

# Main Risk Assessment UI function
def risk_assessment_ui():
    st.subheader("Run Risk Assessment")
    uploaded_file = st.file_uploader("Upload your risk assessment data (CSV, Excel)", type=["csv", "xlsx"])

    portfolio_choice = st.multiselect("Choose sample portfolios", list(sample_portfolios.keys()))
    benchmark_choice = st.multiselect("Select Benchmarks for Beta Calculation", list(beta_data.keys()))

    combined_data = pd.concat([sample_portfolios[choice] for choice in portfolio_choice], ignore_index=True) if portfolio_choice else pd.DataFrame()
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                uploaded_data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                uploaded_data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return
            st.write("Data Preview:")
            st.write(uploaded_data.head())
            combined_data = pd.concat([combined_data, uploaded_data], ignore_index=True)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

    if not combined_data.empty:
        combined_data = assign_asset_classes(combined_data)
        date_col = get_date_column(combined_data)
        price_col = get_price_column(combined_data)
        combined_data['Return'] = combined_data[price_col].pct_change().dropna()
        combined_data = combined_data.dropna(subset=['Return'])

        if st.button("Run Risk Assessment"):
            st.write("Running risk assessment analysis...")

            for benchmark in benchmark_choice:
                beta_result = calculate_beta(combined_data, beta_data[benchmark])
                st.write(f"Beta for {benchmark}: {beta_result:.2f}")

            plot_risk_assessment_metrics(combined_data)

def calculate_beta(portfolio_returns, benchmark_data):
    price_col = get_price_column(benchmark_data)
    benchmark_data['Return'] = benchmark_data[price_col].pct_change().dropna()
    portfolio_returns = portfolio_returns.dropna(subset=['Return'])
    beta = portfolio_returns['Return'].cov(benchmark_data['Return']) / benchmark_data['Return'].var()
    return beta

def plot_risk_assessment_metrics(data):
    # Filter only numeric columns for covariance and correlation matrices
    numeric_data = data.select_dtypes(include=[np.number])

    # Calculate VaR and CVaR
    VaR_95 = np.percentile(numeric_data['Return'], 5)
    CVaR_95 = numeric_data[numeric_data['Return'] <= VaR_95]['Return'].mean()

    # Calculate Standard Deviation
    std_dev = numeric_data['Return'].std()

    # Plot Mean Return
    st.subheader("Mean Return")
    mean_return_fig = px.line(numeric_data, y='Return', title='Mean Return')
    st.plotly_chart(mean_return_fig)

    # Plot Covariance Matrix
    st.subheader("Covariance Matrix")
    cov_matrix = numeric_data.cov()
    fig, ax = plt.subplots()
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Plot Correlation Matrix
    st.subheader("Correlation Matrix")
    corr_matrix = numeric_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Display VaR, CVaR, Standard Deviation
    st.subheader("Risk Metrics")
    st.write(f"Value at Risk (VaR 95): {VaR_95:.2f}")
    st.write(f"Conditional Value at Risk (CVaR 95): {CVaR_95:.2f}")
    st.write(f"Standard Deviation: {std_dev:.2f}")

    # Provide advice based on the calculated metrics
    st.subheader("Advice Based on Risk Assessment")
    if VaR_95 > -0.02:
        st.write("The portfolio is relatively low risk, with a VaR of less than -2%. Consider maintaining your current allocation.")
    else:
        st.write("The portfolio has a higher risk with a VaR greater than -2%. Consider rebalancing to reduce risk.")





# Market Trends Analysis UI
# Function to get the column name
import os
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Function to get the column name
def get_column_name(data, possible_names):
    for name in possible_names:
        if name in data.columns:
            return name
    raise KeyError(f"None of the columns found: {possible_names}")

def load_bi_lstm_model():
    try:
        model = load_model(os.path.join(MODEL_DIR, "bi_lstm_model.keras"), compile=False)
        model.compile(optimizer=Adam(), loss='mse')
        return model
    except Exception as e:
        st.error(f"Error loading Bi-LSTM model: {e}")
        return None

# Function to calculate moving averages
def calculate_moving_averages(data, close_col, window=20):
    data[f'SMA_{window}'] = data[close_col].rolling(window=window).mean()
    data[f'EMA_{window}'] = data[close_col].ewm(span=window, adjust=False).mean()
    return data

# Function to calculate RSI
def calculate_rsi(data, close_col, window=14):
    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data, close_col):
    exp1 = data[close_col].ewm(span=12, adjust=False).mean()
    exp2 = data[close_col].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to make predictions using the Bi-LSTM model
def make_bi_lstm_predictions(data, model, close_col, volume_col, lookback=60):
    try:
        data = data.dropna(subset=[close_col, volume_col])
        if len(data) < lookback:
            st.error("Not enough data to form the lookback window for Bi-LSTM predictions.")
            return data
        
        X = []
        y_pred = []
        
        for i in range(lookback, len(data)):
            X.append(data[[close_col, volume_col]].values[i-lookback:i])
        
        X = np.array(X)
        
        # Ensure the shape of X matches the model's input shape
        required_features = 7
        if X.shape[-1] != required_features:
            X = np.concatenate([X, np.zeros((X.shape[0], X.shape[1], required_features - X.shape[-1]))], axis=-1)
        
        y_pred = model.predict(X)
        
        data['Predicted Close'] = np.nan
        data.loc[data.index[lookback:], 'Predicted Close'] = y_pred.flatten()
        return data
    
    except Exception as e:
        st.error(f"Error during Bi-LSTM prediction: {e}")
        return data

# Function to plot market trends
def plot_market_trends(data, close_col, volume_col, ticker):
    st.write(f"### {ticker} Market Trends")
    st.line_chart(data[[close_col, 'Predicted Close']].rename(columns={close_col: 'Actual', 'Predicted Close': 'Predicted'}))
    st.bar_chart(data[volume_col].rename(volume_col))
    st.write(f"### {ticker} Technical Indicators")
    st.line_chart(data[['SMA_20', 'EMA_20']].rename(columns={'SMA_20': 'SMA 20', 'EMA_20': 'EMA 20'}))
    st.line_chart(data['RSI'].rename('RSI'))
    st.line_chart(data[['MACD', 'MACD_Signal']].rename(columns={'MACD': 'MACD', 'MACD_Signal': 'MACD Signal'}))

# Function for market trends analysis UI
def market_trends_analysis_ui():
    st.title("Market Trends Analysis")

    # Creating larger sample datasets to meet the Bi-LSTM model's lookback requirement
    date_range = pd.date_range(start='1/1/2020', periods=150, freq='D')
    sample_data_1 = {
        "Date": date_range,
        "Ticker": ["AAPL"] * 150,
        "Close": np.random.normal(150, 10, 150).tolist(),
        "Volume": np.random.normal(1000000, 50000, 150).tolist()
    }
    sample_data_2 = {
        "Date": date_range,
        "Ticker": ["AMZN"] * 150,
        "Close": np.random.normal(3500, 100, 150).tolist(),
        "Volume": np.random.normal(1200000, 60000, 150).tolist()
    }
    sample_data_3 = {
        "Date": date_range,
        "Ticker": ["FB"] * 150,
        "Close": np.random.normal(350, 15, 150).tolist(),
        "Volume": np.random.normal(1100000, 55000, 150).tolist()
    }
    sample_data_4 = {
        "Date": date_range,
        "Ticker": ["ADBE"] * 150,
        "Close": np.random.normal(500, 20, 150).tolist(),
        "Volume": np.random.normal(1250000, 70000, 150).tolist()
    }

    sample_portfolios = {
        "Sample Portfolio 1": pd.DataFrame(sample_data_1).set_index('Date'),
        "Sample Portfolio 2": pd.DataFrame(sample_data_2).set_index('Date'),
        "Sample Portfolio 3": pd.DataFrame(sample_data_3).set_index('Date'),
        "Sample Portfolio 4": pd.DataFrame(sample_data_4).set_index('Date')
    }

    beta_data_ukx = {
        "Date": date_range,
        "Ticker": ["UKX"] * 150,
        "Close": np.random.normal(7000, 100, 150).tolist(),
        "Volume": np.random.normal(500000, 20000, 150).tolist()
    }
    beta_data_sp500 = {
        "Date": date_range,
        "Ticker": ["S&P 500"] * 150,
        "Close": np.random.normal(4500, 100, 150).tolist(),
        "Volume": np.random.normal(600000, 25000, 150).tolist()
    }
    beta_data_hsi = {
        "Date": date_range,
        "Ticker": ["HSI"] * 150,
        "Close": np.random.normal(25000, 500, 150).tolist(),
        "Volume": np.random.normal(700000, 30000, 150).tolist()
    }
    beta_data_sx5e = {
        "Date": date_range,
        "Ticker": ["SX5E"] * 150,
        "Close": np.random.normal(4000, 100, 150).tolist(),
        "Volume": np.random.normal(800000, 35000, 150).tolist()
    }

    beta_portfolios = {
        "UKX": pd.DataFrame(beta_data_ukx).set_index('Date'),
        "S&P 500": pd.DataFrame(beta_data_sp500).set_index('Date'),
        "HSI": pd.DataFrame(beta_data_hsi).set_index('Date'),
        "SX5E": pd.DataFrame(beta_data_sx5e).set_index('Date')
    }

    portfolio_choice = st.multiselect("Choose sample portfolios", list(sample_portfolios.keys()))
    beta_choice = st.multiselect("Choose beta portfolios", list(beta_portfolios.keys()))

    uploaded_file = st.file_uploader("Upload your market data (CSV, Excel)", type=["csv", "xlsx"])
    
    combined_data = pd.DataFrame()  # Initialize an empty DataFrame

    if portfolio_choice:
        try:
            combined_data = pd.concat([sample_portfolios[choice] for choice in portfolio_choice], ignore_index=False)
        except Exception as e:
            st.error(f"Error concatenating sample portfolios: {e}")

    if uploaded_file:
        with st.spinner("Loading data..."):
            try:
                if uploaded_file.name.endswith(".csv"):
                    user_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                else:
                    user_data = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
                    
                user_data['Portfolio'] = 'User Data'
                combined_data = pd.concat([combined_data, user_data], ignore_index=False)
            except Exception as e:
                st.error(f"Error loading data: {e}")

    if not combined_data.empty:
        try:
            close_col = get_column_name(combined_data, ['Close', 'Last Price', 'Adjusted Close', 'Price'])
            volume_col = get_column_name(combined_data, ['Volume', 'Volumes'])
            
            combined_data = combined_data.dropna(subset=[close_col, volume_col])
            combined_data[close_col] = combined_data[close_col].astype(float)
            combined_data[volume_col] = combined_data[volume_col].astype(float)
            
            window = st.slider("Select Moving Average Window", min_value=5, max_value=50, value=20, step=1)
            
            if st.button("Run Market Trends Analysis"):
                st.write("Running market trends analysis...")
                combined_data = calculate_moving_averages(combined_data, close_col, window)
                combined_data = calculate_rsi(combined_data, close_col)
                combined_data = calculate_macd(combined_data, close_col)
                
                bi_lstm_model = load_bi_lstm_model()  # Ensure the model is loaded before use
                if bi_lstm_model:
                    combined_data = make_bi_lstm_predictions(combined_data, bi_lstm_model, close_col, volume_col)
                
                plot_market_trends(combined_data, close_col, volume_col, combined_data['Ticker'].iloc[0])
                
                for beta in beta_choice:
                    try:
                        beta_data = beta_portfolios[beta]
                        beta_data = calculate_moving_averages(beta_data, close_col, window)
                        beta_data = calculate_rsi(beta_data, close_col)
                        beta_data = calculate_macd(beta_data, close_col)
                        beta_data = make_bi_lstm_predictions(beta_data, bi_lstm_model, close_col, volume_col)
                        plot_market_trends(beta_data, close_col, volume_col, beta)
                    except Exception as e:
                        st.error(f"Error processing beta data: {e}")
        except Exception as e:
            st.error(f"Error processing combined data: {e}")
            st.write(f"Debug Info: combined_data columns - {combined_data.columns}, combined_data shape - {combined_data.shape}")

    st.markdown("### Analysis and Advice")
    st.write("""
        Based on the analysis of the selected portfolios and market trends, here are some insights:
        
        - **Moving Averages (SMA and EMA)**: Moving averages can help identify the trend direction. If the current price is above the moving average, it suggests an uptrend. Conversely, if the price is below the moving average, it indicates a downtrend.
        
        - **RSI (Relative Strength Index)**: The RSI is a momentum oscillator that measures the speed and change of price movements. An RSI above 70 suggests that the asset is overbought, while an RSI below 30 indicates that it is oversold.
        
        - **MACD (Moving Average Convergence Divergence)**: The MACD is used to identify changes in the strength, direction, momentum, and duration of a trend. When the MACD crosses above the signal line, it may indicate a buy signal, and when it crosses below the signal line, it may indicate a sell signal.

        - **Volume Analysis**: Volume analysis can provide insights into the strength of a price move. Increasing volume suggests that a price move is more likely to be sustained, while decreasing volume can indicate that a price move may be losing strength.

        Based on the indicators and trends observed, consider the following advice:
        
        - **For Uptrending Markets**: If the moving averages and RSI suggest an uptrend, consider holding or buying more of the asset, but be cautious of overbought signals from the RSI.
        
        - **For Downtrending Markets**: If the indicators suggest a downtrend, consider reducing exposure or selling the asset. Watch for oversold signals from the RSI that might indicate a potential buying opportunity.

        - **Diversification**: Ensure that your portfolio is well-diversified to spread risk. Avoid putting all your investments into a single asset or asset class.

        - **Risk Management**: Use stop-loss orders to limit potential losses and protect your investments. Regularly review and adjust your investment strategy based on market conditions and your risk tolerance.
    """)





# Placeholder functions for analysis
def run_portfolio_analysis(data, risk_free_rate, benchmark, lookback_period, optimization_date):
    st.write(f"Running portfolio analysis with Risk Free Rate: {risk_free_rate}%, Benchmark: {benchmark}, Lookback Period: {lookback_period}, Optimization Date: {optimization_date}")
    data = assign_asset_classes(data)
    expected_return, volatility, sharpe_ratio = calculate_performance_metrics(data, risk_free_rate)
    st.write(f"Expected Return: {expected_return:.2%}")
    st.write(f"Volatility: {volatility:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    plot_portfolio_performance(data, 'Portfolio')
    plot_asset_allocation(data)

def run_risk_assessment(data, benchmark_data):
    price_col = get_price_column(data)
    returns = data[price_col].pct_change().dropna()
    if benchmark_data is not None:
        benchmark_col = get_price_column(benchmark_data)
        benchmark_returns = benchmark_data[benchmark_col].pct_change().dropna()
        common_dates = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        beta = returns.cov(benchmark_returns) / benchmark_returns.var()
    else:
        beta = None

    cov_matrix = returns.cov()
    corr_matrix = returns.corr()
    mean_return = returns.mean()
    rolling_std = returns.rolling(window=20).std()
    sma20 = data[price_col].rolling(window=20).mean()
    ema50 = data[price_col].ewm(span=50, adjust=False).mean()

    # Plot the risk metrics
    fig1 = px.line(data, x=data.index, y=price_col, title='SMA and EMA')
    fig1.add_scatter(x=data.index, y=sma20, mode='lines', name='SMA 20')
    fig1.add_scatter(x=data.index, y=ema50, mode='lines', name='EMA 50')

    fig2 = px.imshow(cov_matrix, title='Covariance Matrix', labels=dict(x='Assets', y='Assets'))
    fig3 = px.imshow(corr_matrix, title='Correlation Matrix', labels=dict(x='Assets', y='Assets'))

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)

    return {
        "Beta": beta,
        "Covariance Matrix": cov_matrix,
        "Correlation Matrix": corr_matrix,
        "Mean Return": mean_return,
        "Rolling Std Dev": rolling_std,
        "SMA 20": sma20,
        "EMA 50": ema50
    }

def run_market_trends_analysis(data):
    # Placeholder for market trends analysis implementation
    st.write("Market Trends Analysis Placeholder")
    # Example: Plotting moving averages
    price_col = get_price_column(data)
    sma20 = data[price_col].rolling(window=20).mean()
    ema50 = data[price_col].ewm(span=50, adjust=False).mean()

    fig = px.line(data, x=data.index, y=price_col, title='Market Trends Analysis')
    fig.add_scatter(x=data.index, y=sma20, mode='lines', name='SMA 20')
    fig.add_scatter(x=data.index, y=ema50, mode='lines', name='EMA 50')

    st.plotly_chart(fig)

# Model loading functions
def load_bi_lstm_model():
    try:
        model = load_model(os.path.join(MODEL_DIR, "bi_lstm_model.keras"), compile=False)
        model.compile(optimizer=Adam(), loss='mse')
        return model
    except Exception as e:
        return None

def load_random_forest_model():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "random_forest_volatility_model.pkl"))
        return model
    except Exception as e:
        return None

def load_sac_model():
    try:
        sac_model_dir = os.path.join(MODEL_DIR, "sac_portfolio_optimisation_model")
        actor_path = os.path.join(sac_model_dir, "policy.pth")
        critic_path = os.path.join(sac_model_dir, "critic.optimizer.pth")
        ent_coef_optimizer_path = os.path.join(sac_model_dir, "ent_coef_optimizer.pth")
        
        actor = torch.load(actor_path, map_location=torch.device('cpu'))
        critic = torch.load(critic_path, map_location=torch.device('cpu'))
        ent_coef_optimizer = torch.load(ent_coef_optimizer_path, map_location=torch.device('cpu'))
        
        return actor, critic, ent_coef_optimizer
    except Exception as e:
        return None, None, None

# Check if model files exist
def check_model_file(filename):
    return os.path.isfile(os.path.join(MODEL_DIR, filename))

# Check model files
if not check_model_file("bi_lstm_model.keras"):
    st.error("BI-LSTM model file not found.")
if not check_model_file("random_forest_volatility_model.pkl"):
    st.error("Random Forest model file not found.")
if not check_model_file("sac_portfolio_optimisation_model/policy.pth"):
    st.error("SAC model file not found.")
if not check_model_file("imputer.joblib"):
    st.error("Imputer file not found.")
if not check_model_file("preprocessing_steps.pkl"):
    st.error("Preprocessing steps file not found.")

# Load models and preprocessing objects
bi_lstm_model = load_bi_lstm_model()
random_forest_model = load_random_forest_model()
actor, critic, ent_coef_optimizer = load_sac_model()


# Footer with clickable links and moving disclaimer
st.markdown(
    """
    <div class="footer">
        <div class="marquee">
            Disclaimer: This is a fictional app for academic purposes only. Investment involves risks, including the loss of principal. Always consult with a qualified financial advisor before making any investment decisions.
        </div>
        <div class="footer-links">
            <a href="#about-us" onclick="document.querySelector('#about-us').scrollIntoView();">About Us</a> |
            <a href="#services" onclick="document.querySelector('#services').scrollIntoView();">Services</a> |
            <a href="#products" onclick="document.querySelector('#products').scrollIntoView();">Products</a> |
            <a href="#investor-advice" onclick="document.querySelector('#investor-advice').scrollIntoView();">Investor Advice Sessions</a>
        </div>
    </div>
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: black;
            color: white;
            text-align: center;
            padding: 10px 0;
        }
        .footer .marquee {
            display: inline-block;
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
        }
        .footer .marquee div {
            display: inline-block;
            padding-left: 100%;
            animation: marquee 15s linear infinite;
        }
        @keyframes marquee {
            from { transform: translate(0, 0); }
            to { transform: translate(-100%, 0); }
        }
        .footer-links a {
            color: white;
            margin: 0 10px;
            text-decoration: none;
        }
        .footer-links a:hover {
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True
)

import requests
import streamlit as st
from PIL import Image
from io import BytesIO

def render_investor_news():
    st.title("Investor News")

    api_key = '9MYRNNQDJ58W2U89'  # Replace with your actual Alpha Vantage API key
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if request was successful
        
        news_data = response.json()
        articles = news_data.get('feed', [])
        
        if not articles:
            st.write("No news items found.")
        else:
            for article in articles[:5]:  # Limiting to the first 5 news items for simplicity
                headline = article.get('title')
                link = article.get('url')
                description = article.get('summary')
                image_url = article.get('banner_image')
                published = article.get('time_published')
                source = article.get('source')

                if headline and link:
                    st.subheader(headline)
                    if published:
                        st.write(f"Published on: {published}")
                    if source:
                        st.write(f"Source: {source}")
                    if image_url:
                        try:
                            image_response = requests.get(image_url)
                            img = Image.open(BytesIO(image_response.content))
                            st.image(img, width=700)
                        except Exception:
                            pass  # Ignore any image loading issues
                    if description:
                        st.write(description)
                    st.markdown(f"[Read more]({link})")
                    st.write("---")  # Horizontal line for separation

    except requests.RequestException as e:
        st.error(f"Error fetching news: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


import streamlit as st
import requests
from PIL import Image
from io import BytesIO

def render_contact_us():
    st.title("Contact Us")
    
    st.header("Get in Touch")
    st.write("If you have any questions or would like to get in touch, please use the following contact details:")
    
    st.write("**Address:**")
    st.write("Blenheim Flat, Marlborough Street BS1 3NW, Bristol, United Kingdom")
    
    st.write("**Email:**")
    st.write("charlassetmanagement@gmail.com")
    
    st.write("**Social Media:**")
    st.write("[LinkedIn](https://www.linkedin.com)")
    st.write("[Instagram](https://www.instagram.com)")
    st.write("[X (formerly Twitter)](https://twitter.com)")

    st.header("Feedback")
    st.write("We value your feedback. Please fill out the questionnaire below to help us improve our services:")
    st.write("[Investor Feedback on Portfolio Optimization Product](https://forms.office.com/e/bwBXrYYu0j)")

    st.write("Or fill the embedded form below:")
    st.components.v1.iframe("https://forms.office.com/e/bwBXrYYu0j", width=700, height=800)



# Sidebar Navigation
st.sidebar.title("CHARL ASSET MANAGEMENT LIMITED")

st.sidebar.title("Navigation")
pages = ["Home", "Portfolio Performance", "Risk Assessment", "Market Trends Analysis", "Investor News", "Investor Education", "Contact Us"]
page = st.sidebar.radio("Go to", pages, key="unique_navigation_radio")

def render_home():
    st.title("Home")
    st.write("Welcome to the Portfolio Optimisation app by Charlof Asset Management Limited.")
    user_profile_ui()

def render_portfolio_performance():
    st.title("Portfolio Performance")
    portfolio_performance_ui()

def render_risk_assessment():
    st.title("Risk Assessment")
    risk_assessment_ui()

def render_market_trends():
    st.title("Market Trends Analysis")
    market_trends_analysis_ui()


def render_investor_education():
    st.title("Investor Education")
    st.write("Educational content for investors.")

    st.header("Portfolio Management")
    st.write("""
        Portfolio management involves managing an individual's investments in the form of bonds, shares, cash, mutual funds, etc. It is all about the selection and management of an investment policy that minimizes risk and maximizes returns.
        
        Key concepts include asset allocation, diversification, and risk management.
        
        - **Asset Allocation**: The process of dividing investments among different kinds of assets to optimize risk and return.
        - **Diversification**: A risk management strategy that mixes a wide variety of investments within a portfolio.
        - **Risk Management**: The process of identification, analysis, and acceptance or mitigation of uncertainty in investment decisions.
    """)

    st.header("Trading")
    st.write("""
        Trading involves buying and selling financial instruments like stocks, bonds, commodities, and derivatives. There are various types of trading strategies:

        - **Day Trading**: Buying and selling within the same trading day.
        - **Swing Trading**: Taking advantage of short- to medium-term price movements.
        - **Position Trading**: Holding investments for a longer period to benefit from longer-term trends.

        Key concepts include technical analysis, fundamental analysis, and trading psychology.
    """)

    st.header("Securities")
    st.write("""
        Securities are financial instruments that represent ownership in a company (stocks), a creditor relationship with a governmental body or corporation (bonds), or rights to ownership as represented by an option.

        - **Stocks**: Equities that represent ownership in a corporation and entitle the owner to a part of the corporation’s earnings and assets.
        - **Bonds**: Debt securities issued by entities to raise capital, where the issuer owes the holders a debt and is obliged to pay interest and/or repay the principal at a later date.
        - **Mutual Funds**: Investment vehicles that pool money from many investors to purchase securities.

        Understanding these securities helps in making informed investment decisions.
    """)

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


