import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="PJME Energy Demand Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with fixed text visibility
st.markdown("""
    <style>
    /* Base background */
    .stApp {
        background-color: white;
    }
    
    /* Main content area */
    .css-1d391kg {
        background-color: white;
    }
    
    /* Text colors */
    .stMarkdown, .stText, h1, h2, h3, p, span, label {
        color: #2c3e50 !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        color: #2c3e50;
    }
    
    /* Selectbox */
    .stSelectbox>div>div>input {
        color: #2c3e50;
    }
    
    /* Slider text */
    .stSlider>div>div>div {
        color: #2c3e50;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4addbe;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    
    /* Button hover */
    .stButton>button:hover {
        background-color: #3ac7ac;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f8f9fa;
    }
    .stMetric label {
        color: #2c3e50 !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        color: #ffffff !important;
        background-color: #4addbe;
    }
    .stError {
        color: #ffffff !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        color: #2c3e50 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #2c3e50 !important;
    }
    
    /* Sidebar */
    .css-sidebar .css-text {
        color: #2c3e50 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50 !important;
    }
    
    /* Labels */
    label {
        color: #2c3e50 !important;
    }
    
    /* Plot background */
    .js-plotly-plot .plotly .main-svg {
        background-color: white !important;
    }
    
    /* Ensure all text inputs are visible */
    input[type="text"], input[type="number"] {
        color: #2c3e50 !important;
    }
    
    /* Date picker */
    .stDateInput>div>div>input {
        color: #2c3e50 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("10.PJME_hourly.csv", index_col="Datetime")
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return None, str(e)

# Rest of your functions here...

def main():
    # Title with custom styling
    st.markdown("<h1 style='color: #2c3e50; text-align: center;'>PJME Energy Demand Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #2c3e50; text-align: center;'>Predict and analyze energy demand patterns</p>", unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.markdown("<div class='stError'>Could not load the dataset!</div>", unsafe_allow_html=True)
        return
        
    st.markdown("<div class='stSuccess'>Data loaded successfully!</div>", unsafe_allow_html=True)
    
    # Training section with custom styling
    st.markdown("<h2 style='color: #2c3e50; margin-top: 30px;'>Model Training</h2>", unsafe_allow_html=True)
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            try:
                model, test_data, feature_names = train_model(df)
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = feature_names
                st.markdown("<div class='stSuccess'>Model trained successfully!</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"<div class='stError'>Error during training: {str(e)}</div>", unsafe_allow_html=True)
                return

    # Prediction Interface
    if 'model' in st.session_state:
        st.markdown("<h2 style='color: #2c3e50; margin-top: 30px;'>Make Predictions</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Select Date")
            hour = st.slider("Hour of Day", 0, 23, 12)
            
        if st.button("Predict Energy Demand"):
            input_data = pd.DataFrame({
                'year': [date.year],
                'month': [date.month],
                'day': [date.day],
                'day_of_week': [date.weekday()],
                'hour': [hour]
            })
            
            try:
                prediction = st.session_state['model'].predict(input_data)[0]
                st.markdown(f"<div class='stSuccess'>Predicted Energy Demand: {prediction:,.0f} MW</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"<div class='stError'>Error during prediction: {str(e)}</div>", unsafe_allow_html=True)
        
        # Model Analysis section
        st.markdown("<h2 style='color: #2c3e50; margin-top: 30px;'>Model Analysis</h2>", unsafe_allow_html=True)
        
        # Add your plots here with updated styling...
        
if __name__ == "__main__":
    main()