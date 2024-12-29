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

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    .css-1d391kg {
        background-color: white;
    }
    .stButton>button {
        background-color: #4addbe;
        color: white;
        border: 1px solid #2c3e50 !important;
    }
    .stMarkdown, h1, h2, h3, p, span, label {
        color: #2c3e50 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("10.PJME_hourly.csv")
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')
        return df
    except Exception as e:
        return None, str(e)

def time_features(df):
    df = df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["hour"] = df.index.hour
    return df

def train_model(df):
    st.write("Starting model training...")
    
    df = time_features(df)
    st.write("Time features added...")
    
    length = df.shape[0]
    main = int(length * 0.8)
    trainer = df[:main]
    tester = df[main:]
    st.write("Data split completed...")
    
    X = tester.drop(columns=["PJME_MW"])
    y = tester.PJME_MW
    
    X_train, X_val, y_train, y_val = tts(X, y, train_size=0.8, random_state=42)
    st.write("Train-test split done...")
    
    model = xgb.XGBRegressor()
    st.write("Training XGBoost model...")
    model.fit(X_train, y_train, 
             eval_set=[(X_train, y_train), (X_val, y_val)],
             verbose=False)
    
    st.write("Model training completed!")
    return model, tester, X.columns.tolist()

def plot_predictions(tester, model):
    X_tester = tester.drop(columns=["PJME_MW"])
    result = model.predict(X_tester)
    tester = tester.copy()
    tester["prediction"] = np.round(result, 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tester.index,
        y=tester.PJME_MW,
        name="Actual",
        line=dict(color="#4addbe")
    ))
    
    fig.add_trace(go.Scatter(
        x=tester.index,
        y=tester.prediction,
        name="Predicted",
        line=dict(color="#34495e", dash="dot")
    ))
    
    fig.update_layout(
        title={
            'text': "Energy Demand - Actual vs Predicted",
            'font': {'color': '#2c3e50'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Date",
        yaxis_title="Energy Demand (MW)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#2c3e50"),
        legend=dict(
            font=dict(color="#2c3e50")
        ),
        xaxis=dict(
            title=dict(font=dict(color="#2c3e50")),
            tickfont=dict(color="#2c3e50")
        ),
        yaxis=dict(
            title=dict(font=dict(color="#2c3e50")),
            tickfont=dict(color="#2c3e50")
        )
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(importances, 
                x='importance', 
                y='feature',
                orientation='h',
                title={
                    'text': "Feature Importance",
                    'font': {'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                color_discrete_sequence=["#4addbe"])
    
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#2c3e50"),
        legend=dict(font=dict(color="#2c3e50")),
        xaxis=dict(
            title=dict(font=dict(color="#2c3e50")),
            tickfont=dict(color="#2c3e50")
        ),
        yaxis=dict(
            title=dict(font=dict(color="#2c3e50")),
            tickfont=dict(color="#2c3e50")
        )
    )
    
    return fig

def main():
    st.title("PJME Energy Demand Predictor")
    
    st.write("Attempting to load data...")
    df = load_data()
    
    if df is None:
        st.error("Could not load the dataset!")
        return
    
    st.write("First few rows of the data:")
    st.write(df.head())
        
    st.success("Data loaded successfully!")
    
    # Training section
    st.header("Model Training")
    if st.button("Train Model"):
        st.write("Train Model button clicked!")
        with st.spinner("Training in progress..."):
            try:
                st.write("Starting model training process...")
                model, test_data, feature_names = train_model(df)
                st.write("Model training completed, saving to session state...")
                
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = feature_names
                
                st.success("Model trained successfully!")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                st.write("Full error:", str(e))
                return

    # Always show visualizations if model exists
    if 'model' in st.session_state:
        # Prediction Interface first
        st.header("Make Predictions")
        
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
                st.success(f"Predicted Energy Demand: {prediction:,.0f} MW")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

        # Show visualizations after prediction section
        st.header("Model Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Predictions")
            pred_fig = plot_predictions(st.session_state['test_data'], st.session_state['model'])
            st.plotly_chart(pred_fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance")
            imp_fig = plot_feature_importance(st.session_state['model'], st.session_state['feature_names'])
            st.plotly_chart(imp_fig, use_container_width=True)

if __name__ == "__main__":
    main()