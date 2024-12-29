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

# Custom CSS for white theme
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
    }
    .stProgress .st-bo {
        background-color: #4addbe;
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

def time_features(df):
    df = df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["hour"] = df.index.hour
    return df

def train_model(df):
    df = time_features(df)
    length = df.shape[0]
    main = int(length * 0.8)
    trainer = df[:main]
    tester = df[main:]
    
    X = tester.iloc[:,1:]
    y = tester.PJME_MW
    X_train, X_val, y_train, y_val = tts(X, y, train_size=0.8, random_state=42)
    
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train, 
             eval_set=[(X_train, y_train), (X_val, y_val)],
             verbose=False)
    
    return model, tester, X.columns.tolist()

def plot_training_metrics(eval_results):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(eval_results["validation_0"]["rmse"]))),
        y=eval_results["validation_0"]["rmse"],
        name="Training",
        line=dict(color="#4addbe")
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(eval_results["validation_1"]["rmse"]))),
        y=eval_results["validation_1"]["rmse"],
        name="Validation",
        line=dict(color="#34495e")
    ))
    
    fig.update_layout(
        title="Training Metrics",
        xaxis_title="Rounds",
        yaxis_title="RMSE",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#2c3e50")
    )
    
    return fig

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
        title="Energy Demand - Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Energy Demand (MW)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#2c3e50")
    )
    
    return fig

def main():
    st.title("PJME Energy Demand Predictor")
    
    df = load_data()
    if df is None:
        st.error("Could not load the dataset!")
        return
        
    st.success("Data loaded successfully!")
    
    # Training section
    st.header("Model Training")
    if st.button("Train Model"):
        with st.spinner("Training in progress... Oy, this could take a minute, bubbeleh!"):
            try:
                model, test_data, feature_names = train_model(df)
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = feature_names
                
                eval_results = model.evals_result()
                metrics_fig = plot_training_metrics(eval_results)
                st.plotly_chart(metrics_fig, use_container_width=True)
                
                st.success("Such a beautiful model we've trained! Mazel tov!")
            except Exception as e:
                st.error(f"Oy vey, we got an error: {str(e)}")
                return

    # Prediction Interface
    if 'model' in st.session_state:
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
                st.error(f"Oy gevalt! Error during prediction: {str(e)}")
        
        # Show visualizations
        st.header("Model Analysis")
        
        pred_fig = plot_predictions(
            st.session_state['test_data'],
            st.session_state['model']
        )
        st.plotly_chart(pred_fig, use_container_width=True)
        
        # Feature importance plot
        importances = pd.DataFrame({
            'feature': st.session_state['feature_names'],
            'importance': st.session_state['model'].feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(importances, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title="Feature Importance")
        
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#2c3e50")
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()