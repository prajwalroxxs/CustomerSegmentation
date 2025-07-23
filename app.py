import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Customer Segmentation", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #FFFFFF;
        }
        .stApp {
            background-color: #121212;
        }
        h1, h2, h3 {
            color: #00C4FF;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Customer Segmentation Dashboard")
st.markdown("Segment your customers intelligently using K-Means Clustering.")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    original_data = data.copy()

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        data.drop('CustomerID', axis=1, inplace=True)
    except:
        pass

    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    scaled_data = scaler.transform(data)
    data['Cluster'] = model.predict(scaled_data)
    original_data['Cluster'] = data['Cluster']

    st.subheader("📌 Clustered Data")
    st.dataframe(original_data)

    st.subheader("📈 Cluster Distribution")
    cluster_counts = data['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    st.subheader("🌀 Visualize Clusters")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data.iloc[:, 2], y=data.iloc[:, 3], hue=data['Cluster'], palette='tab10', ax=ax)
    ax.set_xlabel(data.columns[2])
    ax.set_ylabel(data.columns[3])
    ax.set_title("Customer Segmentation View")
    st.pyplot(fig)
else:
    st.info("Upload a `.csv` file to get started.")

st.markdown("---")
st.markdown("<center>Built by Prajwal • Celebal Internship</center>", unsafe_allow_html=True)
