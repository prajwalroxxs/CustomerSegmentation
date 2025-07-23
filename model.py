import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(path="Mall_Customers.csv"):
    return pd.read_csv(path)

def preprocess_data(df, selected_columns):
    df_selected = df[selected_columns].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_selected)
    return scaled_data, df_selected

def train_model(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(data)
    return model, clusters
