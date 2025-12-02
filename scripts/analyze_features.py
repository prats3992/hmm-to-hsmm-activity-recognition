import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def analyze_features():
    # Load data
    print("Loading dataset...")
    train_df = pd.read_csv('data/train.csv')
    
    # Separate features and targets
    # The last two columns are 'subject' and 'Activity'
    X = train_df.iloc[:, :-2].values
    y = train_df['Activity'].values
    
    print(f"Original feature shape: {X.shape}")
    
    # Standardize features (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for 90%, 95%, and 99% variance
    n_90 = np.argmax(cumsum >= 0.90) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    n_99 = np.argmax(cumsum >= 0.99) + 1
    
    print("\nPCA Analysis Results:")
    print(f"Components needed for 90% variance: {n_90}")
    print(f"Components needed for 95% variance: {n_95}")
    print(f"Components needed for 99% variance: {n_99}")
    
    return n_95

if __name__ == "__main__":
    analyze_features()
