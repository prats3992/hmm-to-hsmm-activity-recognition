import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def justify_pca():
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    
    # Drop non-feature columns
    X_raw = train_df.drop(['subject', 'Activity'], axis=1)
    feature_names = X_raw.columns
    
    output_dir = 'outputs/images/eda'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Visualizing Redundancy (Correlation Matrix of a subset)
    # We'll take the first 50 features which likely contain related accelerometer stats
    subset_size = 50
    print(f"\nComputing correlation matrix for first {subset_size} features...")
    corr_matrix = X_raw.iloc[:, :subset_size].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix (First {subset_size} Features)\nRed blocks indicate high redundancy')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_justification_correlation.png')
    print(f"Saved {output_dir}/pca_justification_correlation.png")
    
    # 2. PCA Explained Variance Analysis
    print("\nStandardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print("Fitting PCA...")
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find components for 90% and 95% variance
    n_90 = np.argmax(cumulative_variance >= 0.90) + 1
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    print(f"Components for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    print(f"Total features: {X_raw.shape[1]}")
    
    # Plot Cumulative Explained Variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, linewidth=2)
    plt.axhline(y=0.90, color='r', linestyle='--', label=f'90% Variance ({n_90} components)')
    plt.axhline(y=0.95, color='g', linestyle='--', label=f'95% Variance ({n_95} components)')
    plt.axvline(x=n_90, color='r', linestyle=':', alpha=0.5)
    plt.axvline(x=n_95, color='g', linestyle=':', alpha=0.5)
    
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Analysis\nWhy we can reduce 561 features to ~65')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_explained_variance.png')
    print(f"Saved {output_dir}/pca_explained_variance.png")

if __name__ == "__main__":
    justify_pca()
