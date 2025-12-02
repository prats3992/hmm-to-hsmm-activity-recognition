import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os

def load_data(n_components=65):
    train_df = pd.read_csv('data/train.csv')
    train_df = train_df.sort_values(by=['subject'])
    
    X_raw = train_df.iloc[:, :-2].values
    y = train_df['Activity'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, y_enc, le.classes_

def test_downsampling_effects(X, y, classes, output_dir):
    print("\n--- Testing Downsampling Effects ---")
    
    # Factors to test
    factors = [1, 5, 10, 20]
    
    results = []
    
    target_class = 'WALKING'
    cls_idx = np.where(classes == target_class)[0][0]
    
    # Get continuous segment
    X_cls = X[y == cls_idx, 0]
    
    plt.figure(figsize=(12, 8))
    
    for i, factor in enumerate(factors):
        # Downsample
        X_sub = X_cls[::factor]
        
        # 1. Check Autocorrelation (Independence)
        residuals = X_sub - np.mean(X_sub)
        acf = sm.tsa.acf(residuals, nlags=10)
        lag1_acf = acf[1]
        
        print(f"Downsample Factor {factor}x: Lag-1 Autocorrelation = {lag1_acf:.4f}")
        
        # Plot ACF
        plt.subplot(2, 2, i+1)
        plt.bar(range(len(acf)), acf)
        plt.axhline(0.05, color='r', linestyle='--')
        plt.title(f'Factor {factor}x (Lag-1 ACF: {lag1_acf:.2f})')
        plt.ylim(-0.2, 1.0)
        
        results.append({'Factor': factor, 'Lag1_ACF': lag1_acf})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'downsampling_acf.png'))
    print(f"Plot saved to {output_dir}/downsampling_acf.png")
    
    return results

def main():
    output_dir = 'outputs/assumptions_downsample'
    os.makedirs(output_dir, exist_ok=True)
    
    X, y, classes = load_data()
    test_downsampling_effects(X, y, classes, output_dir)

if __name__ == "__main__":
    main()
