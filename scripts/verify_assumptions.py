import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(n_components=65):
    print("Loading and preprocessing data...")
    train_df = pd.read_csv('data/train.csv')
    
    # Sort by subject to keep temporal order
    train_df = train_df.sort_values(by=['subject'])
    
    X_raw = train_df.iloc[:, :-2].values
    y = train_df['Activity'].values
    subjects = train_df['subject'].values
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, y_enc, subjects, le.classes_

def check_gaussianity(X, y, classes, output_dir):
    print("\n--- Checking Gaussianity Assumption ---")
    # Assumption: P(x | State=j) ~ Gaussian
    # We check the distribution of the first few Principal Components for each class
    
    n_components_to_check = 3
    results = []
    
    fig, axes = plt.subplots(len(classes), n_components_to_check, figsize=(15, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, cls_idx in enumerate(range(len(classes))):
        cls_name = classes[cls_idx]
        X_cls = X[y == cls_idx]
        
        for j in range(n_components_to_check):
            feature_data = X_cls[:, j]
            
            # Shapiro-Wilk Test
            # Note: Shapiro-Wilk is sensitive to large sample sizes (will reject even small deviations)
            # We take a random subsample if size > 5000 to be safe, though here N ~ 1000 per class
            if len(feature_data) > 5000:
                feature_data_sub = np.random.choice(feature_data, 5000, replace=False)
                stat, p = stats.shapiro(feature_data_sub)
            else:
                stat, p = stats.shapiro(feature_data)
            
            is_gaussian = p > 0.05
            results.append({
                'Activity': cls_name,
                'PC': j+1,
                'Shapiro-Stat': stat,
                'p-value': p,
                'Gaussian': is_gaussian
            })
            
            # Q-Q Plot
            ax = axes[i, j]
            stats.probplot(feature_data, dist="norm", plot=ax)
            ax.set_title(f"{cls_name} - PC{j+1}\np={p:.2e}")
            
    plt.suptitle("Q-Q Plots for Top 3 PCA Components per Activity", fontsize=16)
    plt.savefig(os.path.join(output_dir, 'gaussianity_qq_plots.png'))
    print(f"Q-Q plots saved to {output_dir}/gaussianity_qq_plots.png")
    
    # Save results to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, 'gaussianity_test_results.csv'), index=False)
    print("Gaussianity test statistics saved.")
    return res_df

def check_stationarity(X, y, subjects, classes, output_dir):
    print("\n--- Checking Stationarity Assumption ---")
    # Assumption: Statistical properties (mean, variance) are constant within an activity state.
    # We look at long continuous segments of a single activity.
    
    results = []
    
    # We'll pick the longest continuous segment for each activity from a random subject
    for cls_idx, cls_name in enumerate(classes):
        # Find all indices for this class
        cls_indices = np.where(y == cls_idx)[0]
        
        if len(cls_indices) == 0:
            continue
            
        # Find continuous segments
        # Since we sorted by subject, we can just look for breaks in indices
        breaks = np.where(np.diff(cls_indices) != 1)[0] + 1
        segments = np.split(cls_indices, breaks)
        
        # Find longest segment
        longest_seg = max(segments, key=len)
        
        if len(longest_seg) < 20:
            print(f"Skipping {cls_name}: Longest segment too short ({len(longest_seg)})")
            continue
            
        # Test the first Principal Component
        ts_data = X[longest_seg, 0]
        
        # Augmented Dickey-Fuller Test
        # H0: Non-stationary (has unit root)
        # H1: Stationary
        try:
            adf_result = adfuller(ts_data)
            stat = adf_result[0]
            p = adf_result[1]
            is_stationary = p < 0.05
            
            results.append({
                'Activity': cls_name,
                'Segment_Length': len(longest_seg),
                'ADF_Stat': stat,
                'p-value': p,
                'Stationary': is_stationary
            })
        except Exception as e:
            print(f"ADF test failed for {cls_name}: {e}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, 'stationarity_test_results.csv'), index=False)
    print("Stationarity test results saved.")
    return res_df

def check_linearity(X, y, classes, output_dir):
    print("\n--- Checking Linearity (Residual Analysis) ---")
    # Model: x = mu_j + epsilon
    # We calculate residuals: epsilon = x - mean(x_class)
    # And plot Residuals vs Fitted Values (Mean)
    
    # We'll do this for the first PC
    pc_idx = 0
    
    means = []
    residuals = []
    
    for cls_idx in range(len(classes)):
        X_cls = X[y == cls_idx, pc_idx]
        mu = np.mean(X_cls)
        res = X_cls - mu
        
        means.extend([mu] * len(X_cls))
        residuals.extend(res)
        
    plt.figure(figsize=(10, 6))
    plt.scatter(means, residuals, alpha=0.1)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Fitted Values (Class Means)')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values (PC1)')
    plt.savefig(os.path.join(output_dir, 'linearity_residuals.png'))
    print(f"Residual plot saved to {output_dir}/linearity_residuals.png")

def main():
    output_dir = 'outputs/assumptions'
    os.makedirs(output_dir, exist_ok=True)
    
    X, y, subjects, classes = load_data()
    
    # 1. Gaussianity
    gauss_res = check_gaussianity(X, y, classes, output_dir)
    print("\nGaussianity Summary (First 3 PCs):")
    print(gauss_res.groupby('Activity')['Gaussian'].mean())
    
    # 2. Stationarity
    stat_res = check_stationarity(X, y, subjects, classes, output_dir)
    print("\nStationarity Summary:")
    print(stat_res[['Activity', 'p-value', 'Stationary']])
    
    # 3. Linearity
    check_linearity(X, y, classes, output_dir)

if __name__ == "__main__":
    main()
