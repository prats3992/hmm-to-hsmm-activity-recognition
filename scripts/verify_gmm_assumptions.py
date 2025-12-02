import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import os

def load_data(n_components=65):
    print("Loading data...")
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

def verify_gmm_fit(X, y, classes, output_dir):
    print("\n--- Test 1: GMM vs Gaussian Goodness of Fit (All Classes) ---")
    
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    results = []
    
    for i, cls_name in enumerate(classes):
        cls_idx = i
        X_cls = X[y == cls_idx, 0].reshape(-1, 1) # Check 1st PC
        
        # Fit Single Gaussian
        g1 = GaussianMixture(n_components=1, random_state=42)
        g1.fit(X_cls)
        aic1 = g1.aic(X_cls)
        
        # Fit GMM (2 components)
        g2 = GaussianMixture(n_components=2, random_state=42)
        g2.fit(X_cls)
        aic2 = g2.aic(X_cls)
        
        improvement = aic1 - aic2
        results.append({'Activity': cls_name, 'AIC_Gauss': aic1, 'AIC_GMM': aic2, 'Improvement': improvement})
        
        # Plot
        ax = plt.subplot(2, 3, i+1)
        sns.histplot(X_cls, stat='density', bins=50, alpha=0.4, label='Data', ax=ax)
        
        x_axis = np.linspace(X_cls.min(), X_cls.max(), 1000).reshape(-1, 1)
        ax.plot(x_axis, np.exp(g1.score_samples(x_axis)), 'r--', label='Single', linewidth=2)
        ax.plot(x_axis, np.exp(g2.score_samples(x_axis)), 'b-', label='GMM(2)', linewidth=2)
        
        ax.set_title(f'{cls_name}\nAIC Imp: {improvement:.0f}')
        if i == 0:
            ax.legend()
            
    plt.suptitle("Goodness of Fit: Single Gaussian vs GMM (2 Components)", fontsize=16)
    plt.savefig(os.path.join(output_dir, 'gmm_vs_gaussian_fit_all.png'))
    print(f"Fit plots saved to {output_dir}/gmm_vs_gaussian_fit_all.png")
    
    print("\nAIC Improvement (Higher is better for GMM):")
    for res in results:
        print(f"{res['Activity']:<20}: {res['Improvement']:.2f}")

def verify_conditional_independence(X, y, classes, output_dir):
    print("\n--- Test 2: Conditional Independence (Autocorrelation) ---")
    # HMM assumes observations are independent given state.
    # We check autocorrelation of residuals within a class.
    
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, cls_name in enumerate(classes):
        cls_idx = i
        
        # Get a continuous segment (simplification: taking all samples)
        X_cls = X[y == cls_idx, 0] # PC1
        
        if len(X_cls) == 0:
            continue
            
        # Residuals = Data - Mean
        residuals = X_cls - np.mean(X_cls)
        
        # Calculate ACF
        acf = sm.tsa.acf(residuals, nlags=20)
        
        ax = plt.subplot(2, 3, i+1)
        ax.bar(range(len(acf)), acf)
        ax.axhline(0.05, color='r', linestyle='--')
        ax.axhline(-0.05, color='r', linestyle='--')
        ax.set_title(f'{cls_name}\nLag-1 ACF: {acf[1]:.2f}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_ylim(-0.2, 1.0)
        
        print(f"Activity: {cls_name}, Lag-1 ACF: {acf[1]:.4f}")

    plt.suptitle("Autocorrelation of Residuals (Conditional Independence Check)", fontsize=16)
    plt.savefig(os.path.join(output_dir, 'residual_acf_all_classes.png'))
    print(f"ACF plots saved to {output_dir}/residual_acf_all_classes.png")

def verify_markov_property(y, classes):
    print("\n--- Test 3: Markov Property (1st vs 2nd Order) ---")
    # We test if P(St | St-1, St-2) provides more info than P(St | St-1)
    # Using Likelihood Ratio Test
    
    # 1. Build Transition Counts
    n_states = len(classes)
    
    # First Order: Count(t-1, t)
    trans_1 = np.zeros((n_states, n_states))
    # Second Order: Count(t-2, t-1, t)
    trans_2 = np.zeros((n_states, n_states, n_states))
    
    for t in range(2, len(y)):
        s_prev2 = y[t-2]
        s_prev1 = y[t-1]
        s_curr = y[t]
        
        trans_1[s_prev1, s_curr] += 1
        trans_2[s_prev2, s_prev1, s_curr] += 1
        
    # Calculate Log Likelihoods
    # L1 = sum log P(st | st-1)
    # L2 = sum log P(st | st-1, st-2)
    
    log_l1 = 0
    log_l2 = 0
    
    # Avoid log(0)
    epsilon = 1e-10
    
    # 1st Order Probabilities
    row_sums_1 = trans_1.sum(axis=1, keepdims=True) + epsilon
    probs_1 = (trans_1 + epsilon) / row_sums_1
    
    # 2nd Order Probabilities
    row_sums_2 = trans_2.sum(axis=2, keepdims=True) + epsilon
    probs_2 = (trans_2 + epsilon) / row_sums_2
    
    # Compute Likelihoods over the data
    for t in range(2, len(y)):
        s_prev2 = y[t-2]
        s_prev1 = y[t-1]
        s_curr = y[t]
        
        log_l1 += np.log(probs_1[s_prev1, s_curr])
        log_l2 += np.log(probs_2[s_prev2, s_prev1, s_curr])
        
    print(f"Log-Likelihood (1st Order): {log_l1:.2f}")
    print(f"Log-Likelihood (2nd Order): {log_l2:.2f}")
    
    # Likelihood Ratio Statistic
    # D = -2 * (L1 - L2)
    # Degrees of Freedom difference: N^3 - N^2 (roughly, accounting for zeros is complex)
    # We'll just look at the magnitude improvement
    
    improvement = log_l2 - log_l1
    print(f"Likelihood Improvement: {improvement:.2f}")
    
    if improvement > 100: # Arbitrary large threshold for significance given N=7000
        print(">> 2nd Order model is significantly better. Markov assumption (1st order) is a simplification.")
    else:
        print(">> 1st Order model is sufficient.")

def main():
    output_dir = 'outputs/assumptions_gmm'
    os.makedirs(output_dir, exist_ok=True)
    
    X, y, classes = load_data()
    
    verify_gmm_fit(X, y, classes, output_dir)
    verify_conditional_independence(X, y, classes, output_dir)
    verify_markov_property(y, classes)

if __name__ == "__main__":
    main()
