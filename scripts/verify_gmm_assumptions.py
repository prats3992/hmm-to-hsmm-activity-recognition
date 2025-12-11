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
    # Using Likelihood Ratio Test (Chi-Squared)
    
    n_states = len(classes)
    
    # 1. Build Transition Counts
    # First Order: Count(t-1, t)
    trans_1 = np.zeros((n_states, n_states))
    # Second Order: Count(t-2, t-1, t)
    trans_2 = np.zeros((n_states, n_states, n_states))
    
    # Map labels to 0..N-1 just in case
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    for t in range(len(y_enc) - 1):
        i = y_enc[t]
        j = y_enc[t+1]
        trans_1[i, j] += 1
        
        if t < len(y_enc) - 2:
            k = y_enc[t+2]
            trans_2[i, j, k] += 1
            
    # 2. Calculate Log-Likelihoods
    epsilon = 1e-10
    
    # Order 1
    row_sums_1 = trans_1.sum(axis=1, keepdims=True)
    probs_1 = (trans_1 + epsilon) / (row_sums_1 + epsilon * n_states)
    
    log_L1 = 0
    for t in range(len(y_enc) - 1):
        i = y_enc[t]
        j = y_enc[t+1]
        log_L1 += np.log(probs_1[i, j])
        
    # Order 2
    probs_2 = np.zeros_like(trans_2)
    for i in range(n_states):
        for j in range(n_states):
            total = trans_2[i, j].sum()
            if total > 0:
                probs_2[i, j, :] = (trans_2[i, j, :] + epsilon) / (total + epsilon * n_states)
            else:
                probs_2[i, j, :] = 1.0 / n_states
                
    log_L2 = 0
    for t in range(len(y_enc) - 2):
        i = y_enc[t]
        j = y_enc[t+1]
        k = y_enc[t+2]
        log_L2 += np.log(probs_2[i, j, k])
        
    # 3. Likelihood Ratio Test
    # Statistic = 2 * (LogL_Complex - LogL_Simple)
    LR_stat = 2 * (log_L2 - log_L1)
    
    # Degrees of Freedom: N(N-1)^2
    df = n_states * (n_states - 1)**2
    
    p_value = stats.chi2.sf(LR_stat, df)
    
    print(f"Log-Likelihood (1st Order): {log_L1:.2f}")
    print(f"Log-Likelihood (2nd Order): {log_L2:.2f}")
    print(f"LR Statistic: {LR_stat:.2f}")
    print(f"Degrees of Freedom: {df}")
    print(f"P-Value: {p_value:.4e}")
    
    if p_value < 0.05:
        print(">> Result: REJECT Null Hypothesis.")
        print(">> The sequence is NOT First-Order Markov (2nd order is significantly better).")
    else:
        print(">> Result: ACCEPT Null Hypothesis.")
        print(">> The sequence IS First-Order Markov (2nd order adds no significant info).")

def main():
    output_dir = 'outputs/assumptions_gmm'
    os.makedirs(output_dir, exist_ok=True)
    
    X, y, classes = load_data()
    
    verify_gmm_fit(X, y, classes, output_dir)
    verify_conditional_independence(X, y, classes, output_dir)
    verify_markov_property(y, classes)

if __name__ == "__main__":
    main()
