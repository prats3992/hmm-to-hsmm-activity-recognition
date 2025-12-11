import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def get_durations(y):
    durations = {label: [] for label in np.unique(y)}
    current_label = y[0]
    current_len = 1
    
    for i in range(1, len(y)):
        if y[i] == current_label:
            current_len += 1
        else:
            durations[current_label].append(current_len)
            current_label = y[i]
            current_len = 1
    durations[current_label].append(current_len)
    return durations

def verify_duration_distributions():
    print("Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    y_train = train_df['Activity'].values
    
    # Map activity IDs to names if possible, but we'll use IDs for now
    # Assuming standard mapping: 1: WALKING, 2: WALKING_UPSTAIRS, 3: WALKING_DOWNSTAIRS, 
    # 4: SITTING, 5: STANDING, 6: LAYING
    # But let's just use the unique values found.
    
    durations = get_durations(y_train)
    results = []
    
    os.makedirs('outputs/images', exist_ok=True)
    plt.figure(figsize=(18, 12))
    
    unique_labels = sorted(durations.keys())
    
    for idx, label in enumerate(unique_labels):
        durs = np.array(durations[label])
        
        # 1. Fit Distributions
        # Normal
        mu, std = stats.norm.fit(durs)
        
        # Log-Normal (requires positive values)
        shape, loc, scale = stats.lognorm.fit(durs, floc=0)
        
        # Geometric (p = 1/mean)
        p_geom = 1.0 / np.mean(durs)
        
        # 2. Goodness of Fit (KS Test)
        # Compare empirical CDF with theoretical CDF
        
        # Normal KS
        ks_stat_norm, p_val_norm = stats.kstest(durs, 'norm', args=(mu, std))
        
        # Log-Normal KS
        ks_stat_log, p_val_log = stats.kstest(durs, 'lognorm', args=(shape, loc, scale))
        
        # Geometric KS (Discrete)
        # We approximate by checking against the geometric CDF
        # stats.geom.cdf(k, p)
        ks_stat_geom, p_val_geom = stats.kstest(durs, 'geom', args=(p_geom,))
        
        best_dist = "None"
        best_p = -1
        
        # Simple heuristic for "best" fit based on p-value (higher is better for null hypothesis "data follows dist")
        # But with large N, p-value often -> 0. We can also look at KS statistic (lower is better).
        
        fits = {
            'Normal': (ks_stat_norm, p_val_norm),
            'LogNorm': (ks_stat_log, p_val_log),
            'Geometric': (ks_stat_geom, p_val_geom)
        }
        
        best_fit_name = min(fits, key=lambda k: fits[k][0]) # Min KS stat
        
        results.append({
            'Activity': label,
            'Count': len(durs),
            'Mean_Dur': np.mean(durs),
            'Best_Fit': best_fit_name,
            'Normal_KS': ks_stat_norm,
            'LogNorm_KS': ks_stat_log,
            'Geom_KS': ks_stat_geom
        })
        
        # 3. Plotting
        plt.subplot(2, 3, idx+1)
        
        # Histogram
        sns.histplot(durs, stat='density', alpha=0.4, label='Data')
        
        # X range for plotting
        x = np.linspace(min(durs), max(durs), 100)
        
        # Plot Normal
        plt.plot(x, stats.norm.pdf(x, mu, std), 'r-', label=f'Normal (KS={ks_stat_norm:.2f})')
        
        # Plot Log-Normal
        plt.plot(x, stats.lognorm.pdf(x, shape, loc, scale), 'g--', label=f'LogNorm (KS={ks_stat_log:.2f})')
        
        # Plot Geometric (Discrete)
        # x_int = np.arange(min(durs), max(durs))
        # plt.plot(x_int, stats.geom.pmf(x_int, p_geom), 'b:', label=f'Geom (KS={ks_stat_geom:.2f})')
        # Geometric decays very fast, might not be visible well on same scale if mean is large
        
        plt.title(f"Activity {label} Duration\nBest Fit: {best_fit_name}")
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('outputs/images/hsmm_duration_assumptions.png')
    print("Saved duration distribution plots to outputs/images/hsmm_duration_assumptions.png")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/results/hsmm_assumption_tests.csv', index=False)
    print("\nAssumption Test Results (KS Statistic - Lower is Better):")
    print(results_df[['Activity', 'Best_Fit', 'Normal_KS', 'LogNorm_KS', 'Geom_KS']])

if __name__ == "__main__":
    verify_duration_distributions()
