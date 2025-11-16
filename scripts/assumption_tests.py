"""
Statistical Tests for HMM Assumption Verification (Section 4.3)

Implements tests for:
1. Linearity of observation model
2. Stationarity of observations
3. Gaussianity of emission distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AssumptionVerifier:
    """
    Comprehensive assumption testing for HMM models
    
    Tests three key assumptions:
    1. Linearity: Observations are linear functions of states + Gaussian noise
    2. Stationarity: Statistical properties don't change over time
    3. Gaussianity: Emission distributions are approximately Gaussian
    """
    
    def __init__(self, output_dir: str = 'outputs/results'):
        self.output_dir = output_dir
        self.test_results = {}
        
    # ========== LINEARITY TESTS ==========
    
    def test_linearity_residuals(self, X: np.ndarray, y: np.ndarray, 
                                 activity_names: List[str], 
                                 feature_names: List[str],
                                 n_features_plot: int = 10) -> Dict:
        """
        Test 1: Residual Analysis for Linearity
        
        For each activity class, compute residuals:
        ε_ti = x_ti - μ_j  for all samples with S_t = j
        
        Plot residuals vs fitted values. Linear model is appropriate if:
        - Residuals scatter randomly around zero
        - No systematic patterns (U-shape, funnel, clustering)
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Activity labels (0 to n_classes-1)
        activity_names : List[str]
            Names of activity classes
        feature_names : List[str]
            Names of features
        n_features_plot : int
            Number of features to plot
            
        Returns:
        --------
        results : Dict
            R² values and residual statistics per activity
        """
        print("\n" + "="*80)
        print("TEST 1: RESIDUAL ANALYSIS FOR LINEARITY")
        print("="*80)
        
        n_classes = len(activity_names)
        n_features = X.shape[1]
        
        results = {
            'r_squared': {},
            'residual_std': {},
            'residual_skewness': {},
            'residual_kurtosis': {}
        }
        
        # Create figure for residual plots
        fig, axes = plt.subplots(n_classes, 2, figsize=(14, 3*n_classes))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for activity_idx in range(n_classes):
            activity = activity_names[activity_idx]
            mask = (y == activity_idx)
            X_activity = X[mask]
            
            if len(X_activity) == 0:
                continue
            
            # Compute class mean (fitted value under linear model)
            mu = X_activity.mean(axis=0)
            
            # Compute residuals
            residuals = X_activity - mu
            
            # Compute R² (coefficient of determination)
            ss_total = np.sum((X_activity - X.mean(axis=0))**2)
            ss_residual = np.sum(residuals**2)
            r_squared = 1 - (ss_residual / (ss_total + 1e-10))
            
            results['r_squared'][activity] = r_squared
            results['residual_std'][activity] = np.std(residuals, axis=0).mean()
            results['residual_skewness'][activity] = stats.skew(residuals, axis=0).mean()
            results['residual_kurtosis'][activity] = stats.kurtosis(residuals, axis=0).mean()
            
            # Plot 1: Residuals vs Fitted (fitted = mu repeated for each sample)
            ax1 = axes[activity_idx, 0]
            fitted_values = np.repeat(mu.reshape(1, -1), X_activity.shape[0], axis=0)
            for feat_idx in range(min(n_features_plot, n_features)):
                ax1.scatter(fitted_values[:, feat_idx], residuals[:, feat_idx], alpha=0.3, s=10)
            ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax1.set_xlabel('Fitted Value (Class Mean)')
            ax1.set_ylabel('Residuals')
            ax1.set_title(f'{activity}: Residuals vs Fitted\nR² = {r_squared:.3f}')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Histogram of residuals (should be centered at 0)
            ax2 = axes[activity_idx, 1]
            residual_flat = residuals.flatten()
            ax2.hist(residual_flat, bins=50, density=True, alpha=0.7, edgecolor='black')
            
            # Overlay Gaussian fit
            mu_res, std_res = residual_flat.mean(), residual_flat.std()
            x_range = np.linspace(residual_flat.min(), residual_flat.max(), 100)
            ax2.plot(x_range, stats.norm.pdf(x_range, mu_res, std_res), 
                    'r-', linewidth=2, label='Gaussian fit')
            ax2.axvline(x=0, color='k', linestyle='--', linewidth=2)
            ax2.set_xlabel('Residual Value')
            ax2.set_ylabel('Density')
            ax2.set_title(f'{activity}: Residual Distribution\nSkew={results["residual_skewness"][activity]:.3f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Print results
            print(f"\n{activity}:")
            print(f"  R² = {r_squared:.4f} ", end='')
            if r_squared > 0.7:
                print("✓ (Strong linear relationship)")
            elif r_squared > 0.4:
                print("⚠ (Moderate linear relationship)")
            else:
                print("✗ (Weak linear relationship - consider non-linear model)")
            
            print(f"  Residual std = {results['residual_std'][activity]:.4f}")
            print(f"  Residual skewness = {results['residual_skewness'][activity]:.4f}")
            print(f"  Residual kurtosis = {results['residual_kurtosis'][activity]:.4f}")
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/linearity_residual_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n→ Saved residual plots to {self.output_dir}/linearity_residual_analysis.png")
        plt.close()
        
        return results
    
    def test_linearity_reset(self, X: np.ndarray, y: np.ndarray, 
                            n_features_test: int = 20) -> Dict:
        """
        Test 2: Ramsey RESET Test for Linearity
        
        Tests whether adding powers of fitted values improves the model.
        H0: Linear model is adequate (γ₁ = γ₂ = 0)
        H1: Non-linear terms are significant
        
        Uses F-test to compare restricted vs unrestricted models.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Activity labels
        n_features_test : int
            Number of features to test (computational constraint)
            
        Returns:
        --------
        results : Dict
            F-statistics and p-values
        """
        print("\n" + "="*80)
        print("TEST 2: RAMSEY RESET TEST FOR LINEARITY")
        print("="*80)
        
        from sklearn.linear_model import LinearRegression
        from scipy.stats import f as f_dist
        
        n_samples, n_features = X.shape
        n_test = min(n_features_test, n_features)
        
        results = {
            'f_statistics': [],
            'p_values': [],
            'reject_linearity': 0
        }
        
        # Test subset of features
        for feat_idx in range(n_test):
            x = X[:, feat_idx].reshape(-1, 1)
            
            # Create one-hot encoding of activity labels
            y_onehot = np.eye(len(np.unique(y)))[y]
            
            # Fit restricted model (linear)
            model_restricted = LinearRegression()
            model_restricted.fit(y_onehot, x)
            y_pred = model_restricted.predict(y_onehot)
            rss_restricted = np.sum((x.flatten() - y_pred.flatten())**2)
            
            # Fit unrestricted model (with polynomial terms)
            y_pred_sq = y_pred ** 2
            y_pred_cube = y_pred ** 3
            X_unrestricted = np.hstack([y_onehot, y_pred_sq, y_pred_cube])
            
            model_unrestricted = LinearRegression()
            model_unrestricted.fit(X_unrestricted, x)
            y_pred_unres = model_unrestricted.predict(X_unrestricted)
            rss_unrestricted = np.sum((x.flatten() - y_pred_unres.flatten())**2)
            
            # Compute F-statistic
            q = 2  # Number of additional parameters (squared and cubed terms)
            k = X_unrestricted.shape[1]
            n = n_samples
            
            f_stat = ((rss_restricted - rss_unrestricted) / q) / (rss_unrestricted / (n - k))
            p_value = 1 - f_dist.cdf(f_stat, q, n - k)
            
            results['f_statistics'].append(f_stat)
            results['p_values'].append(p_value)
            
            if p_value < 0.05:
                results['reject_linearity'] += 1
        
        # Compute average statistics
        avg_f = np.mean(results['f_statistics'])
        avg_p = np.mean(results['p_values'])
        rejection_rate = results['reject_linearity'] / n_test
        
        print(f"\nTested {n_test} features:")
        print(f"  Average F-statistic = {avg_f:.4f}")
        print(f"  Average p-value = {avg_p:.4f}")
        print(f"  Rejection rate (p < 0.05) = {rejection_rate:.2%}")
        
        if rejection_rate < 0.1:
            print("  ✓ Linear model is adequate (few rejections)")
        elif rejection_rate < 0.3:
            print("  ⚠ Moderate evidence against linearity")
        else:
            print("  ✗ Strong evidence against linearity - consider non-linear transformations")
        
        return results
    
    # ========== STATIONARITY TESTS ==========
    
    def test_stationarity_adf(self, X: np.ndarray, y: np.ndarray,
                              activity_names: List[str],
                              n_features_test: int = 30) -> Dict:
        """
        Test 3: Augmented Dickey-Fuller Test for Stationarity
        
        For each feature within each activity:
        Δx_t = α + βt + γx_{t-1} + ε_t
        
        H0: γ = 0 (unit root, non-stationary)
        H1: γ < 0 (stationary)
        
        Reject H0 if p < 0.05 → feature is stationary
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Activity labels
        activity_names : List[str]
            Names of activities
        n_features_test : int
            Number of features to test per activity
            
        Returns:
        --------
        results : Dict
            ADF statistics and stationarity rates
        """
        print("\n" + "="*80)
        print("TEST 3: AUGMENTED DICKEY-FULLER TEST FOR STATIONARITY")
        print("="*80)
        
        n_classes = len(activity_names)
        n_features = X.shape[1]
        n_test = min(n_features_test, n_features)
        
        results = {
            'stationary_rates': {},
            'adf_statistics': {},
            'p_values': {}
        }
        
        for activity_idx in range(n_classes):
            activity = activity_names[activity_idx]
            mask = (y == activity_idx)
            X_activity = X[mask]
            
            if len(X_activity) < 50:  # Need sufficient samples for ADF
                continue
            
            stationary_count = 0
            adf_stats = []
            p_vals = []
            
            # Test subset of features
            for feat_idx in range(n_test):
                series = X_activity[:, feat_idx]
                
                try:
                    # Perform ADF test
                    adf_result = adfuller(series, autolag='AIC')
                    adf_stat = adf_result[0]
                    p_value = adf_result[1]
                    
                    adf_stats.append(adf_stat)
                    p_vals.append(p_value)
                    
                    # If p < 0.05, reject H0 (non-stationary) → series is stationary
                    if p_value < 0.05:
                        stationary_count += 1
                except:
                    continue
            
            if len(adf_stats) > 0:
                stationary_rate = stationary_count / len(adf_stats)
                results['stationary_rates'][activity] = stationary_rate
                results['adf_statistics'][activity] = np.mean(adf_stats)
                results['p_values'][activity] = np.mean(p_vals)
                
                print(f"\n{activity}:")
                print(f"  Tested {len(adf_stats)} features")
                print(f"  Stationary rate = {stationary_rate:.2%} ", end='')
                
                if stationary_rate > 0.8:
                    print("✓ (Most features stationary)")
                elif stationary_rate > 0.5:
                    print("⚠ (Mixed stationarity)")
                else:
                    print("✗ (Non-stationary - consider differencing)")
                
                print(f"  Average ADF statistic = {np.mean(adf_stats):.4f}")
                print(f"  Average p-value = {np.mean(p_vals):.4f}")
        
        return results
    
    def test_stationarity_acf(self, X: np.ndarray, y: np.ndarray,
                             activity_names: List[str],
                             n_lags: int = 20) -> Dict:
        """
        Test 4: Autocorrelation Function Analysis
        
        Compute ACF: ρ(h) = Cov(x_t, x_{t+h}) / Var(x_t)
        
        Stationary process: ACF decays exponentially to zero
        Non-stationary: ACF decays slowly or oscillates
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Activity labels
        activity_names : List[str]
            Names of activities
        n_lags : int
            Number of lags to compute
            
        Returns:
        --------
        results : Dict
            ACF values and decay rates
        """
        print("\n" + "="*80)
        print("TEST 4: AUTOCORRELATION FUNCTION (ACF) ANALYSIS")
        print("="*80)
        
        n_classes = len(activity_names)
        
        # Create plots
        fig, axes = plt.subplots(n_classes, 1, figsize=(12, 3*n_classes))
        if n_classes == 1:
            axes = [axes]
        
        results = {
            'acf_decay_rates': {},
            'significant_lags': {}
        }
        
        for activity_idx in range(n_classes):
            activity = activity_names[activity_idx]
            mask = (y == activity_idx)
            X_activity = X[mask]
            
            if len(X_activity) < 50:
                continue
            
            # Compute ACF for first feature (representative)
            series = X_activity[:, 0]
            acf_values = acf(series, nlags=n_lags, fft=True)
            
            # Compute decay rate (fit exponential)
            lags = np.arange(1, len(acf_values))
            # Avoid log of negative numbers
            valid_acf = acf_values[1:]
            valid_mask = valid_acf > 0
            if valid_mask.sum() > 2:
                log_acf = np.log(valid_acf[valid_mask] + 1e-10)
                decay_rate = -np.polyfit(lags[valid_mask], log_acf, 1)[0]
            else:
                decay_rate = 0
            
            # Count significant lags (outside confidence interval)
            confidence_interval = 1.96 / np.sqrt(len(series))
            significant = np.sum(np.abs(acf_values[1:]) > confidence_interval)
            
            results['acf_decay_rates'][activity] = decay_rate
            results['significant_lags'][activity] = significant
            
            # Plot ACF
            ax = axes[activity_idx]
            ax.bar(range(len(acf_values)), acf_values, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=1, 
                      label='95% CI')
            ax.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            ax.set_title(f'{activity}: ACF (Decay rate = {decay_rate:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            print(f"\n{activity}:")
            print(f"  ACF decay rate = {decay_rate:.4f} ", end='')
            if decay_rate > 0.1:
                print("✓ (Fast decay, stationary)")
            elif decay_rate > 0.05:
                print("⚠ (Moderate decay)")
            else:
                print("✗ (Slow decay, possibly non-stationary)")
            print(f"  Significant lags (outside 95% CI) = {significant}/{n_lags}")
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/stationarity_acf_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n→ Saved ACF plots to {self.output_dir}/stationarity_acf_analysis.png")
        plt.close()
        
        return results
    
    # ========== GAUSSIANITY TESTS ==========
    
    def test_gaussianity_shapiro(self, X: np.ndarray, y: np.ndarray,
                                 activity_names: List[str],
                                 n_features_test: int = 30) -> Dict:
        """
        Test 5: Shapiro-Wilk Test for Univariate Normality
        
        For each feature in each activity:
        H0: Feature is normally distributed
        H1: Feature is not normally distributed
        
        Reject H0 if p < 0.05 → feature is non-Gaussian
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Activity labels
        activity_names : List[str]
            Names of activities
        n_features_test : int
            Number of features to test
            
        Returns:
        --------
        results : Dict
            Gaussianity rates per activity
        """
        print("\n" + "="*80)
        print("TEST 5: SHAPIRO-WILK TEST FOR GAUSSIANITY")
        print("="*80)
        
        n_classes = len(activity_names)
        n_features = X.shape[1]
        n_test = min(n_features_test, n_features)
        
        results = {
            'gaussian_rates': {},
            'shapiro_statistics': {},
            'p_values': {}
        }
        
        for activity_idx in range(n_classes):
            activity = activity_names[activity_idx]
            mask = (y == activity_idx)
            X_activity = X[mask]
            
            if len(X_activity) < 20:  # Need sufficient samples
                continue
            
            # Sample for computational efficiency (Shapiro-Wilk max 5000 samples)
            if len(X_activity) > 5000:
                sample_idx = np.random.choice(len(X_activity), 5000, replace=False)
                X_sample = X_activity[sample_idx]
            else:
                X_sample = X_activity
            
            gaussian_count = 0
            w_stats = []
            p_vals = []
            
            for feat_idx in range(n_test):
                series = X_sample[:, feat_idx]
                
                try:
                    # Shapiro-Wilk test
                    w_stat, p_value = shapiro(series)
                    w_stats.append(w_stat)
                    p_vals.append(p_value)
                    
                    # If p > 0.05, fail to reject H0 → series is Gaussian
                    if p_value > 0.05:
                        gaussian_count += 1
                except:
                    continue
            
            if len(w_stats) > 0:
                gaussian_rate = gaussian_count / len(w_stats)
                results['gaussian_rates'][activity] = gaussian_rate
                results['shapiro_statistics'][activity] = np.mean(w_stats)
                results['p_values'][activity] = np.mean(p_vals)
                
                print(f"\n{activity}:")
                print(f"  Tested {len(w_stats)} features")
                print(f"  Gaussian rate = {gaussian_rate:.2%} ", end='')
                
                if gaussian_rate > 0.7:
                    print("✓ (Most features Gaussian)")
                elif gaussian_rate > 0.4:
                    print("⚠ (Moderate Gaussianity)")
                else:
                    print("✗ (Non-Gaussian - consider GMM or transformations)")
                
                print(f"  Average W-statistic = {np.mean(w_stats):.4f}")
                print(f"  Average p-value = {np.mean(p_vals):.4f}")
        
        return results
    
    def test_gaussianity_qq(self, X: np.ndarray, y: np.ndarray,
                           activity_names: List[str],
                           n_features_plot: int = 6) -> Dict:
        """
        Test 6: Q-Q Plots for Visual Gaussianity Check
        
        Plot sample quantiles vs theoretical Gaussian quantiles.
        Points should lie on straight line if data is Gaussian.
        
        Deviations indicate:
        - S-curve: Heavy tails
        - Curved tails: Skewness
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Activity labels
        activity_names : List[str]
            Names of activities
        n_features_plot : int
            Number of features to plot per activity
            
        Returns:
        --------
        results : Dict
            Visual assessment results
        """
        print("\n" + "="*80)
        print("TEST 6: Q-Q PLOTS FOR GAUSSIANITY")
        print("="*80)
        
        n_classes = len(activity_names)
        
        # Create figure
        fig, axes = plt.subplots(n_classes, n_features_plot, 
                                figsize=(3*n_features_plot, 3*n_classes))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        results = {'activities': activity_names}
        
        for activity_idx in range(n_classes):
            activity = activity_names[activity_idx]
            mask = (y == activity_idx)
            X_activity = X[mask]
            
            if len(X_activity) < 20:
                continue
            
            # Sample for plotting
            if len(X_activity) > 1000:
                sample_idx = np.random.choice(len(X_activity), 1000, replace=False)
                X_sample = X_activity[sample_idx]
            else:
                X_sample = X_activity
            
            for feat_idx in range(n_features_plot):
                ax = axes[activity_idx, feat_idx]
                series = X_sample[:, feat_idx]
                
                # Create Q-Q plot
                stats.probplot(series, dist="norm", plot=ax)
                ax.set_title(f'{activity}\nFeature {feat_idx}', fontsize=9)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gaussianity_qq_plots.png', dpi=300, bbox_inches='tight')
        print(f"\n→ Saved Q-Q plots to {self.output_dir}/gaussianity_qq_plots.png")
        print("  Visual inspection: Points should lie on red line if Gaussian")
        plt.close()
        
        return results
    
    def run_all_tests(self, X: np.ndarray, y: np.ndarray,
                     activity_names: List[str],
                     feature_names: List[str]) -> Dict:
        """
        Run all assumption tests and generate comprehensive report
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Activity labels
        activity_names : List[str]
            Names of activities
        feature_names : List[str]
            Names of features
            
        Returns:
        --------
        all_results : Dict
            Complete test results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE HMM ASSUMPTION VERIFICATION")
        print("="*80)
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(activity_names)} activities")
        
        all_results = {}
        
        # Linearity tests
        all_results['linearity_residuals'] = self.test_linearity_residuals(
            X, y, activity_names, feature_names, n_features_plot=10)
        
        all_results['linearity_reset'] = self.test_linearity_reset(
            X, y, n_features_test=20)
        
        # Stationarity tests
        all_results['stationarity_adf'] = self.test_stationarity_adf(
            X, y, activity_names, n_features_test=30)
        
        all_results['stationarity_acf'] = self.test_stationarity_acf(
            X, y, activity_names, n_lags=20)
        
        # Gaussianity tests
        all_results['gaussianity_shapiro'] = self.test_gaussianity_shapiro(
            X, y, activity_names, n_features_test=30)
        
        all_results['gaussianity_qq'] = self.test_gaussianity_qq(
            X, y, activity_names, n_features_plot=6)
        
        # Save comprehensive summary
        self._save_summary_report(all_results, activity_names)
        
        return all_results
    
    def _save_summary_report(self, results: Dict, activity_names: List[str]):
        """Generate and save comprehensive summary report"""
        
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        # Linearity summary
        print("\n1. LINEARITY ASSUMPTION:")
        r_squared = results['linearity_residuals']['r_squared']
        avg_r2 = np.mean(list(r_squared.values()))
        print(f"   Average R² = {avg_r2:.3f}")
        if avg_r2 > 0.7:
            print("   ✓ Strong linear relationship - HMM assumption satisfied")
        elif avg_r2 > 0.4:
            print("   ⚠ Moderate linearity - consider feature transformations")
        else:
            print("   ✗ Weak linearity - recommend non-linear models (GMM, kernel methods)")
        
        # Stationarity summary
        print("\n2. STATIONARITY ASSUMPTION:")
        if 'stationary_rates' in results['stationarity_adf']:
            stat_rates = results['stationarity_adf']['stationary_rates']
            avg_stat = np.mean(list(stat_rates.values()))
            print(f"   Average stationarity rate = {avg_stat:.1%}")
            if avg_stat > 0.8:
                print("   ✓ Most features stationary - HMM assumption satisfied")
            elif avg_stat > 0.5:
                print("   ⚠ Mixed stationarity - monitor over longer sequences")
            else:
                print("   ✗ Non-stationary data - recommend differencing or detrending")
        
        # Gaussianity summary
        print("\n3. GAUSSIANITY ASSUMPTION:")
        if 'gaussian_rates' in results['gaussianity_shapiro']:
            gauss_rates = results['gaussianity_shapiro']['gaussian_rates']
            avg_gauss = np.mean(list(gauss_rates.values()))
            print(f"   Average Gaussianity rate = {avg_gauss:.1%}")
            if avg_gauss > 0.7:
                print("   ✓ Most features Gaussian - HMM emission model appropriate")
            elif avg_gauss > 0.4:
                print("   ⚠ Moderate Gaussianity - diagonal covariance may help")
            else:
                print("   ✗ Non-Gaussian distributions - recommend GMM emissions or Box-Cox transform")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        # Overall recommendation
        checks_passed = 0
        total_checks = 3
        
        if avg_r2 > 0.4:
            checks_passed += 1
        if 'stationary_rates' in results['stationarity_adf'] and avg_stat > 0.5:
            checks_passed += 1
        if 'gaussian_rates' in results['gaussianity_shapiro'] and avg_gauss > 0.4:
            checks_passed += 1
        
        print(f"\nPassed {checks_passed}/{total_checks} assumption checks")
        
        if checks_passed == 3:
            print("\n✓ Standard Gaussian HMM is appropriate for this data")
            print("  Recommended configuration:")
            print("  - Discrete-time HMM with 6 states")
            print("  - Gaussian emissions with diagonal covariance")
            print("  - Baum-Welch training with 50-100 iterations")
        elif checks_passed == 2:
            print("\n⚠ Standard HMM applicable with modifications")
            print("  Recommended adjustments:")
            if avg_r2 <= 0.4:
                print("  - Apply feature transformations (log, sqrt, Box-Cox)")
            if 'stationary_rates' in results['stationarity_adf'] and avg_stat <= 0.5:
                print("  - Use first-order differences for non-stationary features")
            if 'gaussian_rates' in results['gaussianity_shapiro'] and avg_gauss <= 0.4:
                print("  - Consider Gaussian Mixture Models for emissions")
        else:
            print("\n✗ Standard HMM assumptions violated")
            print("  Recommended alternatives:")
            print("  - Gaussian Mixture Model HMM (more flexible emissions)")
            print("  - Non-parametric HMM with kernel density estimation")
            print("  - Deep learning approaches (LSTM, Transformer)")
        
        print("\n" + "="*80)


def demo():
    """Demonstrate assumption testing on HAR dataset"""
    import sys
    sys.path.append('.')
    from scripts.data_loader import StochasticDataLoader
    
    print("Loading HAR dataset...")
    loader = StochasticDataLoader()
    
    # Load with selected features
    X_train, X_test, y_train, y_test, _, _, features = \
        loader.load_with_selected_features(method='anova')
    
    activity_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                     'SITTING', 'STANDING', 'LAYING']
    
    # Run all tests
    verifier = AssumptionVerifier(output_dir='outputs/results')
    results = verifier.run_all_tests(X_train, y_train, activity_names, features)
    
    print("\n" + "="*80)
    print("Assumption verification complete!")
    print("Check outputs/results/ for detailed plots and analysis")
    print("="*80)


if __name__ == '__main__':
    demo()
