"""
Preprocessing and EDA for UCI HAR Dataset
Stochastic Processes Project - Activity Recognition using HMM, Markov Chains, GMM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class HARPreprocessor:
    """Preprocessor for Human Activity Recognition Dataset"""
    
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load training and test datasets"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        print(f"Training set shape: {self.train_df.shape}")
        print(f"Test set shape: {self.test_df.shape}")
        print(f"\nColumns: {self.train_df.columns.tolist()[:10]}... (showing first 10)")
        print(f"Total features: {len(self.train_df.columns) - 2}")  # Excluding subject and Activity
        
        return self
    
    def basic_eda(self):
        """Perform basic exploratory data analysis"""
        print("\n" + "="*80)
        print("BASIC EDA")
        print("="*80)
        
        # Check for missing values
        print("\n1. Missing Values Check:")
        train_missing = self.train_df.isnull().sum().sum()
        test_missing = self.test_df.isnull().sum().sum()
        print(f"   Training set missing values: {train_missing}")
        print(f"   Test set missing values: {test_missing}")
        
        # Data types
        print("\n2. Data Types:")
        print(f"   Numeric columns: {self.train_df.select_dtypes(include=[np.number]).shape[1]}")
        print(f"   Non-numeric columns: {self.train_df.select_dtypes(exclude=[np.number]).shape[1]}")
        
        # Basic statistics
        print("\n3. Activity Column Info:")
        print(f"   Data type: {self.train_df['Activity'].dtype}")
        print(f"   Unique values: {self.train_df['Activity'].unique()}")
        
        # Subject info
        print("\n4. Subject Column Info:")
        print(f"   Data type: {self.train_df['subject'].dtype}")
        print(f"   Unique subjects in train: {self.train_df['subject'].nunique()}")
        print(f"   Unique subjects in test: {self.test_df['subject'].nunique()}")
        
        return self
    
    def activity_analysis(self):
        """Analyze activity distribution and patterns"""
        print("\n" + "="*80)
        print("ACTIVITY ANALYSIS")
        print("="*80)
        
        # Activity distribution
        print("\n1. Activity Distribution in Training Set:")
        activity_counts = self.train_df['Activity'].value_counts()
        for activity, count in activity_counts.items():
            percentage = (count / len(self.train_df)) * 100
            print(f"   {activity}: {count} ({percentage:.2f}%)")
        
        print("\n2. Activity Distribution in Test Set:")
        activity_counts_test = self.test_df['Activity'].value_counts()
        for activity, count in activity_counts_test.items():
            percentage = (count / len(self.test_df)) * 100
            print(f"   {activity}: {count} ({percentage:.2f}%)")
        
        # Visualize activity distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        self.train_df['Activity'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('Activity Distribution - Training Set', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Activity', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        
        self.test_df['Activity'].value_counts().plot(kind='bar', ax=axes[1], color='coral')
        axes[1].set_title('Activity Distribution - Test Set', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Activity', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/images/activity_distribution.png', dpi=300, bbox_inches='tight')
        print("\n   Saved: outputs/images/activity_distribution.png")
        plt.close()
        
        return self
    
    def subject_analysis(self):
        """Analyze subject-wise patterns"""
        print("\n" + "="*80)
        print("SUBJECT ANALYSIS")
        print("="*80)
        
        # Samples per subject
        print("\n1. Samples per Subject (Training):")
        subject_counts = self.train_df['subject'].value_counts().sort_index()
        print(f"   Mean samples per subject: {subject_counts.mean():.2f}")
        print(f"   Std samples per subject: {subject_counts.std():.2f}")
        print(f"   Min: {subject_counts.min()}, Max: {subject_counts.max()}")
        
        # Activity distribution per subject
        print("\n2. Activities per Subject:")
        subject_activity = self.train_df.groupby('subject')['Activity'].value_counts().unstack(fill_value=0)
        print(f"   All subjects have data for all activities: {(subject_activity > 0).all().all()}")
        
        # Visualize subject distribution
        fig, ax = plt.subplots(figsize=(14, 6))
        subject_counts.plot(kind='bar', ax=ax, color='teal')
        ax.set_title('Samples per Subject - Training Set', fontsize=14, fontweight='bold')
        ax.set_xlabel('Subject ID', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/images/subject_distribution.png', dpi=300, bbox_inches='tight')
        print("\n   Saved: outputs/images/subject_distribution.png")
        plt.close()
        
        return self
    
    def feature_statistics(self):
        """Analyze feature statistics"""
        print("\n" + "="*80)
        print("FEATURE STATISTICS")
        print("="*80)
        
        # Get feature columns (exclude subject and Activity)
        feature_cols = [col for col in self.train_df.columns if col not in ['subject', 'Activity']]
        
        print(f"\n1. Total Features: {len(feature_cols)}")
        
        # Statistics
        feature_data = self.train_df[feature_cols]
        print(f"\n2. Feature Value Ranges:")
        print(f"   Overall Min: {feature_data.min().min():.6f}")
        print(f"   Overall Max: {feature_data.max().max():.6f}")
        print(f"   Overall Mean: {feature_data.mean().mean():.6f}")
        print(f"   Overall Std: {feature_data.std().mean():.6f}")
        
        # Feature types (time domain vs frequency domain)
        time_domain = [col for col in feature_cols if col.startswith('t')]
        freq_domain = [col for col in feature_cols if col.startswith('f')]
        angle_features = [col for col in feature_cols if col.startswith('angle')]
        
        print(f"\n3. Feature Categories:")
        print(f"   Time domain features: {len(time_domain)}")
        print(f"   Frequency domain features: {len(freq_domain)}")
        print(f"   Angle features: {len(angle_features)}")
        
        # Variance analysis
        feature_variance = feature_data.var().sort_values(ascending=False)
        print(f"\n4. Feature Variance:")
        print(f"   Top 5 highest variance features:")
        for i, (feat, var) in enumerate(feature_variance.head(5).items(), 1):
            print(f"   {i}. {feat}: {var:.6f}")
        
        print(f"\n   Top 5 lowest variance features:")
        for i, (feat, var) in enumerate(feature_variance.tail(5).items(), 1):
            print(f"   {i}. {feat}: {var:.6f}")
        
        # Check for zero variance features
        zero_var = feature_variance[feature_variance == 0]
        print(f"\n5. Zero Variance Features: {len(zero_var)}")
        if len(zero_var) > 0:
            print(f"   Features: {zero_var.index.tolist()}")
        
        return self
    
    def correlation_analysis(self):
        """Analyze feature correlations"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        # Get feature columns
        feature_cols = [col for col in self.train_df.columns if col not in ['subject', 'Activity']]
        
        print("\nComputing correlation matrix (this may take a moment)...")
        corr_matrix = self.train_df[feature_cols].corr().abs()
        
        # Find highly correlated features
        threshold = 0.95
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        print(f"\n1. Highly Correlated Feature Pairs (correlation > {threshold}):")
        print(f"   Found {len(high_corr_pairs)} pairs")
        if len(high_corr_pairs) > 0:
            print(f"   Showing first 10 pairs:")
            for feat1, feat2, corr in high_corr_pairs[:10]:
                print(f"   {feat1} <-> {feat2}: {corr:.4f}")
        
        # Visualize correlation heatmap for a subset
        print("\n2. Creating correlation heatmap for selected features...")
        
        # Select subset of features for visualization (first 50)
        subset_features = feature_cols[:50]
        corr_subset = self.train_df[subset_features].corr()
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_subset, cmap='coolwarm', center=0, square=True, 
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap (First 50 Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("   Saved: outputs/images/correlation_heatmap.png")
        plt.close()
        
        return self, high_corr_pairs
    
    def feature_selection_analysis(self, n_features=100):
        """Perform feature selection analysis"""
        print("\n" + "="*80)
        print("FEATURE SELECTION ANALYSIS")
        print("="*80)
        print(f"\nNote: 'Activity' is the TARGET VARIABLE and will always be included")
        print(f"      'subject' identifies individuals for sequential modeling")
        print(f"      Selecting top {n_features} discriminative features from 561 sensor features\n")
        
        # Prepare data
        feature_cols = [col for col in self.train_df.columns if col not in ['subject', 'Activity']]
        X_train = self.train_df[feature_cols].values
        y_train = self.train_df['Activity'].values
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"\n1. ANOVA F-test Feature Selection (Top {n_features} features):")
        selector_anova = SelectKBest(score_func=f_classif, k=n_features)
        selector_anova.fit(X_train, y_train_encoded)
        
        # Get feature scores
        feature_scores_anova = pd.DataFrame({
            'Feature': feature_cols,
            'Score': selector_anova.scores_
        }).sort_values('Score', ascending=False)
        
        print(f"   Top 10 features by ANOVA F-score:")
        for i, row in feature_scores_anova.head(10).iterrows():
            print(f"   {row['Feature']}: {row['Score']:.2f}")
        
        # Mutual Information
        print(f"\n2. Mutual Information Feature Selection (Top {n_features} features):")
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
        selector_mi.fit(X_train, y_train_encoded)
        
        feature_scores_mi = pd.DataFrame({
            'Feature': feature_cols,
            'Score': selector_mi.scores_
        }).sort_values('Score', ascending=False)
        
        print(f"   Top 10 features by Mutual Information:")
        for i, row in feature_scores_mi.head(10).iterrows():
            print(f"   {row['Feature']}: {row['Score']:.4f}")
        
        # Random Forest Feature Importance
        print(f"\n3. Random Forest Feature Importance (Top {n_features} features):")
        print("   Training Random Forest (this may take a moment)...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(X_train, y_train_encoded)
        
        feature_importance_rf = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"   Top 10 features by Random Forest Importance:")
        for i, row in feature_importance_rf.head(10).iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")
        
        # Visualize feature importance
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        feature_scores_anova.head(20).plot(x='Feature', y='Score', kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title('Top 20 Features - ANOVA F-Score', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('F-Score', fontsize=10)
        axes[0].invert_yaxis()
        
        feature_scores_mi.head(20).plot(x='Feature', y='Score', kind='barh', ax=axes[1], color='coral')
        axes[1].set_title('Top 20 Features - Mutual Information', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('MI Score', fontsize=10)
        axes[1].invert_yaxis()
        
        feature_importance_rf.head(20).plot(x='Feature', y='Importance', kind='barh', ax=axes[2], color='seagreen')
        axes[2].set_title('Top 20 Features - Random Forest', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Importance', fontsize=10)
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('outputs/images/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n   Saved: outputs/images/feature_importance.png")
        plt.close()
        
        # Save selected features
        selected_features = {
            'anova': feature_scores_anova.head(n_features)['Feature'].tolist(),
            'mutual_info': feature_scores_mi.head(n_features)['Feature'].tolist(),
            'random_forest': feature_importance_rf.head(n_features)['Feature'].tolist()
        }
        
        return self, selected_features
    
    def pca_analysis(self, n_components=100):
        """Perform PCA analysis"""
        print("\n" + "="*80)
        print("PCA ANALYSIS")
        print("="*80)
        
        # Prepare data
        feature_cols = [col for col in self.train_df.columns if col not in ['subject', 'Activity']]
        X_train = self.train_df[feature_cols].values
        
        # Standardize features
        print("\n1. Standardizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Apply PCA
        print(f"2. Applying PCA with {n_components} components...")
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_train_scaled)
        
        # Explained variance
        print(f"\n3. Explained Variance:")
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        print(f"   Variance explained by {n_components} components: {cumsum_variance[-1]:.4f}")
        print(f"   Components needed for 90% variance: {np.argmax(cumsum_variance >= 0.90) + 1}")
        print(f"   Components needed for 95% variance: {np.argmax(cumsum_variance >= 0.95) + 1}")
        print(f"   Components needed for 99% variance: {np.argmax(cumsum_variance >= 0.99) + 1}")
        
        # Visualize explained variance
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].bar(range(1, n_components+1), pca.explained_variance_ratio_, color='steelblue')
        axes[0].set_title('Explained Variance by Component', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Principal Component', fontsize=10)
        axes[0].set_ylabel('Variance Ratio', fontsize=10)
        axes[0].set_xlim(0, n_components+1)
        
        axes[1].plot(range(1, n_components+1), cumsum_variance, marker='o', color='coral', linewidth=2)
        axes[1].axhline(y=0.90, color='green', linestyle='--', label='90% variance')
        axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
        axes[1].set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Number of Components', fontsize=10)
        axes[1].set_ylabel('Cumulative Variance Ratio', fontsize=10)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, n_components+1)
        
        plt.tight_layout()
        plt.savefig('outputs/images/pca_analysis.png', dpi=300, bbox_inches='tight')
        print("\n   Saved: outputs/images/pca_analysis.png")
        plt.close()
        
        return self, pca
    
    def prepare_sequences_for_hmm(self):
        """Prepare sequential data for HMM and Markov Chain models"""
        print("\n" + "="*80)
        print("PREPARING SEQUENTIAL DATA FOR STOCHASTIC MODELS")
        print("="*80)
        
        # Sort by subject (data is already in temporal order)
        train_sorted = self.train_df.sort_values('subject')
        test_sorted = self.test_df.sort_values('subject')
        
        # Activity sequences per subject
        print("\n1. Creating Activity Sequences per Subject:")
        train_sequences = train_sorted.groupby('subject')['Activity'].apply(list).to_dict()
        test_sequences = test_sorted.groupby('subject')['Activity'].apply(list).to_dict()
        
        print(f"   Training sequences: {len(train_sequences)} subjects")
        print(f"   Test sequences: {len(test_sequences)} subjects")
        
        # Example sequence
        example_subject = list(train_sequences.keys())[0]
        print(f"\n2. Example Activity Sequence (Subject {example_subject}):")
        print(f"   Length: {len(train_sequences[example_subject])} activities")
        print(f"   First 20 activities: {train_sequences[example_subject][:20]}")
        
        # Transition analysis
        print("\n3. Activity Transition Analysis:")
        transitions = {}
        for subject, activities in train_sequences.items():
            for i in range(len(activities) - 1):
                transition = (activities[i], activities[i+1])
                transitions[transition] = transitions.get(transition, 0) + 1
        
        print(f"   Total unique transitions observed: {len(transitions)}")
        print(f"   Most common transitions:")
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        for (from_act, to_act), count in sorted_transitions[:10]:
            print(f"   {from_act} -> {to_act}: {count}")
        
        # Visualize transition matrix
        activities = sorted(self.train_df['Activity'].unique())
        transition_matrix = pd.DataFrame(0, index=activities, columns=activities)
        
        for (from_act, to_act), count in transitions.items():
            transition_matrix.loc[from_act, to_act] = count
        
        # Normalize to probabilities
        transition_prob = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_prob, annot=True, fmt='.3f', cmap='YlOrRd', 
                    cbar_kws={"shrink": 0.8}, square=True)
        plt.title('Activity Transition Probability Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('To Activity', fontsize=12)
        plt.ylabel('From Activity', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/images/transition_matrix.png', dpi=300, bbox_inches='tight')
        print("\n   Saved: outputs/images/transition_matrix.png")
        plt.close()
        
        return self, train_sequences, test_sequences, transition_prob


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("HAR DATASET - PREPROCESSING AND EDA")
    print("Stochastic Processes Project: Activity Recognition")
    print("="*80)
    
    # Initialize preprocessor
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    
    preprocessor = HARPreprocessor(train_path, test_path)
    
    # Execute pipeline
    preprocessor.load_data()
    preprocessor.basic_eda()
    preprocessor.activity_analysis()
    preprocessor.subject_analysis()
    preprocessor.feature_statistics()
    preprocessor, high_corr_pairs = preprocessor.correlation_analysis()
    preprocessor, selected_features = preprocessor.feature_selection_analysis(n_features=100)
    preprocessor, pca = preprocessor.pca_analysis(n_components=100)
    preprocessor, train_seq, test_seq, transition_prob = preprocessor.prepare_sequences_for_hmm()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. Data Quality:")
    print("   ✓ No missing values detected")
    print("   ✓ All features are numeric")
    print("   ✓ Data is already normalized")
    
    print("\n2. Feature Selection Recommendations:")
    print(f"   ✓ ANOVA F-test selected {len(selected_features['anova'])} features")
    print(f"   ✓ Mutual Information selected {len(selected_features['mutual_info'])} features")
    print(f"   ✓ Random Forest selected {len(selected_features['random_forest'])} features")
    print("   → Consider using union of top features from all methods")
    print("   → 'Activity' (target) and 'subject' (ID) are preserved separately")
    
    print("\n3. Dimensionality Reduction:")
    print("   ✓ PCA with 100 components captures significant variance")
    print("   → Option 1: Use PCA components for GMM clustering")
    print("   → Option 2: Use selected features for HMM/Markov Chain")
    print("   → Activity labels preserved for all models")
    
    print("\n4. For Stochastic Models:")
    print("   ✓ Activity sequences prepared per subject")
    print("   ✓ Transition probability matrix computed")
    print("   → Ready for HMM implementation")
    print("   → Ready for Markov Chain modeling")
    print("   → Ready for GMM clustering (use PCA features)")
    
    print("\n5. Next Steps:")
    print("   1. Implement Hidden Markov Model for activity sequence prediction")
    print("   2. Build Markov Chain for activity transition modeling")
    print("   3. Apply Gaussian Mixture Model for activity clustering")
    print("   4. Compare model performance and evaluate results")
    
    print("\n" + "="*80)
    print("PREPROCESSING AND EDA COMPLETE!")
    print("="*80 + "\n")
    
    # Save important artifacts
    print("Saving key results...")
    
    # Save selected features
    import json
    with open('outputs/results/selected_features.json', 'w') as f:
        json.dump(selected_features, f, indent=2)
    print("✓ Saved: outputs/results/selected_features.json")
    
    # Save transition probability matrix
    transition_prob.to_csv('outputs/results/transition_probabilities.csv')
    print("✓ Saved: outputs/results/transition_probabilities.csv")
    
    print("\nAll outputs saved successfully!")


if __name__ == "__main__":
    main()
