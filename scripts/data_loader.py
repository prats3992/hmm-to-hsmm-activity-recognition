"""
Data Loader Utility for Stochastic Models
Handles loading data with proper feature selection while preserving Activity labels
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler


class StochasticDataLoader:
    """
    Load and prepare data for Discrete-Time HMM and Markov Chain models.
    Always preserves 'Activity' (hidden states/target) and 'subject' (ID) columns.
    """
    
    def __init__(self, train_path="data/train.csv", test_path="data/test.csv", 
                 selected_features_path="outputs/results/selected_features.json"):
        self.train_path = train_path
        self.test_path = test_path
        self.selected_features_path = selected_features_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_raw_data(self):
        """Load raw training and test data"""
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        print(f"Raw Data Loaded:")
        print(f"  Training: {train_df.shape}")
        print(f"  Test: {test_df.shape}")
        
        return train_df, test_df
    
    def load_with_selected_features(self, method='anova', combine_methods=False):
        """
        Load data with selected features.
        
        Parameters:
        -----------
        method : str
            Feature selection method: 'anova', 'mutual_info', or 'random_forest'
        combine_methods : bool
            If True, use union of all three methods
            
        Returns:
        --------
        X_train, X_test : DataFrames with selected features
        y_train, y_test : Activity labels
        subject_train, subject_test : Subject IDs
        feature_names : List of selected feature names
        """
        # Load data
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # Load selected features
        with open(self.selected_features_path, 'r') as f:
            selected_features = json.load(f)
        
        # Get feature list
        if combine_methods:
            # Union of all three methods
            feature_set = set(selected_features['anova'] + 
                            selected_features['mutual_info'] + 
                            selected_features['random_forest'])
            feature_names = sorted(list(feature_set))
            print(f"Using UNION of all methods: {len(feature_names)} unique features")
        else:
            feature_names = selected_features[method]
            print(f"Using {method}: {len(feature_names)} features")
        
        # Extract features and labels
        X_train = train_df[feature_names]
        X_test = test_df[feature_names]
        
        y_train = train_df['Activity']
        y_test = test_df['Activity']
        
        subject_train = train_df['subject']
        subject_test = test_df['subject']
        
        print(f"\nData prepared for modeling:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  Activities: {y_train.nunique()} classes")
        print(f"  Training subjects: {subject_train.nunique()}")
        print(f"  Test subjects: {subject_test.nunique()}")
        
        return X_train, X_test, y_train, y_test, subject_train, subject_test, feature_names
    
    def load_all_features(self):
        """
        Load data with ALL features (for comparison baseline).
        
        Returns:
        --------
        X_train, X_test : DataFrames with all 561 features
        y_train, y_test : Activity labels
        subject_train, subject_test : Subject IDs
        """
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # All features except Activity and subject
        feature_cols = [col for col in train_df.columns if col not in ['Activity', 'subject']]
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        
        y_train = train_df['Activity']
        y_test = test_df['Activity']
        
        subject_train = train_df['subject']
        subject_test = test_df['subject']
        
        print(f"Loaded ALL features:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  Activities: {y_train.nunique()} classes")
        
        return X_train, X_test, y_train, y_test, subject_train, subject_test
    
    def prepare_for_hmm(self, method='anova', standardize=True):
        """
        Prepare data specifically for Discrete-Time HMM modeling.
        Returns sequences grouped by subject with continuous observations.
        
        HMM Structure:
        - Hidden States: 6 discrete activities
        - Observations: Continuous sensor features (Gaussian emissions)
        
        Returns:
        --------
        train_sequences : list of (observations, labels, subject_id) tuples
        test_sequences : list of (observations, labels, subject_id) tuples
        """
        X_train, X_test, y_train, y_test, subject_train, subject_test, _ = \
            self.load_with_selected_features(method=method)
        
        # Standardize features
        if standardize:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Combine features with metadata
        train_df = X_train.copy()
        train_df['Activity'] = y_train.values
        train_df['subject'] = subject_train.values
        
        test_df = X_test.copy()
        test_df['Activity'] = y_test.values
        test_df['subject'] = subject_test.values
        
        # Create sequences per subject
        train_sequences = []
        for subject_id in sorted(train_df['subject'].unique()):
            subject_data = train_df[train_df['subject'] == subject_id]
            observations = subject_data.drop(['Activity', 'subject'], axis=1).values
            labels = subject_data['Activity'].values
            train_sequences.append((observations, labels, subject_id))
        
        test_sequences = []
        for subject_id in sorted(test_df['subject'].unique()):
            subject_data = test_df[test_df['subject'] == subject_id]
            observations = subject_data.drop(['Activity', 'subject'], axis=1).values
            labels = subject_data['Activity'].values
            test_sequences.append((observations, labels, subject_id))
        
        print(f"\nHMM Sequences prepared:")
        print(f"  Training: {len(train_sequences)} subjects")
        print(f"  Test: {len(test_sequences)} subjects")
        print(f"  Example sequence length: {len(train_sequences[0][0])} timesteps")
        print(f"  Feature dimension: {train_sequences[0][0].shape[1]}")
        
        return train_sequences, test_sequences
    
    def prepare_for_markov_chain(self):
        """
        Prepare data for Markov Chain modeling.
        Returns activity sequences per subject (discrete states only).
        
        Returns:
        --------
        train_sequences : dict {subject_id: [activity_list]}
        test_sequences : dict {subject_id: [activity_list]}
        activity_labels : list of unique activities
        """
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # Create activity sequences per subject
        train_sequences = {}
        for subject_id in sorted(train_df['subject'].unique()):
            activities = train_df[train_df['subject'] == subject_id]['Activity'].tolist()
            train_sequences[subject_id] = activities
        
        test_sequences = {}
        for subject_id in sorted(test_df['subject'].unique()):
            activities = test_df[test_df['subject'] == subject_id]['Activity'].tolist()
            test_sequences[subject_id] = activities
        
        activity_labels = sorted(train_df['Activity'].unique())
        
        print(f"\nMarkov Chain sequences prepared:")
        print(f"  Training: {len(train_sequences)} subjects")
        print(f"  Test: {len(test_sequences)} subjects")
        print(f"  Activities: {activity_labels}")
        print(f"  Example sequence (Subject 1): {len(train_sequences[1])} transitions")
        
        return train_sequences, test_sequences, activity_labels


def demo():
    """Demonstrate usage of StochasticDataLoader"""
    print("="*80)
    print("STOCHASTIC DATA LOADER - USAGE DEMO")
    print("Discrete-Time HMM and Markov Chain Analysis")
    print("="*80)
    
    loader = StochasticDataLoader()
    
    print("\n1. Loading data with ANOVA-selected features:")
    print("-" * 60)
    X_train, X_test, y_train, y_test, subj_train, subj_test, features = \
        loader.load_with_selected_features(method='anova')
    
    print("\n2. Loading data with COMBINED features (union of all methods):")
    print("-" * 60)
    X_train_combined, X_test_combined, y_train, y_test, _, _, features_combined = \
        loader.load_with_selected_features(combine_methods=True)
    
    print("\n3. Preparing sequences for Discrete-Time HMM:")
    print("-" * 60)
    train_seq, test_seq = loader.prepare_for_hmm(method='anova')
    print(f"   → Hidden States: 6 discrete activities")
    print(f"   → Observations: Continuous sensor features ({train_seq[0][0].shape[1]}-dim)")
    print(f"   → Model Type: Gaussian HMM (continuous observations)")
    
    print("\n4. Preparing sequences for Markov Chain:")
    print("-" * 60)
    train_mc, test_mc, activities = loader.prepare_for_markov_chain()
    print(f"   → Pure discrete state transitions")
    print(f"   → No continuous observations needed")
    
    print("\n" + "="*80)
    print("KEY POINTS:")
    print("="*80)
    print("✓ 'Activity' represents discrete hidden states (HMM) or discrete states (Markov)")
    print("✓ 'subject' enables sequential modeling per individual")
    print("✓ Feature selection: 561 → 100-191 features")
    print("✓ Discrete-Time HMM: continuous observations → discrete hidden states")
    print("✓ Markov Chain: discrete state → discrete state transitions")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo()
