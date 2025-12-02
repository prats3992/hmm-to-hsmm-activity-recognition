import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_preprocess_data(n_components=65):
    print("Loading datasets...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    train_df = train_df.sort_values(by=['subject'])
    test_df = test_df.sort_values(by=['subject'])
    
    X_train_raw = train_df.iloc[:, :-2].values
    y_train_raw = train_df['Activity'].values
    subjects_train = train_df['subject'].values
    
    X_test_raw = test_df.iloc[:, :-2].values
    y_test_raw = test_df['Activity'].values
    subjects_test = test_df['subject'].values
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc = le.transform(y_test_raw)
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return (X_train_pca, y_train_enc, subjects_train), (X_test_pca, y_test_enc, subjects_test), le

def train_gmm_hmm(X, y, subjects, n_states=3, n_mix=2):
    print(f"Training GMM-HMMs...")
    models = {}
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        mask = (y == label)
        X_class = X[mask]
        subjects_class = subjects[mask]
        
        lengths = []
        for subj in np.unique(subjects_class):
            subj_len = np.sum(subjects_class == subj)
            if subj_len > 0:
                lengths.append(subj_len)
        
        model = hmm.GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X_class, lengths)
        models[label] = model
        
    return models

def compute_global_transitions(y, n_classes):
    print("Computing global transition matrix from training labels...")
    trans_mat = np.zeros((n_classes, n_classes))
    
    for t in range(1, len(y)):
        prev = y[t-1]
        curr = y[t]
        trans_mat[prev, curr] += 1
        
    # Normalize + Laplace Smoothing
    trans_mat += 1
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    trans_prob = trans_mat / row_sums
    
    return np.log(trans_prob)

def viterbi_decode(log_emissions, log_trans, log_start):
    # Standard Viterbi Algorithm
    T, N = log_emissions.shape
    
    # delta[t, i] = max prob of path ending at state i at time t
    delta = np.zeros((T, N))
    # psi[t, i] = best predecessor state for state i at time t
    psi = np.zeros((T, N), dtype=int)
    
    # Initialization
    delta[0] = log_start + log_emissions[0]
    
    # Recursion
    for t in range(1, T):
        for j in range(N):
            # max_i (delta[t-1, i] + log_trans[i, j])
            prev_scores = delta[t-1] + log_trans[:, j]
            best_prev = np.argmax(prev_scores)
            
            delta[t, j] = prev_scores[best_prev] + log_emissions[t, j]
            psi[t, j] = best_prev
            
    # Termination
    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(delta[T-1])
    
    # Backtracking
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
        
    return path

def main():
    # 1. Load Data
    (X_train, y_train, subj_train), (X_test, y_test, subj_test), le = load_and_preprocess_data()
    n_classes = len(le.classes_)
    
    # 2. Train Models
    models = train_gmm_hmm(X_train, y_train, subj_train)
    
    # 3. Global Transitions
    log_trans_mat = compute_global_transitions(y_train, n_classes)
    
    # Prior (Start) Probabilities (Uniform or from data)
    # Using uniform for simplicity as we pick a random subject start
    log_start_prob = np.log(np.ones(n_classes) / n_classes)
    
    # 4. Select Test Subjects
    # Subject IDs in test set: 2, 4, 9, 10, 12, 13, 18, 20, 24
    test_subjects = [2, 9, 12]
    
    for test_subj_id in test_subjects:
        print(f"\nVisualizing Viterbi decoding for Subject {test_subj_id}...")
        
        mask = (subj_test == test_subj_id)
        X_subj = X_test[mask]
        y_subj = y_test[mask]
        
        if len(X_subj) == 0:
            print(f"No data for subject {test_subj_id}")
            continue
        
        # 5. Compute Emission Probabilities
        # P(Observation | Activity)
        T = len(X_subj)
        log_emissions = np.zeros((T, n_classes))
        
        for i in range(T):
            sample = X_subj[i].reshape(1, -1)
            for label, model in models.items():
                # score() returns log-likelihood of the sequence (here length 1)
                try:
                    log_emissions[i, label] = model.score(sample)
                except:
                    log_emissions[i, label] = -np.inf
                    
        # 6. Run Viterbi
        print("Running Viterbi algorithm...")
        pred_path = viterbi_decode(log_emissions, log_trans_mat, log_start_prob)
        
        # 7. Compare with "Naive" Classification (Argmax of emissions)
        naive_path = np.argmax(log_emissions, axis=1)
        
        # Metrics
        acc_viterbi = accuracy_score(y_subj, pred_path)
        acc_naive = accuracy_score(y_subj, naive_path)
        
        print(f"Subject {test_subj_id} Accuracy:")
        print(f"Naive (Max Likelihood): {acc_naive:.4f}")
        print(f"Viterbi (Smoothed):     {acc_viterbi:.4f}")
        
        # 8. Visualization
        plt.figure(figsize=(15, 6))
        
        # Create a time axis
        t_axis = np.arange(T)
        
        # Plot True Labels
        plt.plot(t_axis, y_subj, 'k-', label='True Label', linewidth=2, alpha=0.6)
        
        # Plot Viterbi
        # Offset slightly to see overlap
        plt.plot(t_axis, pred_path, 'r--', label='Viterbi Prediction', linewidth=1.5)
        
        plt.yticks(range(n_classes), le.classes_)
        plt.xlabel('Time Window Index')
        plt.title(f'Activity Sequence Decoding (Subject {test_subj_id})\nViterbi Accuracy: {acc_viterbi:.2%}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs('outputs/images', exist_ok=True)
        plt.savefig(f'outputs/images/viterbi_decoding_subj{test_subj_id}.png')
        print(f"Visualization saved to outputs/images/viterbi_decoding_subj{test_subj_id}.png")

if __name__ == "__main__":
    main()
