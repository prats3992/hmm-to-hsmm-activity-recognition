import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
    print(f"Training GMM-HMM with {n_states} states and {n_mix} mixtures per state...")
    
    models = {}
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        print(f"Training GMM-HMM for class {label}...")
        
        mask = (y == label)
        X_class = X[mask]
        subjects_class = subjects[mask]
        
        lengths = []
        for subj in np.unique(subjects_class):
            subj_len = np.sum(subjects_class == subj)
            if subj_len > 0:
                lengths.append(subj_len)
        
        # GMMHMM usage
        # n_components = number of hidden states
        # n_mix = number of Gaussian mixtures per hidden state
        model = hmm.GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X_class, lengths)
        models[label] = model
        
    return models

def evaluate_model(models, X, y):
    print("Evaluating...")
    preds = []
    
    for i in range(len(X)):
        sample = X[i].reshape(1, -1)
        best_score = -np.inf
        best_label = -1
        
        for label, model in models.items():
            try:
                score = model.score(sample)
                if score > best_score:
                    best_score = score
                    best_label = label
            except:
                continue
        
        preds.append(best_label)
        
    return np.array(preds)

def main():
    # 1. Prepare Data
    (X_train, y_train, subj_train), (X_test, y_test, subj_test), le = load_and_preprocess_data(n_components=65)
    
    # 2. Train GMM-HMM
    # We use 3 hidden states (start, middle, end of action)
    # And 2 Gaussian mixtures per state to handle non-normality
    models = train_gmm_hmm(X_train, y_train, subj_train, n_states=3, n_mix=2)
    
    # 3. Evaluate
    y_pred = evaluate_model(models, X_test, y_test)
    
    # 4. Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 5. Save Results
    os.makedirs('outputs/results', exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('GMM-HMM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/images/confusion_matrix_gmm_hmm.png')
    print("Confusion matrix saved to outputs/images/confusion_matrix_gmm_hmm.png")

if __name__ == "__main__":
    main()
