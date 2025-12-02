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
    
    # Ensure data is sorted by subject to maintain temporal sequence chunks
    # (Assuming original file order is temporal within subjects, but grouping helps)
    train_df = train_df.sort_values(by=['subject'])
    test_df = test_df.sort_values(by=['subject'])
    
    # Extract features and targets
    # Last two columns are 'subject' and 'Activity'
    X_train_raw = train_df.iloc[:, :-2].values
    y_train_raw = train_df['Activity'].values
    subjects_train = train_df['subject'].values
    
    X_test_raw = test_df.iloc[:, :-2].values
    y_test_raw = test_df['Activity'].values
    subjects_test = test_df['subject'].values
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc = le.transform(y_test_raw)
    
    print(f"Classes: {le.classes_}")
    
    # Standardize
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # PCA
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return (X_train_pca, y_train_enc, subjects_train), (X_test_pca, y_test_enc, subjects_test), le

def train_hmm(X, y, subjects, n_states=6):
    print(f"Training HMM with {n_states} states...")
    
    # HMM in hmmlearn is unsupervised (it learns states from data).
    # However, we have labels. We can use them to initialize the HMM 
    # or train separate HMMs per class (common in speech recognition).
    #
    # Approach: Train one HMM per activity class.
    # Then for prediction, calculate likelihood of sequence under each model.
    
    models = {}
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        print(f"Training HMM for class {label}...")
        
        # Filter data for this class
        mask = (y == label)
        X_class = X[mask]
        subjects_class = subjects[mask]
        
        # Calculate lengths of contiguous sequences for this class
        # Since we filtered by class, we might have broken the temporal continuity 
        # if the user switches activities frequently. 
        # But usually in this dataset, activities are performed in blocks.
        # We will treat each subject's block of this activity as a sequence.
        
        lengths = []
        for subj in np.unique(subjects_class):
            subj_len = np.sum(subjects_class == subj)
            if subj_len > 0:
                lengths.append(subj_len)
        
        # Train GaussianHMM
        # n_components here refers to hidden states within the activity
        # We can try 1 state (just Gaussian) or more (sub-states of the activity)
        # Let's try 3 hidden states per activity to capture dynamics (e.g. start, middle, end of a step)
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X_class, lengths)
        models[label] = model
        
    return models

def evaluate_hmm(models, X, y, subjects):
    print("Evaluating...")
    preds = []
    true_labels = []
    
    # For evaluation, we need to predict the label for each sample.
    # In a real HMM scenario, we classify a *sequence*.
    # Here we want to classify each window.
    # We can compute the log-likelihood of a small window or single point under each model.
    
    # Note: Predicting single points with HMM loses the transition benefit.
    # But standard evaluation for this dataset is per-window accuracy.
    
    # Let's predict for each sample by calculating score under each model
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
        true_labels.append(y[i])
        
    return np.array(preds)

def main():
    # 1. Prepare Data
    (X_train, y_train, subj_train), (X_test, y_test, subj_test), le = load_and_preprocess_data(n_components=65)
    
    # 2. Train Models (One HMM per Activity Class)
    models = train_hmm(X_train, y_train, subj_train)
    
    # 3. Evaluate
    y_pred = evaluate_hmm(models, X_test, y_test, subj_test)
    
    # 4. Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 5. Save Results
    os.makedirs('outputs/results', exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('HMM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/images/confusion_matrix_hmm.png')
    print("Confusion matrix saved to outputs/images/confusion_matrix_hmm.png")

if __name__ == "__main__":
    main()
