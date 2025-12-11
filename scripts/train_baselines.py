import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
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

def evaluate_model(model, X_test, y_test, subjects, model_name, le):
    # Overall Accuracy
    y_pred = model.predict(X_test)
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Overall Accuracy: {overall_acc:.4f}")
    
    # Per-Subject Accuracy (2, 9, 12)
    target_subjects = [2, 9, 12]
    subj_accs = {}
    
    for subj in target_subjects:
        mask = (subjects == subj)
        if np.sum(mask) > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            subj_accs[subj] = acc
            print(f"{model_name} Subject {subj} Accuracy: {acc:.4f}")
            
    return overall_acc, subj_accs

def main():
    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('outputs/images', exist_ok=True)
    
    # 1. Load Data
    (X_train, y_train, subj_train), (X_test, y_test, subj_test), le = load_and_preprocess_data()
    
    # 2. Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 3. Train Naive Bayes
    print("\nTraining Gaussian Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    # 4. Evaluate
    rf_overall, rf_subj = evaluate_model(rf, X_test, y_test, subj_test, "Random Forest", le)
    nb_overall, nb_subj = evaluate_model(nb, X_test, y_test, subj_test, "Naive Bayes", le)
    
    # 5. Save Results
    results = []
    
    # Add RF results
    results.append({'Model': 'Random Forest', 'Subject': 'Overall', 'Accuracy': rf_overall})
    for s, acc in rf_subj.items():
        results.append({'Model': 'Random Forest', 'Subject': f'Subject {s}', 'Accuracy': acc})
        
    # Add NB results
    results.append({'Model': 'Naive Bayes', 'Subject': 'Overall', 'Accuracy': nb_overall})
    for s, acc in nb_subj.items():
        results.append({'Model': 'Naive Bayes', 'Subject': f'Subject {s}', 'Accuracy': acc})
        
    pd.DataFrame(results).to_csv('outputs/results/baseline_results.csv', index=False)
    print("\nBaseline results saved to outputs/results/baseline_results.csv")
    
    # 6. Plot Comparison (Including HMM/HSMM placeholders - user can fill or I can hardcode from previous runs)
    # Hardcoding previous HMM/HSMM results for the plot
    # Subject 2: HMM=0.9371, HSMM=0.9238
    # Subject 9: HMM=0.8611, HSMM=0.8576
    # Subject 12: HMM=0.9187, HSMM=0.9344
    
    hmm_accs = {'Subject 2': 0.9371, 'Subject 9': 0.8611, 'Subject 12': 0.9187}
    hsmm_accs = {'Subject 2': 0.9238, 'Subject 9': 0.8576, 'Subject 12': 0.9344}
    
    subjects = ['Subject 2', 'Subject 9', 'Subject 12']
    rf_vals = [rf_subj[2], rf_subj[9], rf_subj[12]]
    nb_vals = [nb_subj[2], nb_subj[9], nb_subj[12]]
    hmm_vals = [hmm_accs[s] for s in subjects]
    hsmm_vals = [hsmm_accs[s] for s in subjects]
    
    x = np.arange(len(subjects))
    width = 0.2
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, nb_vals, width, label='Naive Bayes', alpha=0.8)
    plt.bar(x - 0.5*width, rf_vals, width, label='Random Forest', alpha=0.8)
    plt.bar(x + 0.5*width, hmm_vals, width, label='GMM-HMM', alpha=0.8)
    plt.bar(x + 1.5*width, hsmm_vals, width, label='HSMM', alpha=0.8)
    
    plt.ylabel('Accuracy')
    plt.title('Model Comparison: Baselines vs HMM vs HSMM')
    plt.xticks(x, subjects)
    plt.legend(loc='lower right')
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig('outputs/images/model_comparison_bar.png')
    print("Comparison plot saved to outputs/images/model_comparison_bar.png")

if __name__ == "__main__":
    main()
