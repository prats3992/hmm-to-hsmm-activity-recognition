import pandas as pd
import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import accuracy_score
from scripts.visualize_viterbi import load_and_preprocess_data, train_gmm_hmm, compute_global_transitions, viterbi_decode

def test_median_filter():
    # Reuse the loading and training logic
    (X_train, y_train, subj_train), (X_test, y_test, subj_test), le = load_and_preprocess_data()
    n_classes = len(le.classes_)
    
    models = train_gmm_hmm(X_train, y_train, subj_train)
    log_trans_mat = compute_global_transitions(y_train, n_classes)
    log_start_prob = np.log(np.ones(n_classes) / n_classes)
    
    test_subjects = [2, 9, 12]
    
    print("\n--- Testing Median Filter (Window=5) on Viterbi Output ---")
    
    for subj_id in test_subjects:
        mask = (subj_test == subj_id)
        X_subj = X_test[mask]
        y_subj = y_test[mask]
        
        # 1. Compute Emissions
        T = len(X_subj)
        log_emissions = np.zeros((T, n_classes))
        for i in range(T):
            sample = X_subj[i].reshape(1, -1)
            for label, model in models.items():
                try:
                    log_emissions[i, label] = model.score(sample)
                except:
                    log_emissions[i, label] = -np.inf
                    
        # 2. Run Viterbi
        viterbi_path = viterbi_decode(log_emissions, log_trans_mat, log_start_prob)
        acc_viterbi = accuracy_score(y_subj, viterbi_path)
        
        # 3. Apply Median Filter
        # Window size 5 (must be odd)
        filtered_path = medfilt(viterbi_path, kernel_size=5)
        acc_filtered = accuracy_score(y_subj, filtered_path)
        
        print(f"Subject {subj_id}:")
        print(f"  Viterbi Accuracy:       {acc_viterbi:.4f}")
        print(f"  Viterbi + Median(5):    {acc_filtered:.4f}")
        print(f"  Improvement:            {acc_filtered - acc_viterbi:.4f}")

if __name__ == "__main__":
    test_median_filter()
