import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Reuse loading functions
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

def train_gmm_emissions(X, y, subjects, n_states=3, n_mix=2):
    print(f"Training GMM Emission Models...")
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
        
        # We use GMMHMM just to learn the emission density P(x|state)
        # We will ignore its internal transitions for the HSMM logic
        model = hmm.GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X_class, lengths)
        models[label] = model
        
    return models

def estimate_duration_params(y, n_classes):
    print("Estimating duration distributions...")
    durations = {i: [] for i in range(n_classes)}
    
    # Calculate run lengths
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
    
    # Fit Normal distribution to log-durations (Log-Normal) or just Normal
    # Durations are strictly positive, so Log-Normal is often better
    duration_params = {}
    
    plt.figure(figsize=(15, 10))
    
    for label in range(n_classes):
        durs = np.array(durations[label])
        if len(durs) < 2:
            mu, std = np.mean(durs), 1.0
        else:
            mu, std = np.mean(durs), np.std(durs)
            
        duration_params[label] = (mu, std)
        
        # Plot
        plt.subplot(2, 3, label+1)
        sns.histplot(durs, kde=True, stat='density')
        plt.title(f"Class {label} Duration\nMean={mu:.1f}, Std={std:.1f}")
        plt.xlim(0, np.percentile(durs, 95) * 1.5)
    
    plt.tight_layout()
    os.makedirs('outputs/images', exist_ok=True)
    plt.savefig('outputs/images/duration_distributions.png')
    
    return duration_params

def compute_inter_state_transitions(y, n_classes):
    print("Computing inter-state transition matrix (no self-transitions)...")
    trans_mat = np.zeros((n_classes, n_classes))
    
    for t in range(1, len(y)):
        prev = y[t-1]
        curr = y[t]
        if prev != curr:
            trans_mat[prev, curr] += 1
            
    # Normalize
    # Add small epsilon to avoid division by zero if a state never transitions
    trans_mat += 1e-5
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    trans_prob = trans_mat / row_sums
    
    return np.log(trans_prob)

def hsmm_viterbi(log_emissions, log_trans_inter, duration_params, max_dur=100):
    # HSMM Viterbi Algorithm
    # T: Time steps
    # N: Number of states
    # D: Max duration
    
    T, N = log_emissions.shape
    
    # Precompute duration log-probabilities for each state up to max_dur
    # Using Gaussian approximation for duration
    log_dur_probs = np.zeros((N, max_dur + 1))
    log_dur_probs[:, 0] = -np.inf # Duration 0 impossible
    
    for j in range(N):
        mu, std = duration_params[j]
        # Avoid std=0
        std = max(std, 1e-3)
        
        # Discretized Gaussian PDF
        d_range = np.arange(1, max_dur + 1)
        probs = stats.norm.pdf(d_range, loc=mu, scale=std)
        
        # Normalize to ensure sum(P(d)) <= 1 (it's truncated)
        probs = probs / (np.sum(probs) + 1e-9)
        log_dur_probs[j, 1:] = np.log(probs + 1e-20)

    # Precompute cumulative log emissions to speed up range sums
    # cum_log_emissions[t, j] = sum(log_emissions[0...t-1, j])
    cum_log_emissions = np.zeros((T + 1, N))
    cum_log_emissions[1:, :] = np.cumsum(log_emissions, axis=0)
    
    def get_emission_sum(t_start, t_end, state):
        # Sum from t_start to t_end-1
        return cum_log_emissions[t_end, state] - cum_log_emissions[t_start, state]

    # delta[t, j]: Max log-prob of ending in state j at time t (completing the state at t)
    delta = np.full((T, N), -np.inf)
    
    # psi_state[t, j]: Best previous state
    psi_state = np.zeros((T, N), dtype=int)
    
    # psi_dur[t, j]: Best duration of current state j ending at t
    psi_dur = np.zeros((T, N), dtype=int)
    
    # Initialization (t=0 to max_dur)
    # Assume we start at t=0 with a fresh state
    # We ignore the "start probability" vector for simplicity or assume uniform
    
    for t in range(min(max_dur, T)):
        # Duration d = t + 1
        d = t + 1
        for j in range(N):
            # Prob of starting at 0, being in state j for duration d, and emitting 0...t
            # We assume uniform start prob: log(1/N)
            log_start = -np.log(N)
            emission_score = get_emission_sum(0, t+1, j)
            dur_score = log_dur_probs[j, d]
            
            delta[t, j] = log_start + dur_score + emission_score
            psi_state[t, j] = -1 # Start
            psi_dur[t, j] = d

    # Recursion
    for t in range(1, T):
        # We want to compute delta[t, j]
        # We look back at previous state i ending at t-d
        
        for j in range(N):
            best_score = -np.inf
            best_prev_state = -1
            best_d = -1
            
            # Try all possible durations d for current state j
            # ending at t. So it started at t-d+1.
            # Previous state ended at t-d.
            
            for d in range(1, max_dur + 1):
                if t - d < 0:
                    continue
                
                t_prev = t - d
                
                # Find best previous state i
                # max_i ( delta[t_prev, i] + log_trans_inter[i, j] )
                
                # Vectorized search for best prev state
                prev_scores = delta[t_prev, :] + log_trans_inter[:, j]
                best_i = np.argmax(prev_scores)
                max_prev_score = prev_scores[best_i]
                
                if max_prev_score == -np.inf:
                    continue
                
                # Total score
                emission_score = get_emission_sum(t_prev + 1, t + 1, j)
                dur_score = log_dur_probs[j, d]
                
                total_score = max_prev_score + dur_score + emission_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_prev_state = best_i
                    best_d = d
            
            if best_score > -np.inf:
                delta[t, j] = best_score
                psi_state[t, j] = best_prev_state
                psi_dur[t, j] = best_d
                
    # Backtracking
    path = np.zeros(T, dtype=int)
    
    # Find best final state
    curr_state = np.argmax(delta[T-1])
    curr_t = T - 1
    
    while curr_t >= 0:
        d = psi_dur[curr_t, curr_state]
        prev_state = psi_state[curr_t, curr_state]
        
        # Fill path
        start_t = curr_t - d + 1
        path[start_t : curr_t + 1] = curr_state
        
        curr_t = start_t - 1
        curr_state = prev_state
        
        if curr_state == -1:
            break
            
    return path

def main():
    # 1. Load Data
    (X_train, y_train, subj_train), (X_test, y_test, subj_test), le = load_and_preprocess_data()
    n_classes = len(le.classes_)
    
    # 2. Train Emission Models (GMM)
    models = train_gmm_emissions(X_train, y_train, subj_train)
    
    # 3. Estimate Duration Parameters
    duration_params = estimate_duration_params(y_train, n_classes)
    
    # 4. Compute Inter-State Transitions
    log_trans_inter = compute_inter_state_transitions(y_train, n_classes)
    
    # 5. Test on Subjects 2, 9, 12
    test_subjects = [2, 9, 12]
    results = []
    
    for test_subj_id in test_subjects:
        print(f"\nRunning HSMM Viterbi for Subject {test_subj_id}...")
        
        mask = (subj_test == test_subj_id)
        if not np.any(mask):
            print(f"Subject {test_subj_id} not found in test set.")
            continue
            
        X_subj = X_test[mask]
        y_subj = y_test[mask]
        
        # Compute Emissions
        T = len(X_subj)
        log_emissions = np.zeros((T, n_classes))
        for i in range(T):
            sample = X_subj[i].reshape(1, -1)
            for label, model in models.items():
                try:
                    log_emissions[i, label] = model.score(sample)
                except:
                    log_emissions[i, label] = -np.inf
                    
        # Run HSMM Viterbi
        # Max duration: 2.56s windows. Activities usually last < 1 min.
        # 1 min = 60s / 1.28s (overlap) ~ 50 windows.
        # Let's set max_dur = 50 to be safe and fast.
        print("Decoding...")
        pred_path = hsmm_viterbi(log_emissions, log_trans_inter, duration_params, max_dur=50)
        
        acc = accuracy_score(y_subj, pred_path)
        print(f"HSMM Accuracy (Subject {test_subj_id}): {acc:.4f}")
        
        results.append({
            'Subject': test_subj_id,
            'Accuracy': acc
        })
        
        # Visualization
        plt.figure(figsize=(15, 6))
        t_axis = np.arange(T)
        plt.plot(t_axis, y_subj, 'k-', label='True', linewidth=2, alpha=0.6)
        plt.plot(t_axis, pred_path, 'g--', label='HSMM Prediction', linewidth=1.5)
        plt.yticks(range(n_classes), le.classes_)
        plt.title(f'HSMM Decoding (Subject {test_subj_id}) - Accuracy: {acc:.2%}')
        plt.legend()
        plt.savefig(f'outputs/images/hsmm_decoding_subj{test_subj_id}.png')
        print(f"Saved plot to outputs/images/hsmm_decoding_subj{test_subj_id}.png")
        
        # Confusion Matrix
        cm = confusion_matrix(y_subj, pred_path)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix (Subject {test_subj_id})')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'outputs/images/hsmm_confusion_matrix_subj{test_subj_id}.png')

    # Save overall results
    pd.DataFrame(results).to_csv('outputs/results/hsmm_accuracy_results.csv', index=False)

if __name__ == "__main__":
    main()
