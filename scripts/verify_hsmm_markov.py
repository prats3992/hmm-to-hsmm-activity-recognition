import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def get_embedded_sequence(y):
    """Extracts the sequence of distinct states (transitions only)."""
    embedded = [y[0]]
    for i in range(1, len(y)):
        if y[i] != y[i-1]:
            embedded.append(y[i])
    return np.array(embedded)

def test_markov_property(sequence, order=1):
    """
    Performs a Likelihood Ratio Test (LRT) to check if the sequence is Markovian.
    Compares Order-1 Markov Model vs Order-2 Markov Model.
    Null Hypothesis (H0): The process is Order-1 Markov.
    """
    n_states = len(np.unique(sequence))
    
    # Count transitions for Order 1: C(i -> j)
    trans_1 = np.zeros((n_states, n_states))
    # Count transitions for Order 2: C(i -> j -> k)
    trans_2 = np.zeros((n_states, n_states, n_states))
    
    # Map labels to 0..N-1
    le_map = {label: i for i, label in enumerate(np.unique(sequence))}
    mapped_seq = np.array([le_map[x] for x in sequence])
    
    # Fill counts
    for t in range(len(mapped_seq) - 1):
        i = mapped_seq[t]
        j = mapped_seq[t+1]
        trans_1[i, j] += 1
        
        if t < len(mapped_seq) - 2:
            k = mapped_seq[t+2]
            trans_2[i, j, k] += 1
            
    # Calculate Log-Likelihoods
    # P(j|i) = C(i,j) / C(i)
    # P(k|i,j) = C(i,j,k) / C(i,j)
    
    log_L1 = 0
    log_L2 = 0
    
    # Avoid log(0)
    epsilon = 1e-10
    
    # Order 1 Likelihood
    row_sums = trans_1.sum(axis=1, keepdims=True)
    probs_1 = (trans_1 + epsilon) / (row_sums + epsilon * n_states)
    
    for t in range(len(mapped_seq) - 1):
        i = mapped_seq[t]
        j = mapped_seq[t+1]
        log_L1 += np.log(probs_1[i, j])
        
    # Order 2 Likelihood
    # For each pair (i,j), calculate transition to k
    probs_2 = np.zeros_like(trans_2)
    for i in range(n_states):
        for j in range(n_states):
            total = trans_2[i, j].sum()
            if total > 0:
                probs_2[i, j, :] = (trans_2[i, j, :] + epsilon) / (total + epsilon * n_states)
            else:
                probs_2[i, j, :] = 1.0 / n_states # Uniform fallback
                
    for t in range(len(mapped_seq) - 2):
        i = mapped_seq[t]
        j = mapped_seq[t+1]
        k = mapped_seq[t+2]
        log_L2 += np.log(probs_2[i, j, k])
        
    # Likelihood Ratio Statistic
    # G^2 = -2 * (log_L1 - log_L2)
    # However, since L2 is a more complex model, it usually has higher likelihood.
    # We test if the improvement is significant.
    # Statistic = 2 * (LogL_Complex - LogL_Simple)
    
    LR_stat = 2 * (log_L2 - log_L1)
    
    # Degrees of Freedom
    # Order 1 params: N*(N-1)
    # Order 2 params: N*N*(N-1)
    # Diff: N^2(N-1) - N(N-1) = N(N-1)^2
    df = n_states * (n_states - 1)**2
    
    p_value = stats.chi2.sf(LR_stat, df)
    
    return LR_stat, p_value, df

def main():
    print("Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    y_train = train_df['Activity'].values
    
    print("\n--- 1. Testing Frame-Level Markov Property (Standard HMM Assumption) ---")
    # This is what we likely tested before and failed
    lr_stat, p_val, df = test_markov_property(y_train)
    print(f"Likelihood Ratio Statistic: {lr_stat:.2f}")
    print(f"Degrees of Freedom: {df}")
    print(f"P-Value: {p_val:.4e}")
    if p_val < 0.05:
        print("Result: REJECT Null Hypothesis. Frame-level sequence is NOT First-Order Markov.")
    else:
        print("Result: ACCEPT Null Hypothesis. Frame-level sequence IS First-Order Markov.")
        
    print("\n--- 2. Testing Embedded Markov Chain (HSMM Assumption) ---")
    # Extract embedded sequence (transitions only)
    y_embedded = get_embedded_sequence(y_train)
    print(f"Original Length: {len(y_train)}")
    print(f"Embedded Length: {len(y_embedded)} (Transitions only)")
    
    lr_stat_emb, p_val_emb, df_emb = test_markov_property(y_embedded)
    print(f"Likelihood Ratio Statistic: {lr_stat_emb:.2f}")
    print(f"Degrees of Freedom: {df_emb}")
    print(f"P-Value: {p_val_emb:.4f}")
    
    if p_val_emb < 0.05:
        print("Result: REJECT Null Hypothesis. Embedded sequence is NOT First-Order Markov.")
        print("Interpretation: The next activity depends on the previous activity AND the one before that.")
    else:
        print("Result: ACCEPT Null Hypothesis. Embedded sequence IS First-Order Markov.")
        print("Interpretation: The transition to the next activity depends ONLY on the current activity.")

if __name__ == "__main__":
    main()
