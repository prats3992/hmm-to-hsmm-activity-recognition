import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

def plot_tpms():
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    y_train = train_df['Activity'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    classes = le.classes_
    n_classes = len(classes)
    
    output_dir = 'outputs/images'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Frame-Level TPM (GMM-HMM / Standard HMM View)
    # Transitions every time step (0.02s)
    print("Computing Frame-Level TPM...")
    frame_trans = np.zeros((n_classes, n_classes))
    
    for t in range(len(y_enc) - 1):
        curr_s = y_enc[t]
        next_s = y_enc[t+1]
        # We only count transitions within the same subject
        # But for simplicity here, we assume continuous stream or handle breaks
        # Given the dataset structure, breaks are rare compared to data size
        frame_trans[curr_s, next_s] += 1
        
    # Normalize
    frame_trans_prob = frame_trans / frame_trans.sum(axis=1, keepdims=True)
    
    # Plot Frame-Level
    plt.figure(figsize=(10, 8))
    sns.heatmap(frame_trans_prob, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Frame-Level Transition Probability Matrix (GMM-HMM)\nNote the dominant diagonal (Self-Transitions)')
    plt.ylabel('Current Activity (t)')
    plt.xlabel('Next Activity (t+1)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tpm_frame_level_gmm_hmm.png')
    print(f"Saved {output_dir}/tpm_frame_level_gmm_hmm.png")
    
    # 2. Embedded TPM (HSMM View)
    # Transitions only when state changes
    print("Computing Embedded TPM...")
    embedded_trans = np.zeros((n_classes, n_classes))
    
    # Compress sequence (remove self-transitions)
    compressed_seq = [y_enc[0]]
    for t in range(1, len(y_enc)):
        if y_enc[t] != y_enc[t-1]:
            compressed_seq.append(y_enc[t])
            
    compressed_seq = np.array(compressed_seq)
    
    for t in range(len(compressed_seq) - 1):
        curr_s = compressed_seq[t]
        next_s = compressed_seq[t+1]
        embedded_trans[curr_s, next_s] += 1
        
    # Normalize
    # Add epsilon to avoid division by zero if a state is never left (e.g. end of seq)
    row_sums = embedded_trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    embedded_trans_prob = embedded_trans / row_sums
    
    # Plot Embedded
    plt.figure(figsize=(10, 8))
    sns.heatmap(embedded_trans_prob, annot=True, fmt='.2f', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Embedded Transition Probability Matrix (HSMM)\nTransitions GIVEN that the activity changes')
    plt.ylabel('Current Activity')
    plt.xlabel('Next Activity')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tpm_embedded_hsmm.png')
    print(f"Saved {output_dir}/tpm_embedded_hsmm.png")

if __name__ == "__main__":
    plot_tpms()
