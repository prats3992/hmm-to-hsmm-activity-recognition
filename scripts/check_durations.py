import pandas as pd
import numpy as np

def check_durations():
    train_df = pd.read_csv('data/test.csv')
    y = train_df['Activity'].values
    
    durations = []
    current_label = y[0]
    current_len = 1
    
    for i in range(1, len(y)):
        if y[i] == current_label:
            current_len += 1
        else:
            durations.append(current_len)
            current_label = y[i]
            current_len = 1
    durations.append(current_len)
    
    print(f"Max duration: {max(durations)}")
    print(f"95th percentile: {np.percentile(durations, 95)}")
    print(f"99th percentile: {np.percentile(durations, 99)}")

if __name__ == "__main__":
    check_durations()
