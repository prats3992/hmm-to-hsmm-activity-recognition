import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def get_feature_importance():
    print("Loading dataset...")
    train_df = pd.read_csv('data/train.csv')
    
    X = train_df.iloc[:, :-2]
    y = train_df['Activity']
    feature_names = X.columns
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Training Random Forest to identify important features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)
    
    # Get importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 20 Most Important Original Features:")
    print("-" * 50)
    for f in range(20):
        print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
        
    return feature_names[indices[:60]]

if __name__ == "__main__":
    get_feature_importance()
