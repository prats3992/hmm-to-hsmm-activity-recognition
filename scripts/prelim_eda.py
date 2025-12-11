import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Combine for some analyses
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    all_df = pd.concat([train_df, test_df], axis=0)
    
    output_dir = 'outputs/images/eda'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Missing Values
    print("\nChecking for missing values...")
    missing = all_df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    
    # 2. Class Distribution
    print("\nPlotting Class Distribution...")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=all_df, x='Activity', hue='dataset')
    plt.title('Activity Distribution in Train and Test Sets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png')
    print(f"Saved {output_dir}/class_distribution.png")
    
    # 3. Subject Distribution
    print("\nPlotting Subject Distribution...")
    plt.figure(figsize=(15, 6))
    sns.countplot(data=all_df, x='subject', hue='dataset')
    plt.title('Data points per Subject')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/subject_distribution.png')
    print(f"Saved {output_dir}/subject_distribution.png")
    
    # 4. Feature Analysis (First few features)
    # Assuming last 2 columns are 'subject' and 'Activity' (and we added 'dataset')
    feature_cols = [c for c in train_df.columns if c not in ['subject', 'Activity', 'dataset']]
    print(f"\nNumber of feature columns: {len(feature_cols)}")
    
    print("Plotting distribution of first 3 features...")
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(feature_cols[:3]):
        plt.subplot(1, 3, i+1)
        sns.histplot(data=train_df, x=col, hue='Activity', element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions_top3.png')
    print(f"Saved {output_dir}/feature_distributions_top3.png")

    # 5. Correlation of first 20 features
    print("\nPlotting Correlation Matrix (First 20 features)...")
    plt.figure(figsize=(12, 10))
    corr = train_df[feature_cols[:20]].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix (First 20 Features)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix_top20.png')
    print(f"Saved {output_dir}/correlation_matrix_top20.png")
    
    # 6. Boxplot of first feature by Activity
    print("\nPlotting Boxplot of first feature by Activity...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_df, x='Activity', y=feature_cols[0])
    plt.title(f'Boxplot of {feature_cols[0]} by Activity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_feature1.png')
    print(f"Saved {output_dir}/boxplot_feature1.png")

if __name__ == "__main__":
    perform_eda()
