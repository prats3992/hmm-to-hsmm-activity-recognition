# Stochastic Processes Project: Human Activity Recognition

## Project Overview

This project applies **Discrete-Time Hidden Markov Models (HMM)** and **Markov Chain analysis** to the **UCI Human Activity Recognition (HAR) Dataset** for predicting and analyzing human activity sequences from smartphone sensor data.

**Course:** Stochastic Processes (Semester 7)  
**Dataset:** UCI HAR Dataset with 561 sensor features from accelerometer and gyroscope  
**Activities:** 6 classes (STANDING, SITTING, LAYING, WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS)

---

## 📁 Project Structure

```
PROJECT/
├── data/                          # Dataset files
│   ├── train.csv                  # Training data (7,352 samples)
│   └── test.csv                   # Test data (2,947 samples)
├── scripts/                       # Python scripts
│   ├── preprocessing_eda.py       # Data preprocessing and EDA
│   └── data_loader.py            # Utility for loading prepared data
├── outputs/                       # Generated outputs
│   ├── images/                    # Visualizations
│   │   ├── activity_distribution.png
│   │   ├── subject_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_importance.png
│   │   ├── pca_analysis.png
│   │   └── transition_matrix.png
│   └── results/                   # Analysis results
│       ├── selected_features.json
│       └── transition_probabilities.csv
├── docs/                          # Documentation
│   └── stochastic_proposal.pdf    # Project proposal
└── README.md                      # This file
```

---

## 🎯 Problem Statement

Given sequential sensor data from smartphones worn by subjects performing various activities, we aim to:

1. **Model activity transitions** using Markov Chains (discrete state transitions)
2. **Predict activity sequences** using Discrete-Time Hidden Markov Models (continuous observations, discrete hidden states)
3. **Analyze temporal patterns** in activity sequences and transition dynamics

---

## 📊 Dataset Description

### UCI Human Activity Recognition Dataset

**Source:** Sensor data from 30 volunteers performing 6 activities  
**Sensors:** Accelerometer and Gyroscope (3-axial raw signals)  
**Sampling Rate:** 50Hz  
**Feature Extraction:** 561 features computed from time and frequency domains

### Data Split
- **Training:** 21 subjects, 7,352 samples
- **Test:** 9 subjects, 2,947 samples

### Feature Categories
- **Time domain features (265):** `tBodyAcc`, `tGravityAcc`, `tBodyAccJerk`, `tBodyGyro`, etc.
- **Frequency domain features (289):** `fBodyAcc`, `fBodyAccJerk`, `fBodyGyro`, etc.
- **Angle features (7):** Angles between sensor axis and gravity vectors

### Target Variable
- **Activity:** Categorical label (6 classes)
- **Class Distribution:** Relatively balanced (~13-19% per class)

### Metadata
- **subject:** Integer ID identifying individual participants (preserved for sequential modeling)

---

## 🔧 Preprocessing Pipeline

### Step 1: Data Loading and Basic EDA

**What we do:**
- Load `train.csv` and `test.csv`
- Check for missing values
- Examine data types and basic statistics
- Analyze activity and subject distributions

**Why:**
- Understand data quality and completeness
- Verify that data is ready for analysis
- Identify any preprocessing requirements

**Results:**
- ✅ No missing values
- ✅ All 561 features are numeric and already normalized to [-1, 1]
- ✅ Balanced class distribution
- ✅ All subjects have samples for all activities

---

### Step 2: Activity Analysis

**What we do:**
- Calculate activity distribution in train/test sets
- Visualize class balance
- Analyze per-subject activity coverage

**Why:**
- Check for class imbalance (affects model training)
- Ensure representative sampling across activities
- Verify data split maintains distribution

**Key Findings:**
- LAYING: 19.14% (train), 18.22% (test)
- STANDING: 18.69% (train), 18.05% (test)
- SITTING: 17.49% (train), 16.66% (test)
- WALKING: 16.68% (train), 16.83% (test)
- WALKING_UPSTAIRS: 14.59% (train), 15.98% (test)
- WALKING_DOWNSTAIRS: 13.41% (train), 14.25% (test)

---

### Step 3: Feature Statistics and Correlation Analysis

**What we do:**
- Compute variance for all 561 features
- Identify zero/low variance features
- Calculate correlation matrix
- Find highly correlated feature pairs (>0.95 correlation)

**Why:**
- **Low variance features** contain little information and can be removed
- **High correlation** indicates redundancy - keeping both features doesn't add value
- Reduces dimensionality while preserving information
- Improves computational efficiency

**Mathematical Foundation:**

**Pearson Correlation Coefficient:**
$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Where:
- $r_{xy}$ ranges from -1 to +1
- $|r_{xy}| > 0.95$ indicates high correlation (redundancy)

**Key Findings:**
- **2,281 feature pairs** with correlation > 0.95
- High redundancy among std, mad, and iqr features
- Justifies aggressive feature selection

---

### Step 4: Feature Selection (100 Features)

**CRITICAL DECISION:** Initially selected only 50 features, but this was too aggressive.

**Problem Identified:**
- With 561 features and only 6 activity classes, 50 features might miss important discriminative information
- The **Activity column is the TARGET VARIABLE**, not a feature to select
- Must always preserve Activity and subject for sequential modeling

**Solution:** Increased to **100 features** using three complementary methods.

#### Method 1: ANOVA F-Test

**What it does:**
Measures the ratio of between-group variance to within-group variance for each feature.

**Mathematical Foundation:**
$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}} = \frac{\sum_{k=1}^{K} n_k(\bar{x}_k - \bar{x})^2 / (K-1)}{\sum_{k=1}^{K}\sum_{i=1}^{n_k}(x_{ik} - \bar{x}_k)^2 / (N-K)}$$

Where:
- $K$ = number of classes (6 activities)
- $n_k$ = samples in class $k$
- $\bar{x}_k$ = mean of feature in class $k$
- $\bar{x}$ = overall mean
- $N$ = total samples

**Why use it:**
- Higher F-score → feature better separates activity classes
- Captures linear relationships between features and activities
- Fast computation
- Statistically interpretable

**Top Features:**
1. `fBodyAccJerk-entropy()-X` (F=36,918)
2. `tGravityAcc-mean()-X` (F=29,363)
3. `tGravityAcc-min()-X` (F=28,175)

**Interpretation:** Gravity-related features and jerk entropy strongly discriminate between static (sitting/standing) and dynamic (walking) activities.

---

#### Method 2: Mutual Information

**What it does:**
Measures the mutual dependence between feature and target - how much information about activity is gained by observing the feature.

**Mathematical Foundation:**
$$I(X;Y) = \sum_{y \in Y}\sum_{x \in X} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$$

Where:
- $I(X;Y)$ = mutual information between feature $X$ and activity $Y$
- $p(x,y)$ = joint probability
- $p(x), p(y)$ = marginal probabilities

**Why use it:**
- Captures **non-linear** relationships (unlike ANOVA)
- Information-theoretic measure (no distributional assumptions)
- Detects complex dependencies

**Top Features:**
1. `tBodyAcc-max()-X` (MI=1.007)
2. `tGravityAcc-min()-Y` (MI=0.944)
3. `tBodyAccJerk-max()-X` (MI=0.938)

**Interpretation:** Maximum and minimum values of acceleration capture extremes of motion patterns.

---

#### Method 3: Random Forest Feature Importance

**What it does:**
Trains ensemble of decision trees and measures feature importance based on impurity decrease (Gini importance).

**Mathematical Foundation:**
$$\text{Importance}(f) = \frac{1}{T}\sum_{t=1}^{T}\sum_{n \in \text{nodes}(t, f)} \Delta i(n)$$

Where:
- $T$ = number of trees
- $\Delta i(n)$ = decrease in Gini impurity at node $n$ when splitting on feature $f$
- Gini impurity: $G = \sum_{k=1}^{K} p_k(1-p_k)$

**Why use it:**
- Captures **feature interactions** (unlike univariate methods)
- Robust to outliers and non-linear relationships
- Considers feature combinations
- Empirically strong performance

**Top Features:**
1. `tGravityAcc-mean()-X` (Importance=0.0349)
2. `angle(X,gravityMean)` (Importance=0.0323)
3. `tGravityAcc-max()-X` (Importance=0.0309)

**Interpretation:** Gravity direction strongly indicates body orientation (sitting/standing vs. laying).

---

### Final Feature Set

**Strategy:** Use **union of all three methods** = **191 unique features**

**Why combine methods:**
- ANOVA captures linear separability
- Mutual Information captures non-linear patterns
- Random Forest captures interactions
- Union ensures no important feature is missed

**What we preserve:**
✅ **Activity** - Hidden states for HMM / Observable states for Markov Chain  
✅ **subject** - ID for sequential modeling (always included)  
✅ **100-191 sensor features** - Continuous observations for discrete-time HMM

---

### Step 5: Principal Component Analysis (PCA)

**What we do:**
Apply PCA to extract 100 principal components from 561 features.

**Why:**
- **Dimensionality reduction** while preserving variance
- **Removes multicollinearity** (decorrelates features)
- **Noise reduction** (minor components often represent noise)
- **Computational efficiency** for HMM (fewer observation dimensions → faster training)

**Mathematical Foundation:**

PCA finds orthogonal directions of maximum variance:

$$\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X}\mathbf{w}) = \arg\max_{\|\mathbf{w}\|=1} \mathbf{w}^T\mathbf{\Sigma}\mathbf{w}$$

Where:
- $\mathbf{\Sigma}$ = covariance matrix of features
- $\mathbf{w}_i$ = $i$-th eigenvector (principal component direction)
- $\lambda_i$ = $i$-th eigenvalue (variance explained)

**Explained Variance:**
$$\text{Cumulative Variance} = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{561}\lambda_i}$$

**Results:**
- **100 components** explain **94.89%** of variance
- **63 components** needed for 90% variance
- First few components capture major patterns of motion

**When to use PCA:**
- ✅ For **visualization** (2D/3D plots)
- ✅ If HMM struggles with high dimensionality (computational efficiency)
- ❌ Generally prefer **selected features** for HMM (more interpretable observations)

---

### Step 6: Sequential Data Preparation

**What we do:**
- Group data by subject to create activity sequences
- Compute activity transition probabilities
- Build transition matrix for Markov Chain

**Why:**
- HMM and Markov Chains model **sequences**, not independent samples
- Each subject performs activities in temporal order
- Transition probabilities reveal activity patterns

**Activity Transition Matrix:**

$$P(A_t = j \mid A_{t-1} = i) = \frac{\text{Count}(i \to j)}{\sum_{j'}\text{Count}(i \to j')}$$

Where:
- $A_t$ = activity at time $t$
- $P(i \to j)$ = probability of transitioning from activity $i$ to $j$

**Key Findings:**
- **Most transitions are self-transitions** (staying in same activity)
  - LAYING → LAYING: 1,192 transitions (84.7%)
  - STANDING → STANDING: 1,179 transitions (85.8%)
  - SITTING → SITTING: 1,063 transitions (82.6%)
  
- **Common activity switches:**
  - SITTING ↔ LAYING (115 transitions)
  - STANDING ↔ SITTING (92 transitions)
  - LAYING → WALKING (110 transitions)

**Interpretation:**
- People tend to remain in an activity for multiple consecutive samples (50Hz sampling rate)
- Transitions between static postures (sitting/standing/laying) are common
- Walking activities often separate static activities

---

## 🔍 Why Each Step Matters

### 1. Why Feature Selection (not just use all 561)?

**Problems with using all features:**
- ❌ **Curse of dimensionality** - models need exponentially more data
- ❌ **Overfitting** - models memorize noise in redundant features
- ❌ **Computational cost** - 561 features slow training significantly
- ❌ **Interpretability** - hard to understand what matters

**Benefits of selection:**
- ✅ **Better generalization** - focuses on discriminative information
- ✅ **Faster training** - fewer parameters to estimate
- ✅ **Reduced noise** - removes irrelevant/redundant features
- ✅ **Interpretable models** - understand which movements matter

### 2. Why Keep Activity Separate (CRITICAL FIX)?

**Initial Mistake:** Treated dataset like a typical ML problem where features and target are separated.

**Problem:** For stochastic models:
- HMM predicts **hidden states** from **observations** - Activity is the hidden state!
- Markov Chain models **state transitions** - Activity is the state!
- GMM clusters observations, but Activity validates clustering

**Correct Approach:**
```python
X = selected_features  # 100-191 sensor features (observations)
y = Activity          # Target variable (states) - ALWAYS PRESERVED
subject = subject_id   # For sequential grouping - ALWAYS PRESERVED
```

**Why this matters:**
- HMM needs to learn $P(\text{Activity} \mid \text{Sensors})$
- Markov Chain needs activity sequence: [SITTING, SITTING, STANDING, ...]
- Cannot predict Activity if we throw it away!

### 3. Why 100 Features (not 50)?

**Initial approach:** 50 features seemed reasonable (10% of 561)

**Problem discovered:**
- 6 activity classes with subtle differences (SITTING vs. STANDING)
- High-dimensional sensor space (561) likely needs more features to capture nuances
- Risk of missing discriminative patterns

**Solution:** Increased to 100 features (~18% of original)
- More conservative dimensionality reduction
- Union of three methods = 191 features available
- Balance between reduction and information preservation

### 4. Why Three Feature Selection Methods?

**Complementary strengths:**

| Method | Captures | Assumptions | Use Case |
|--------|----------|-------------|----------|
| **ANOVA F-test** | Linear separability | Normal distribution, equal variance | Fast, interpretable, good for linearly separable classes |
| **Mutual Information** | Non-linear relationships | None | Detects complex dependencies, robust |
| **Random Forest** | Feature interactions | None | Ensemble wisdom, captures combinations |

**By combining all three:**
- Cover different types of feature-target relationships
- Reduce risk of missing important features
- Robust feature set for diverse modeling approaches

---

## 🧮 Mathematical Concepts Summary

### 1. Correlation
$$r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$
- Measures linear relationship between features
- Used to detect redundancy

### 2. ANOVA F-Statistic
$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$$
- Higher F → better class separation
- Tests if feature means differ across activities

### 3. Mutual Information
$$I(X;Y) = H(Y) - H(Y|X)$$
- Reduction in uncertainty about Y given X
- Measures information gain

### 4. Gini Impurity (Random Forest)
$$G = 1 - \sum_{i=1}^{K}p_i^2$$
- Lower Gini → purer node
- Importance = total impurity decrease from splitting on feature

### 5. PCA (Eigendecomposition)
$$\mathbf{\Sigma} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^T$$
- $\mathbf{U}$ = eigenvectors (principal directions)
- $\mathbf{\Lambda}$ = eigenvalues (variance explained)
- Projects data onto directions of maximum variance

### 6. Transition Probability (Markov Property)
$$P(A_t \mid A_{t-1}, A_{t-2}, \ldots, A_0) = P(A_t \mid A_{t-1})$$
- Future depends only on current state (memoryless)
- Basis for Markov Chain modeling

---

## 🚀 Usage Guide

### 1. Run Preprocessing and EDA

```bash
python scripts/preprocessing_eda.py
```

**Output:**
- `outputs/images/*.png` - Visualizations
- `outputs/results/selected_features.json` - Top 100 features (3 methods)
- `outputs/results/transition_probabilities.csv` - Activity transition matrix

### 2. Load Data for Modeling

```python
from scripts.data_loader import StochasticDataLoader

loader = StochasticDataLoader()

# For Discrete-Time HMM (sequences with continuous observations)
train_seq, test_seq = loader.prepare_for_hmm(method='anova')
# Returns: list of (observations, labels, subject_id) tuples
# - observations: continuous sensor features (N_timesteps x N_features)
# - labels: true activity sequence (for evaluation)

# For Markov Chain (discrete activity state transitions only)
train_mc, test_mc, activities = loader.prepare_for_markov_chain()
# Returns: dict {subject_id: [activity_list]}
# - Pure activity sequences for transition analysis
```

---

## 📈 Next Steps

### 1. Discrete-Time Hidden Markov Model (HMM)
**Goal:** Predict hidden activity states from continuous sensor observations

**Model Structure:**
- **Hidden States**: 6 discrete activities
- **Observations**: Continuous sensor features (100-191 dimensions)
- **Transition Matrix** $A$: $P(\text{state}_t | \text{state}_{t-1})$
- **Emission Probabilities** $B$: $P(\text{observation}_t | \text{state}_t)$ - Gaussian emissions

**Approach:**
- Use **Gaussian HMM** (continuous observations)
- Learn parameters: initial state distribution $\pi$, transition matrix $A$, emission parameters $B$
- Inference: Viterbi algorithm for most likely state sequence
- Evaluation: Sequence prediction accuracy, state-wise F1 scores

**Implementation:**
```python
from hmmlearn import hmm
# Initialize Gaussian HMM with 6 states
model = hmm.GaussianHMM(n_components=6, covariance_type='diag', n_iter=100)
# Train on observation sequences
# Predict hidden state sequences using Viterbi
```

### 2. Markov Chain Analysis
**Goal:** Analyze discrete activity state transition dynamics

**Approach:**
- Estimate transition matrix $P$ from observed activity sequences
- Analyze stationary distribution: $\pi P = \pi$
- Compute transition probabilities: $P(A_t = j | A_{t-1} = i)$
- Predict next activity given current activity
- Compare Markov Chain transitions with HMM's learned transitions

**Implementation:**
```python
# Load precomputed transition matrix
import pandas as pd
transition_matrix = pd.read_csv('outputs/results/transition_probabilities.csv')

# Analyze steady-state distribution
# Simulate activity sequences
# Compare with HMM predictions
```

---

## 📝 Key Takeaways

✅ **Activity is the target variable** - never discard it  
✅ **Subject ID enables sequential modeling** - always preserve it  
✅ **Feature selection reduces dimensionality** without losing information  
✅ **Multiple selection methods** provide robustness  
✅ **PCA optional for HMM** (computational efficiency vs. interpretability trade-off)  
✅ **Transition matrix** reveals temporal activity patterns  
✅ **100-191 features** balances reduction and preservation  
✅ **Discrete-time HMM** handles continuous observations with discrete hidden states  

---

## 📚 References

1. UCI HAR Dataset: Anguita et al., 2013
2. Hidden Markov Models: Rabiner, 1989
3. Feature Selection Methods: Guyon & Elisseeff, 2003
4. PCA: Jolliffe, 2002

---

## 👤 Author

**Project:** Stochastic Processes (SEM7)  
**Dataset:** UCI Human Activity Recognition  
**Repository:** stoDS (prats3992)
