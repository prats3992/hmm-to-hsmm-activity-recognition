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

### Step 4: Correlation-Based Feature Pruning (NEW - Nov 2025)

**CRITICAL REFINEMENT:** Before feature selection, we now prune highly correlated features.

**Problem Identified:**
- **2,281 feature pairs** with correlation > 0.95 (massive redundancy!)
- Running feature selection on all 561 features with high multicollinearity:
  - ❌ **Inefficient:** Evaluating redundant features wastes computation
  - ❌ **Splits importance:** Feature importance gets divided among correlated features
  - ❌ **Unreliable estimates:** Multicollinearity destabilizes statistical tests

**Solution Implemented:**
```python
def prune_correlated_features(high_corr_pairs, threshold=0.95):
    # For each pair (A, B) with corr > 0.95, drop one feature (B)
    # Ensures we keep only one representative from each correlated group
```

**Results:**
- **561 → 276 features** (dropped 285 redundant features, 50.8% reduction)
- **Benefits:**
  - More reliable feature importance scores
  - Faster feature selection (276 vs 561 features to evaluate)
  - Reduced multicollinearity improves model stability
  - Better interpretability (each feature is distinct)

**Mathematical Justification:**

If features $X_i$ and $X_j$ have correlation $r_{ij} > 0.95$:
$$r_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sigma_i \sigma_j} > 0.95$$

Then they provide nearly identical information. Keeping both:
- Adds no discriminative power
- Increases dimensionality without benefit
- Can split feature importance: $\text{Importance}(X_i) + \text{Importance}(X_j) \approx 2 \times \text{Importance}(\text{either})$

**Example:** `tBodyAcc-std()-X` and `tBodyAcc-mad()-X` have correlation 0.9986 - they're measuring essentially the same thing (variability in acceleration).

---

### Step 5: Feature Selection (200 Features - Increased from 100)

**EVOLUTION:** Originally 50 → 100 → **NOW 200 features**

**Why Increase Again?**

After pruning correlated features:
- **Before pruning:** 100 from 561 = 18% of features
- **After pruning:** 100 from 276 = 36% of features
- **New approach:** 200 from 276 = **72% of pruned features**

**Rationale:**
1. **More features available:** Pruning removed redundancy, not information
2. **Less aggressive selection:** 276 pruned features → can afford to keep more
3. **Better coverage:** 200 features better capture activity nuances than 100
4. **HMM capacity:** With 200 dimensions, HMM has richer observation model:
   - 6 states × 200 features = 1,200 emission means + 1,200 variances
   - vs. 6 states × 100 features = 600 means + 600 variances

**Trade-off Analysis:**
- ✅ **Benefit:** More discriminative power (capture subtle differences between SITTING vs STANDING)
- ✅ **Acceptable cost:** Still 65% reduction from original 561 features
- ✅ **No redundancy:** All 200 features are relatively independent (pruned correlations)

---

### Step 6: Feature Selection Methods (Using Pruned Features)

**NOW OPERATING ON 276 PRUNED FEATURES (not 561)**

#### Method 1: ANOVA F-Test

**What it does:**
Measures the ratio of between-group variance to within-group variance for each feature.

**Why it's better after pruning:**
- No wasted computation on redundant features
- Feature scores more reliable (no multicollinearity)

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

**Why it's better after pruning:**
- Eliminates redundant information sources
- Each feature provides unique information gain

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

**Why it's better after pruning:**
- Importance not artificially split among correlated features
- More accurate representation of true feature contribution

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

### Final Feature Set (Updated Pipeline)

**New Workflow:**
```
561 Original Features
    ↓ [Correlation Analysis: Find 2,281 high-corr pairs]
    ↓ [Prune Correlated: Drop 285 features]
276 Pruned Features (50.8% reduction)
    ↓ [ANOVA F-Test: Select top 200]
    ↓ [Mutual Information: Select top 200]  
    ↓ [Random Forest: Select top 200]
200 Features per Method
```

**Old Workflow (for comparison):**
```
561 Original Features
    ↓ [Direct selection on all features]
100 Features per Method
```

**Why New Approach is Better:**

1. **Removes Redundancy First:** Correlation pruning eliminates duplicates before selection
2. **More Efficient:** Feature selection on 276 vs 561 features (49% faster)
3. **Better Estimates:** No multicollinearity → more reliable importance scores
4. **Larger Final Set:** 200 features (from 276) vs 100 features (from 561)
   - Same reduction ratio (~65% removed)
   - But starting from clean, independent features

**What we preserve:**
✅ **Activity** - Hidden states for HMM / Observable states for Markov Chain  
✅ **subject** - ID for sequential modeling (always included)  
✅ **200 sensor features** - Continuous observations for discrete-time HMM (increased from 100)

---

### Step 7: Principal Component Analysis (PCA) - REFINED

**CRITICAL FIX (Nov 2025):** Removed redundant scaling!

**What we do:**
Apply PCA to extract 100 principal components from 561 features.

**What Changed:**

**OLD (Incorrect):**
```python
X_train_scaled = StandardScaler().fit_transform(X_train)  # ❌ Wrong!
X_pca = PCA(n_components=100).fit_transform(X_train_scaled)
```

**NEW (Correct):**
```python
# UCI HAR data is ALREADY normalized to [-1, 1]
X_pca = PCA(n_components=100).fit_transform(X_train)  # ✅ Direct PCA
```

**Why This Matters:**

**Problem with old approach:**
- UCI HAR dataset is **already normalized** to range [-1, 1]
- `StandardScaler` transforms to mean=0, std=1 (changes variance structure!)
- This **redundant scaling** distorts the original variance relationships
- PCA is sensitive to scaling → wrong principal components

**Correct approach:**
- Data is already on comparable scale (all features in [-1, 1])
- Apply PCA directly to preserve original variance structure
- First PC captures true direction of maximum variance

**Mathematical Impact:**

PCA finds eigenvectors of covariance matrix $\mathbf{\Sigma}$:
- **With StandardScaler:** $\mathbf{\Sigma}_{\text{scaled}} = \text{Corr}(\mathbf{X})$ (correlation matrix)
- **Without StandardScaler:** $\mathbf{\Sigma}_{\text{original}} = \text{Cov}(\mathbf{X})$ (covariance matrix)

For pre-normalized data, the covariance matrix preserves meaningful variance relationships!

**Results (unchanged):**
- **100 components** explain **94.89%** of variance
- **63 components** needed for 90% variance
- First few components capture major patterns of motion

**When to use PCA:**
- ✅ For **visualization** (2D/3D plots)
- ✅ If HMM struggles with high dimensionality (computational efficiency)
- ❌ Generally prefer **selected features** for HMM (more interpretable observations)
- ⚠️ **NEVER double-scale pre-normalized data!**

---

### Step 8: Sequential Data Preparation

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

### 3. Why the Feature Selection Journey: 50 → 100 → 200?

**Evolution of our approach:**

**Version 1 (Initial):** 50 features
- Too aggressive: Only 9% of original features
- Risk of missing discriminative patterns

**Version 2 (Revised):** 100 features  
- More conservative: 18% of original features
- Better but still potentially limiting

**Version 3 (Current - Nov 2025):** 200 features **after pruning**
- **Key insight:** Pruning removes redundancy, not information!
- 561 features → 276 pruned (drop duplicates) → 200 selected
- 72% of pruned features retained
- Only 65% total reduction (still efficient)

**Why This Works:**
- **Before:** 100 from 561 = risk of missing patterns
- **After:** 200 from 276 = comprehensive coverage without redundancy
- **Trade-off:** Slightly more features, but all are **independent** and **informative**

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

### 1. Run Preprocessing and EDA (Updated Pipeline)

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run preprocessing with new correlation pruning + feature selection
python scripts/preprocessing_eda.py
```

**What it does (NEW):**
1. Load data (7,352 train, 2,947 test samples)
2. Basic EDA and activity analysis
3. **Correlation analysis** (find 2,281 high-corr pairs)
4. **Prune correlated features** (561 → 276 features)
5. **Feature selection** on pruned set (select 200 from 276)
6. PCA analysis (without redundant scaling)
7. Transition matrix computation

**Output:**
- `outputs/images/*.png` - Visualizations
- `outputs/results/selected_features.json` - **Top 200 features** (3 methods, updated)
- `outputs/results/transition_probabilities.csv` - Empirical activity transition matrix

### 2. Load Data for Modeling (Updated with 200 Features)

```python
from scripts.data_loader import StochasticDataLoader

loader = StochasticDataLoader()

# For Discrete-Time HMM (sequences with continuous observations)
train_seq, test_seq = loader.prepare_for_hmm(method='anova')
# Returns: list of (observations, labels, subject_id) tuples
# - observations: 200 continuous sensor features (N_timesteps x 200) ← UPDATED
# - labels: integer-encoded activity sequence 0-5 (for evaluation)
# - subject_id: identifier for subject

# NEW: Label encoding built-in
# Activities automatically encoded: {0: 'LAYING', 1: 'SITTING', 2: 'STANDING', 
#                                    3: 'WALKING', 4: 'WALKING_DOWNSTAIRS', 5: 'WALKING_UPSTAIRS'}

# For Markov Chain (discrete activity state transitions only)
train_mc, test_mc, activities = loader.prepare_for_markov_chain()
# Returns: dict {subject_id: [activity_list]}
# - Pure activity sequences for transition analysis
```

---

## ✅ Implementation Complete: Sections 4.2-4.5

### Section 4.2: Model Formulation

Implemented **discrete-time Gaussian HMM** in `scripts/hmm_model.py`:

**Model Architecture:**
- **Hidden States:** 6 discrete activities (WALKING, SITTING, STANDING, etc.)
- **Observations:** Continuous sensor features (100-191 dimensions)
- **Emissions:** Multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$ for each state $j$
- **Parameters** $\boldsymbol{\lambda} = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$:
  - $\mathbf{A}$: 6×6 transition matrix
  - $\mathbf{B}$: Emission distributions $\{\mathcal{N}(\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)\}_{j=1}^6$
  - $\boldsymbol{\pi}$: Initial state distribution

**Key Features:**
- Supervised initialization from labeled data
- Diagonal or full covariance matrices
- Numerical stability with log-space computations
- Stationary distribution computation

### Section 4.3: Assumption Verification

Comprehensive statistical tests in `scripts/assumption_tests.py`:

**1. Linearity Tests:**
- Residual analysis ($R^2$ computation, residual plots)
- Ramsey RESET test (F-test for non-linear terms)
- Detects if observations are linear in states + Gaussian noise

**2. Stationarity Tests:**
- Augmented Dickey-Fuller (ADF) test
- Autocorrelation Function (ACF) analysis
- Verifies time-invariant statistical properties

**3. Gaussianity Tests:**
- Shapiro-Wilk test (univariate normality)
- Q-Q plots (visual assessment)
- Validates emission distribution assumptions

**Outputs:**
- Detailed plots in `outputs/results/`
- Recommendations for model adjustments
- Alternative approaches when assumptions fail

### Section 4.4: Baum-Welch Algorithm

EM algorithm for parameter estimation in `scripts/hmm_model.py`:

**E-Step Components:**
- **Forward Algorithm:** Compute $\alpha_t(j) = P(x_1,\ldots,x_t, S_t=j | \boldsymbol{\lambda})$
- **Backward Algorithm:** Compute $\beta_t(i) = P(x_{t+1},\ldots,x_T | S_t=i, \boldsymbol{\lambda})$
- **Gamma Computation:** $\gamma_t(i) = P(S_t=i | \mathbf{X}, \boldsymbol{\lambda})$
- **Xi Computation:** $\xi_t(i,j) = P(S_t=i, S_{t+1}=j | \mathbf{X}, \boldsymbol{\lambda})$

**M-Step Updates:**
- **Initial Distribution:** $\hat{\pi}_i = \gamma_1(i)$
- **Transition Matrix:** $\hat{A}_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}$
- **Emission Means:** $\hat{\boldsymbol{\mu}}_j = \frac{\sum_t \gamma_t(j) \cdot \mathbf{x}_t}{\sum_t \gamma_t(j)}$
- **Emission Covariances:** $\hat{\boldsymbol{\Sigma}}_j = \frac{\sum_t \gamma_t(j) \cdot (\mathbf{x}_t-\hat{\boldsymbol{\mu}}_j)(\mathbf{x}_t-\hat{\boldsymbol{\mu}}_j)^T}{\sum_t \gamma_t(j)}$

**Features:**
- Handles multiple sequences (one per subject)
- Scaling to prevent numerical underflow
- Convergence monitoring with log-likelihood
- Training curve visualization

### Section 4.5: Viterbi Algorithm

Optimal state sequence decoding in `scripts/hmm_model.py`:

**Dynamic Programming:**
- **Initialization:** $\delta_1(j) = \pi_j \cdot P(x_1|S_t=j)$
- **Recursion:** $\delta_t(j) = \max_i[\delta_{t-1}(i) \cdot A_{ij}] \cdot P(x_t|S_t=j)$
- **Backtracking:** Reconstruct optimal path using $\psi_t$ pointers

**Complexity:** $O(T \cdot N^2)$ where $T$ is sequence length, $N=6$ states

**Outputs:**
- Most likely state sequence $\mathbf{S}^*$
- Log-probability of optimal path
- Per-sequence predictions

### Complete Evaluation Pipeline

Integrated workflow in `scripts/train_evaluate_hmm.py`:

**Pipeline Steps:**
1. Load sequential data (grouped by subject)
2. Run assumption verification tests
3. Train HMM with Baum-Welch (50 iterations)
4. Decode test sequences with Viterbi
5. Compute performance metrics

**Evaluation Metrics:**
- Confusion matrix (raw counts and normalized)
- Per-class Precision, Recall, F1-score
- Macro and weighted averages
- Transition matrix analysis
- Stationary distribution

**Usage:**
```python
from scripts.train_evaluate_hmm import HMMEvaluator

evaluator = HMMEvaluator()
hmm, metrics = evaluator.run_complete_pipeline(
    feature_method='anova',
    covariance_type='diag',
    n_iter=50,
    run_tests=True
)
```

### Run Complete HMM Pipeline (with Numerical Stability Improvements)

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run HMM training and evaluation (now with 200 features)
python scripts/train_evaluate_hmm.py

# Or run assumption tests separately
python scripts/assumption_tests.py
```

**What Changed (Nov 2025):**
- **200 features** instead of 100 (richer observation model)
- **Enhanced numerical stability:**
  - Stronger covariance regularization (min variance 1e-4 + 1e-3)
  - NaN detection and recovery in forward/backward algorithms
  - Validation in M-step to prevent degenerate solutions
  - Robust stationary distribution computation

**Expected Outputs:**
- `outputs/results/trained_hmm_model.pkl` - Saved HMM model
- `outputs/results/hmm_performance_metrics.json` - Accuracy, F1-scores
- `outputs/results/confusion_matrix_hmm.png` - Confusion matrices
- `outputs/results/learned_transition_matrix.png` - **Learned** transition probabilities (compare with empirical)
- `outputs/results/training_convergence.png` - EM convergence curve
- `outputs/results/assumption_test_results.json` - Statistical test results
- Various plots for linearity, stationarity, Gaussianity tests

**Key Difference: Empirical vs Learned Transition Matrix**

**Empirical TPM** (`transition_probabilities.csv`):
- Computed by counting transitions in labeled training data
- Example: LAYING→LAYING = 84.7% (people stay in activities)
- Used to **initialize** HMM

**Learned TPM** (`learned_transition_matrix.png`):
- **Refined** by Baum-Welch algorithm based on sensor patterns
- Discovered from continuous observations during EM training
- Should match empirical closely but may differ where sensor data reveals different patterns
- This is what HMM uses for **prediction** on unlabeled data

---

## 📈 Understanding the Two Transition Matrices

### Why Do We Have TWO Transition Matrices?

This often confuses people - let's clarify:

**1. Empirical Transition Matrix** (from preprocessing)
- **How:** Count transitions in labeled training data
- **Formula:** `P(j|i) = count(i→j) / count(i→*)`
- **Example:** In training data, LAYING→SITTING happened 87 times out of 1,407 LAYING samples = 6.2%
- **Use:** 
  - Pure Markov Chain analysis
  - **Initialize HMM** with reasonable starting transitions
  - Baseline for comparison

**2. Learned Transition Matrix** (from HMM training)
- **How:** Baum-Welch EM algorithm refines based on **sensor patterns**
- **Key Insight:** Learns what sensor data reveals about transitions
- **Example:** "When acceleration changes THIS way, transition to SITTING is likely"
- **Use:**
  - **Predict activities** from unlabeled sensor streams
  - Real-world deployment (no labels needed!)

### The Critical Difference

```
Scenario: You have NEW unlabeled sensor data from a user

Empirical TPM:   ❌ Cannot help (needs labeled sequences)
Learned TPM:     ✅ Can predict! (uses sensor patterns)
                    "These sensor readings + learned transitions 
                     → Most likely: SITTING → STANDING"
```

### Why Learn TPM When We Have Empirical TPM?

**Short Answer:** Because we want to predict activities from **unlabeled sensor data**!

**Detailed Explanation:**

1. **Empirical TPM** tells us: "In labeled data, what transitions occurred?"
2. **Learned TPM** tells us: "Given sensor patterns, what transitions are likely?"

**Real-World Application:**
- Deploy app on user's phone
- Collect accelerometer/gyroscope data (no labels!)
- Use **learned TPM + emission probabilities** to predict activities
- This is impossible with just empirical TPM (which requires labels)

### Comparison Analysis Available

**Markov Chain (Baseline):**
- Only uses empirical TPM: $P(S_t | S_{t-1})$
- No sensor observations
- Purely discrete state transitions

**HMM (Full Model):**
- Uses learned TPM: $P(S_t | S_{t-1})$ refined from sensor data
- Plus emission probabilities: $P(X_t | S_t)$
- Combined: $P(S_t, X_t | S_{t-1}, X_{<t})$

**Expected Outcome:** HMM should match or outperform Markov Chain because it uses more information (sensor patterns, not just transition counts)

## 📈 Next Steps & Baselines

### Model Comparison (as per Proposal)

Compare HMM against ML baselines:

1. **Naive Bayes** (ignores temporal structure)
2. **Support Vector Machine** with RBF kernel  
3. **Random Forest**

**Expected:** HMM should achieve 3-5% improvement by incorporating temporal dependencies

---

## 📝 Key Takeaways (Updated Nov 2025)

✅ **Activity is the target variable** - never discard it  
✅ **Subject ID enables sequential modeling** - always preserve it  
✅ **Prune correlations BEFORE selection** - removes redundancy efficiently (NEW)  
✅ **200 features from 276 pruned** - better than 100 from 561 (NEW)  
✅ **Multiple selection methods** provide robustness  
✅ **PCA without redundant scaling** - respect pre-normalized data (FIXED)  
✅ **Two transition matrices** - empirical (baseline) vs learned (predictive) (CLARIFIED)  
✅ **Numerical stability critical** - regularization + NaN checks prevent collapse (NEW)  
✅ **Discrete-time HMM** handles continuous observations with discrete hidden states

## 🔄 Recent Refinements (November 2025)

### 1. Correlation-Based Pruning
- **Added:** Pre-selection pruning removes 285 redundant features (50.8%)
- **Impact:** More reliable feature importance, faster selection, better model stability
- **Files:** `preprocessing_eda.py` - new `prune_correlated_features()` method

### 2. Fixed PCA Redundant Scaling
- **Fixed:** Removed StandardScaler (UCI HAR data already normalized)
- **Impact:** PCA now operates on correct variance structure
- **Files:** `preprocessing_eda.py` - removed scaling in `pca_analysis()`

### 3. Increased Feature Count
- **Changed:** 100 → 200 selected features
- **Rationale:** After pruning, 276 clean features available; 200 = 72% coverage
- **Impact:** Richer HMM observation model, better activity discrimination

### 4. Enhanced Numerical Stability
- **Added:** Stronger covariance regularization, NaN detection, validation in M-step
- **Impact:** Prevents degenerate solutions in HMM training with 200 features
- **Files:** `hmm_model.py` - multiple stability improvements

### 5. Documentation Improvements
- **Clarified:** Why we learn TPM (prediction on unlabeled data!)
- **Explained:** Empirical vs Learned transition matrix differences
- **Added:** `REFINEMENTS_SUMMARY.md` with detailed justification  

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
