# Quick Reference: Feature Selection Methods

## 1. ANOVA F-Test

### Purpose
Identifies features where the means differ significantly across activity classes.

### Formula
$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$

**Detailed:**
$$F = \frac{\frac{1}{k-1}\sum_{i=1}^{k}n_i(\bar{x}_i - \bar{x})^2}{\frac{1}{N-k}\sum_{i=1}^{k}\sum_{j=1}^{n_i}(x_{ij} - \bar{x}_i)^2}$$

Where:
- $k$ = number of classes (6 activities)
- $n_i$ = number of samples in class $i$
- $\bar{x}_i$ = mean of feature in class $i$
- $\bar{x}$ = overall mean
- $N$ = total number of samples

### Interpretation
- **Higher F-score** = feature better discriminates between activities
- **Null hypothesis:** All class means are equal
- **Alternative:** At least one class mean differs

### Advantages
✅ Fast computation  
✅ Statistically interpretable  
✅ Good for linearly separable classes  

### Limitations
❌ Assumes normal distribution  
❌ Only captures linear relationships  
❌ Sensitive to outliers  

---

## 2. Mutual Information

### Purpose
Measures the mutual dependence between feature and target - how much knowing the feature reduces uncertainty about the activity.

### Formula
$$I(X;Y) = \sum_{y \in Y}\sum_{x \in X} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$$

**Alternative form:**
$$I(X;Y) = H(Y) - H(Y|X)$$

Where:
- $H(Y)$ = entropy of activity labels
- $H(Y|X)$ = conditional entropy of Y given X
- $p(x,y)$ = joint probability distribution
- $p(x), p(y)$ = marginal probabilities

### Entropy Definition
$$H(Y) = -\sum_{y \in Y} p(y)\log p(y)$$

### Interpretation
- **MI = 0:** X and Y are independent (feature tells us nothing about activity)
- **MI > 0:** X and Y are dependent (feature provides information)
- **Higher MI** = more informative feature

### Advantages
✅ Captures non-linear relationships  
✅ No distributional assumptions  
✅ Information-theoretic foundation  

### Limitations
❌ Requires discretization for continuous features  
❌ Computationally more expensive than ANOVA  
❌ Can be biased for features with many unique values  

---

## 3. Random Forest Feature Importance

### Purpose
Measures feature importance based on how much each feature decreases impurity when building decision trees.

### Formula (Gini Importance)

**Gini Impurity:**
$$G(t) = 1 - \sum_{i=1}^{K}p_i^2(t)$$

Where:
- $p_i(t)$ = proportion of class $i$ at node $t$
- $K$ = number of classes

**Feature Importance:**
$$\text{Importance}(f) = \frac{1}{T}\sum_{t=1}^{T}\sum_{n \in N_f(t)} \frac{|n|}{|N|}\Delta G(n)$$

Where:
- $T$ = number of trees in forest
- $N_f(t)$ = set of nodes in tree $t$ that split on feature $f$
- $|n|$ = number of samples reaching node $n$
- $|N|$ = total samples
- $\Delta G(n)$ = decrease in Gini impurity at node $n$

### Impurity Decrease
$$\Delta G(n) = G(n) - \frac{|n_L|}{|n|}G(n_L) - \frac{|n_R|}{|n|}G(n_R)$$

Where:
- $n_L, n_R$ = left and right child nodes after split

### Interpretation
- **Higher importance** = feature contributes more to classification
- Averaged over many trees (ensemble wisdom)
- Considers feature interactions (unlike univariate methods)

### Advantages
✅ Captures feature interactions  
✅ Handles non-linear relationships  
✅ Robust to outliers  
✅ No distributional assumptions  
✅ Empirically strong performance  

### Limitations
❌ Biased toward high-cardinality features  
❌ Computationally expensive (trains forest)  
❌ Importance values are relative, not absolute  

---

## 4. Principal Component Analysis (PCA)

### Purpose
Transform features into orthogonal directions of maximum variance (dimensionality reduction while preserving information).

### Mathematical Formulation

**Optimization Problem:**
$$\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X}\mathbf{w})$$

Subject to: $\|\mathbf{w}\| = 1$

**Solution via Eigendecomposition:**
$$\mathbf{\Sigma} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^T$$

Where:
- $\mathbf{\Sigma}$ = covariance matrix of features
- $\mathbf{U}$ = matrix of eigenvectors (principal components)
- $\mathbf{\Lambda}$ = diagonal matrix of eigenvalues (variance explained)

**Covariance Matrix:**
$$\mathbf{\Sigma} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$$

(assuming $\mathbf{X}$ is centered: $\bar{x} = 0$)

### Principal Components
- **1st PC:** Direction of maximum variance
- **2nd PC:** Direction of max variance orthogonal to 1st PC
- **kth PC:** Direction of max variance orthogonal to all previous PCs

### Variance Explained
$$\text{Variance explained by PC}_k = \frac{\lambda_k}{\sum_{i=1}^{p}\lambda_i}$$

$$\text{Cumulative variance} = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{p}\lambda_i}$$

Where $p$ = total number of original features

### Interpretation
- **First few PCs** capture main patterns (signal)
- **Later PCs** often capture noise
- **Dimensionality reduction:** Keep components explaining 90-95% variance

### Advantages
✅ Removes multicollinearity  
✅ Reduces dimensionality  
✅ Denoising effect  
✅ Computational efficiency  

### Limitations
❌ Components are linear combinations (harder to interpret)  
❌ Assumes linear relationships  
❌ Sensitive to feature scaling (must standardize first)  

---

## 5. Correlation (Pearson)

### Purpose
Measure linear relationship between two features (detect redundancy).

### Formula
$$r_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

### Properties
- $-1 \leq r \leq 1$
- $r = 1$: Perfect positive linear relationship
- $r = -1$: Perfect negative linear relationship
- $r = 0$: No linear relationship

### Interpretation in Feature Selection
- $|r| > 0.95$: Highly correlated (redundant features)
- Can remove one feature from highly correlated pairs
- Reduces dimensionality without losing information

---

## Comparison Table

| Method | Type | Captures | Complexity | Best For |
|--------|------|----------|------------|----------|
| **ANOVA F-test** | Univariate | Linear separability | O(nk) | Fast screening, interpretable results |
| **Mutual Information** | Univariate | Non-linear relationships | O(n log n) | Complex dependencies, no assumptions |
| **Random Forest** | Multivariate | Feature interactions | O(n log n × T × m) | Robust selection, ensemble learning |
| **PCA** | Transformation | Variance structure | O(p²n + p³) | Dimensionality reduction, decorrelation |
| **Correlation** | Pairwise | Linear redundancy | O(p²n) | Remove redundant features |

Where:
- $n$ = number of samples
- $k$ = number of classes
- $p$ = number of features
- $T$ = number of trees
- $m$ = number of features sampled per split

---

## When to Use Each Method

### Use ANOVA F-test when:
- ✅ Quick feature screening needed
- ✅ Interpretability is important
- ✅ Features approximately normally distributed
- ✅ Linear relationships expected

### Use Mutual Information when:
- ✅ Non-linear relationships suspected
- ✅ No assumptions about distributions
- ✅ Want information-theoretic measure
- ✅ Robustness to outliers needed

### Use Random Forest when:
- ✅ Feature interactions matter
- ✅ Ensemble-based selection desired
- ✅ Empirical performance is priority
- ✅ Handling complex non-linear patterns

### Use PCA when:
- ✅ Multicollinearity is problem
- ✅ Need decorrelated features
- ✅ Working with GMM (Gaussian assumption)
- ✅ Visualization needed (2D/3D plots)

### Use Correlation when:
- ✅ Identifying redundant feature pairs
- ✅ Understanding feature relationships
- ✅ Pre-processing before other methods

---

## Our Strategy: Union of All Methods

**Why combine all three feature selection methods?**

1. **Complementary Coverage:**
   - ANOVA: Linear separability
   - MI: Non-linear patterns
   - RF: Feature interactions

2. **Robustness:**
   - No single method is perfect
   - Different methods may miss different features
   - Union ensures comprehensive coverage

3. **Validation:**
   - Features selected by multiple methods = highly reliable
   - Features selected by one method = might still be valuable

4. **Trade-off:**
   - More features (191) vs. individual methods (100 each)
   - Better coverage vs. computational cost
   - Balance: Use 100-191 features depending on model

**Result:**
- **100 features per method**
- **191 unique features in union**
- **~66% overlap** among methods (high agreement on important features)

---

## References

1. **ANOVA:** Fisher, R.A. (1925). Statistical Methods for Research Workers
2. **Mutual Information:** Cover & Thomas (2006). Elements of Information Theory
3. **Random Forest:** Breiman, L. (2001). Random Forests. Machine Learning
4. **PCA:** Jolliffe, I.T. (2002). Principal Component Analysis
5. **Feature Selection:** Guyon & Elisseeff (2003). An Introduction to Variable and Feature Selection
