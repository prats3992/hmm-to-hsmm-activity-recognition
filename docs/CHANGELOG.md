# Project Changelog

## What We Did and Why

### Initial Approach (First Iteration)

**What we did:**
- Selected top 50 features from 561
- Treated Activity as just another column

**Problems identified:**
1. ❌ **50 features too aggressive** - might miss important patterns with 561 original features
2. ❌ **Activity treated as feature** - but it's actually the TARGET VARIABLE we're trying to predict!
3. ❌ **Subject not preserved** - needed for sequential modeling

### Fix #1: Preserve Target Variable

**Problem:** 
Initially, feature selection might accidentally discard the Activity column, treating it like any other feature.

**Why this is wrong:**
- **Activity is what we're trying to predict** (target variable)
- HMM needs to learn P(Activity | Sensor features)
- Markov Chain models transitions between Activities
- GMM clusters sensor data, but Activity validates the clustering

**Solution:**
- **Always preserve Activity column separately**
- **Always preserve subject column** (for sequential modeling)
- Only perform feature selection on the 561 sensor features

**Code change:**
```python
# Before (WRONG)
all_columns = df.columns  # Might include Activity

# After (CORRECT)
feature_cols = [col for col in df.columns if col not in ['subject', 'Activity']]
X = df[feature_cols]  # Only sensor features
y = df['Activity']     # Target - always preserved
subject = df['subject'] # ID - always preserved
```

### Fix #2: Increase Number of Selected Features

**Problem:**
50 features out of 561 might be too aggressive (only 9% retained).

**Why this is a problem:**
- 561 features cover diverse motion patterns
- 6 activity classes with subtle differences (SITTING vs STANDING)
- Risk of missing discriminative features
- Static activities (sitting/standing/laying) may need different features than dynamic (walking)

**Solution:**
- **Increased to 100 features per method** (~18% of original)
- **Union of all three methods gives 191 features** (~34% of original)
- More conservative approach
- Balance between dimensionality reduction and information preservation

**Justification:**
- ANOVA: Top 100 features
- Mutual Information: Top 100 features  
- Random Forest: Top 100 features
- **Union:** 191 unique features (66% overlap shows consistency)

### Fix #3: Better Documentation

**Problem:**
Initial code lacked clear explanations of:
- Why each method was used
- What mathematical concepts mean
- How to interpret results
- When to use which approach

**Solution:**
- Created comprehensive README.md with:
  - Step-by-step explanations
  - Mathematical formulas with interpretations
  - Decision rationale
  - Usage examples
  
- Created METHODS_REFERENCE.md with:
  - Detailed formulas
  - When to use each method
  - Advantages and limitations
  - Comparison tables

### Fix #4: Organized Directory Structure

**Before:**
```
PROJECT/
├── train.csv
├── test.csv
├── preprocessing_eda.py
├── data_loader.py
├── activity_distribution.png
├── feature_importance.png
└── ... (messy!)
```

**After:**
```
PROJECT/
├── data/                    # All datasets
├── scripts/                 # All Python code
├── outputs/
│   ├── images/             # All visualizations
│   └── results/            # JSON, CSV outputs
├── docs/                    # Documentation
└── README.md               # Main documentation
```

**Why this matters:**
- ✅ Easy to find files
- ✅ Clear separation of concerns
- ✅ Professional project structure
- ✅ Easy to share and collaborate

---

## Summary of Changes

| Issue | Before | After | Why |
|-------|--------|-------|-----|
| **Activity column** | Treated as feature | Always preserved as target | It's what we predict! |
| **Subject column** | Not explicitly handled | Always preserved | Needed for sequences |
| **Number of features** | 50 | 100-191 | More conservative |
| **Documentation** | Minimal | Comprehensive | Understand decisions |
| **Directory structure** | Flat | Organized | Professional |
| **Path references** | `train.csv` | `data/train.csv` | Organized structure |
| **Output location** | Root directory | `outputs/images/` | Clean separation |

---

## Lessons Learned

### 1. Understand Your Problem Type

**Mistake:** Treated this like standard supervised classification.

**Reality:** This is sequential/temporal modeling with stochastic processes.

**Key difference:**
- Classification: Features → Label (independent samples)
- Sequential: Features → Hidden State Sequence (temporal dependencies)

### 2. Domain Knowledge Matters

**Understanding the data:**
- 50Hz sampling → consecutive samples are similar
- Activity transitions are rare
- Gravity features indicate orientation (sitting vs standing)
- Jerk features indicate dynamic motion (walking)

**This informed:**
- Why transition matrix has high diagonal (self-transitions)
- Why gravity features rank high in importance
- Why we need temporal grouping by subject

### 3. Feature Selection is Not One-Size-Fits-All

**Different models need different features:**
- **HMM:** Prefer original features (interpretable observations)
- **Markov Chain:** Only needs Activity labels (no sensor features)
- **GMM:** Can use PCA (Gaussian assumption works better in lower dimensions)

**Solution:** Provide multiple options in data loader

### 4. Always Validate Your Assumptions

**Initial assumption:** "50 features should be enough"

**Validation:** 
- Checked if important features missed
- Compared overlap between methods (66% agreement)
- Considered problem complexity (6 classes, 561 features)

**Result:** Increased to 100 features

### 5. Documentation is Part of the Solution

**Code alone is not enough:**
- Future you will forget why decisions were made
- Collaborators need context
- Mathematical concepts need explanation

**Good documentation includes:**
- ✅ What you did
- ✅ Why you did it
- ✅ What problems you encountered
- ✅ How you fixed them
- ✅ Mathematical foundations

---

## Next Time: Best Practices

1. **Identify target variable first** - never treat it as a feature
2. **Understand your modeling paradigm** - sequential vs. independent
3. **Start conservative** - easier to reduce than to add back
4. **Document as you go** - don't wait until the end
5. **Organize from the start** - directory structure matters
6. **Validate assumptions** - check if decisions make sense
7. **Explain math** - formulas + interpretation
8. **Provide examples** - show how to use your code

---

## References to Key Changes

### Code Files Changed:
1. `scripts/preprocessing_eda.py`
   - Updated feature selection count: 50 → 100
   - Updated paths: `train.csv` → `data/train.csv`
   - Updated output paths: `*.png` → `outputs/images/*.png`
   - Added clarification comments about Activity preservation

2. `scripts/data_loader.py`
   - Updated default paths
   - Added clear separation of features vs. target
   - Documented that Activity is always preserved

### Documentation Created:
1. `README.md` - Main project documentation
2. `docs/METHODS_REFERENCE.md` - Mathematical reference
3. `docs/CHANGELOG.md` - This file
4. `.gitignore` - Clean git repository

### Directory Structure:
- Created `data/`, `scripts/`, `outputs/`, `docs/`
- Moved all files to appropriate locations
- Updated all path references in code

---

Last Updated: November 15, 2025
