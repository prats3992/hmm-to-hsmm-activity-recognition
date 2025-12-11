# Human Activity Recognition Using Hidden Markov Models
## Final Project Report

**Authors:** Pratham Arora, Vaisakh Menon  
**Date:** December 3, 2025  
**Course:** Stochastic Process for Data Science and Engineering

---

## 1. Introduction
This project implements a **Hidden Markov Model (HMM)** framework to recognize human activities (Walking, Standing, Sitting, Laying, etc.) from smartphone sensor data. Unlike traditional static classifiers (e.g., Random Forest, SVM) that treat each time step independently, HMMs explicitly model the **temporal dynamics** of human behavior—leveraging the fact that the current activity depends on the previous one.

We explored three model variants:
1.  **Gaussian HMM:** Standard HMM with single Gaussian emissions.
2.  **GMM-HMM:** HMM with Gaussian Mixture Model emissions (to handle non-Gaussian data).
3.  **HSMM (Hidden Semi-Markov Model):** An extension that explicitly models the *duration* of each activity.

---

## 2. Methodology

### 2.1 Data Preprocessing
The dataset (UCI-HAR) contains 561 features derived from accelerometer and gyroscope readings.
*   **Standardization:** All features were scaled to zero mean and unit variance.
*   **Dimensionality Reduction (PCA):** We applied Principal Component Analysis (PCA) to reduce the 561 features to **65 components**, retaining **90% of the variance**. This was crucial to prevent overfitting and ensure numerical stability in the covariance matrices.

### 2.2 Mathematical Formulation

#### The Hidden Markov Model
An HMM is defined by $\lambda = (A, B, \pi)$:
*   **States ($S$):** The hidden activities $\{1, \dots, N\}$.
*   **Observations ($O$):** The continuous sensor data vectors $\mathbf{x}_t \in \mathbb{R}^{65}$.
*   **Transition Matrix ($A$):** $A_{ij} = P(S_t = j | S_{t-1} = i)$.
*   **Emission Probability ($B$):** $b_j(\mathbf{x}_t) = P(\mathbf{x}_t | S_t = j)$.

#### Emission Modeling (GMM)
Since the sensor data proved to be non-Gaussian (see Section 3), we modeled the emissions using a **Gaussian Mixture Model (GMM)**:
$$ b_j(\mathbf{x}) = \sum_{k=1}^{M} c_{jk} \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_{jk}, \boldsymbol{\Sigma}_{jk}) $$
where $M=2$ mixtures were used per state.

#### Decoding (Viterbi Algorithm)
To find the most likely sequence of activities $S_{1:T}^*$ given observations $O_{1:T}$, we used the **Viterbi Algorithm**:
$$ \delta_t(j) = \max_{i} [\delta_{t-1}(i) A_{ij}] b_j(\mathbf{x}_t) $$

#### HSMM (Explicit Duration)
Standard HMMs imply a geometric distribution for state duration ($P(d) \propto A_{ii}^{d-1}$), which is unrealistic for human activities. Our **HSMM** replaces this with an explicit duration distribution $P(D_j = d)$, modeled as a **Gaussian**:
$$ P(D_j = d) \approx \mathcal{N}(d; \mu_j, \sigma_j^2) $$
The decoding is performed via the **Duration-Viterbi Algorithm**:
$$ \delta_t(j) = \max_{d, i \neq j} \left[ \delta_{t-d}(i) + \log P(D_j=d) + \sum_{\tau=t-d+1}^t \log b_j(\mathbf{x}_\tau) \right] $$

---

## 3. Assumption Verification

We rigorously tested the statistical assumptions underlying our models.

### 3.1 Gaussianity of Emissions
*   **Test:** Shapiro-Wilk & AIC Comparison.
*   **Result:** **FAILED** for single Gaussian.
*   **Solution:** Switched to **GMM (2 components)**.
*   **Evidence:** AIC improved by **+1340** for the 'Laying' class.

### 3.2 Conditional Independence
*   **Assumption:** Observations are independent given the state.
*   **Test:** Autocorrelation Function (ACF) of residuals.
*   **Result:** **FAILED**. High autocorrelation (Lag-1 ACF $\approx 0.9$).
*   **Implication:** Standard HMM ignores this, but the high self-transition probabilities ($A_{ii} \approx 0.99$) compensate for it.

### 3.3 Markov Property
*   **Test:** Likelihood Ratio Test (1st Order vs 2nd Order).
*   **Result:**
    *   **Frame-Level:** **FAILED** ($p < 10^{-16}$). History matters.
    *   **Embedded Level (HSMM):** **PASSED** ($p > 0.05$). The sequence of *transitions* (e.g., Walk $\to$ Sit) is Markovian.
*   **Conclusion:** This justifies the use of **HSMM**, which models the embedded chain directly.

---

## 4. Results & Comparison

We evaluated the models on three test subjects (2, 9, 12) and compared them against a **Random Forest** baseline.

| Subject | Naive Bayes | Random Forest | GMM-HMM | HSMM | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Subject 2** | 89.40% | 93.05% | **93.71%** | 92.38% | **GMM-HMM** |
| **Subject 9** | 77.78% | **86.11%** | **86.11%** | 85.76% | **Tie (RF/HMM)** |
| **Subject 12** | 83.75% | 87.81% | 91.87% | **93.44%** | **HSMM (+5.6%)** |

### Key Findings
1.  **HMM/HSMM Outperforms Baseline:** On Subject 12, the temporal modeling provided a massive **5.6% accuracy boost** over Random Forest.
2.  **HSMM vs HMM:** HSMM is competitive and sometimes superior (Subj 12), proving that explicit duration modeling is valuable when activity durations are consistent.
3.  **Robustness:** Even though some statistical assumptions failed, the probabilistic models (HMM/HSMM) remained robust and effective.

---

## 5. Q&A (Presentation Prep)

**Q1: Why did you use PCA?**
*   **A:** The raw data has 561 features, many of which are highly correlated. Using all of them would cause the covariance matrices in the GMM to be singular (non-invertible) or unstable. PCA reduced this to 65 orthogonal components while keeping 90% of the information.

**Q2: Why did the Markov assumption fail?**
*   **A:** At the frame level (50Hz), human motion is continuous and smooth. Your position at $t$ depends on velocity at $t-1$ and acceleration at $t-2$. A simple 1st-order chain cannot capture this physical inertia. However, the *sequence of activities* (Embedded Chain) was proven to be Markovian.

**Q3: Why not use a Kalman Filter?**
*   **A:** Kalman Filters assume **continuous** hidden states (like position/velocity) and linear dynamics. Our hidden states are **discrete categories** (Walking, Sitting). HMM is the correct tool for discrete latent variables.

**Q4: Why did HSMM perform better on Subject 12 but worse on Subject 2?**
*   **A:** HSMM imposes a strict Gaussian duration. If a subject's activity duration varies wildly (high variance), the rigid Gaussian penalty might hurt. Subject 12 likely had very regular activity patterns that fit the Gaussian model well.

**Q5: What is the advantage of HMM over Random Forest?**
*   **A:** Random Forest classifies every millisecond independently. It produces "flickering" predictions (e.g., Walk-Walk-Sit-Walk). HMMs smooth these out because the transition matrix makes $P(\text{Walk} \to \text{Sit})$ very low compared to $P(\text{Walk} \to \text{Walk})$.

---

## 6. Conclusion
We successfully implemented and verified a GMM-HMM and HSMM for human activity recognition. Our results demonstrate that **modeling temporal dynamics improves accuracy**, particularly for subjects with structured activity patterns, outperforming strong static baselines like Random Forest.
