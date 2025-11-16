"""
Hidden Markov Model Implementation for Human Activity Recognition

This module implements a discrete-time HMM with Gaussian emissions for activity recognition.
Includes:
- Model formulation (Section 4.2)
- Baum-Welch (EM) algorithm for parameter estimation (Section 4.4)
- Viterbi algorithm for inference (Section 4.5)
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple, List, Optional
import warnings

class GaussianHMM:
    """
    Discrete-Time Hidden Markov Model with Gaussian Emissions
    
    States: 6 discrete activity labels (hidden)
    Observations: Continuous sensor feature vectors (observed)
    
    Parameters:
    -----------
    n_states : int
        Number of hidden states (activities)
    n_features : int
        Dimensionality of observation vectors
    covariance_type : str
        Type of covariance matrix ('full' or 'diag')
        - 'full': Full covariance matrices (n_features x n_features)
        - 'diag': Diagonal covariance matrices (computationally efficient)
    """
    
    def __init__(self, n_states: int = 6, n_features: int = 100, 
                 covariance_type: str = 'diag', random_state: int = 42):
        self.n_states = n_states
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Model parameters λ = (A, B, π)
        self.A = None  # Transition matrix (n_states x n_states)
        self.means = None  # Emission means (n_states x n_features)
        self.covars = None  # Emission covariances
        self.pi = None  # Initial state distribution (n_states,)
        
        # Training history
        self.log_likelihoods = []
        self.converged = False
        
    def _initialize_parameters(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize HMM parameters using supervised data
        
        Better than random initialization because we have labeled training data.
        Uses k-means-like approach: compute empirical means/covariances per activity.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Observation sequences
        y : np.ndarray, shape (n_samples,)
            Activity labels (0 to n_states-1)
        """
        n_samples = X.shape[0]
        
        # Initialize transition matrix with small random values + diagonal bias
        # Diagonal elements should be high (self-transitions are common)
        self.A = np.random.dirichlet(np.ones(self.n_states) * 0.5, size=self.n_states)
        self.A = 0.7 * np.eye(self.n_states) + 0.3 * self.A
        
        # Normalize rows to ensure row-stochastic property
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        # Initialize emission parameters from data statistics
        self.means = np.zeros((self.n_states, self.n_features))
        
        if self.covariance_type == 'diag':
            self.covars = np.zeros((self.n_states, self.n_features))
        else:
            self.covars = np.zeros((self.n_states, self.n_features, self.n_features))
        
        # Compute per-class statistics
        for state in range(self.n_states):
            mask = (y == state)
            X_state = X[mask]
            
            if len(X_state) > 0:
                self.means[state] = X_state.mean(axis=0)
                
                if self.covariance_type == 'diag':
                    # Diagonal covariance (feature independence assumption)
                    var = X_state.var(axis=0)
                    # Stronger regularization for numerical stability
                    self.covars[state] = np.maximum(var, 1e-4) + 1e-3
                else:
                    # Full covariance matrix
                    cov = np.cov(X_state.T)
                    # Regularize to ensure positive definiteness
                    self.covars[state] = cov + 1e-3 * np.eye(self.n_features)
            else:
                warnings.warn(f"State {state} has no samples, using random initialization")
                self.means[state] = X.mean(axis=0) + np.random.randn(self.n_features) * 0.1
                if self.covariance_type == 'diag':
                    self.covars[state] = np.ones(self.n_features)
                else:
                    self.covars[state] = np.eye(self.n_features)
        
        # Initialize with empirical state distribution
        state_counts = np.bincount(y, minlength=self.n_states)
        self.pi = (state_counts + 1) / (state_counts.sum() + self.n_states)  # Laplace smoothing
        
    def _emission_probability(self, x: np.ndarray, state: int) -> float:
        """
        Compute P(x_t | S_t = state) using multivariate Gaussian
        
        P(x | S=j) = N(x; μ_j, Σ_j)
        
        Parameters:
        -----------
        x : np.ndarray, shape (n_features,)
            Observation vector
        state : int
            Hidden state index
            
        Returns:
        --------
        prob : float
            Emission probability
        """
        mean = self.means[state]
        
        if self.covariance_type == 'diag':
            # Diagonal covariance: faster computation
            cov = np.maximum(self.covars[state], 1e-6)  # Ensure positive
            # Use univariate formula for each dimension then multiply
            diff = x - mean
            exponent = -0.5 * np.sum((diff ** 2) / cov)
            normalizer = -0.5 * (self.n_features * np.log(2 * np.pi) + np.sum(np.log(cov)))
            log_prob = np.clip(normalizer + exponent, -700, 700)  # Prevent overflow
            return np.exp(log_prob)
        else:
            # Full covariance: use scipy's multivariate_normal
            try:
                return multivariate_normal.pdf(x, mean=mean, cov=self.covars[state])
            except:
                # Fallback if covariance becomes singular
                return 1e-10
    
    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm: compute α_t(j) = P(x_1, ..., x_t, S_t = j | λ)
        
        Initialization: α_1(j) = π_j * P(x_1 | S_1 = j)
        Recursion: α_t(j) = [Σ_i α_{t-1}(i) * A_{ij}] * P(x_t | S_t = j)
        
        Parameters:
        -----------
        X : np.ndarray, shape (T, n_features)
            Observation sequence
            
        Returns:
        --------
        alpha : np.ndarray, shape (T, n_states)
            Forward probabilities
        log_likelihood : float
            Log P(X | λ)
        """
        T = X.shape[0]
        alpha = np.zeros((T, self.n_states))
        
        # Initialization (t=1)
        for j in range(self.n_states):
            alpha[0, j] = self.pi[j] * self._emission_probability(X[0], j)
        
        # Scaling to prevent underflow
        scale_factors = np.zeros(T)
        scale_factors[0] = alpha[0].sum() + 1e-10  # Prevent division by zero
        if scale_factors[0] == 0 or not np.isfinite(scale_factors[0]):
            scale_factors[0] = 1.0
        alpha[0] /= scale_factors[0]
        
        # Recursion (t=2 to T)
        for t in range(1, T):
            for j in range(self.n_states):
                # Sum over all possible previous states
                transition_prob = np.sum(alpha[t-1] * self.A[:, j])
                emission_prob = self._emission_probability(X[t], j)
                alpha[t, j] = transition_prob * emission_prob
            
            # Scale to prevent underflow with additional safeguards
            scale_factors[t] = alpha[t].sum() + 1e-10
            if scale_factors[t] == 0 or not np.isfinite(scale_factors[t]) or not np.all(np.isfinite(alpha[t])):
                # Reset to uniform if numerical issues
                alpha[t] = np.ones(self.n_states) / self.n_states
                scale_factors[t] = 1.0
            else:
                alpha[t] /= scale_factors[t]
        
        # Compute log-likelihood using scale factors
        log_likelihood = np.sum(np.log(scale_factors + 1e-10))
        
        return alpha, log_likelihood
    
    def _backward(self, X: np.ndarray, scale_factors: np.ndarray) -> np.ndarray:
        """
        Backward algorithm: compute β_t(i) = P(x_{t+1}, ..., x_T | S_t = i, λ)
        
        Initialization: β_T(i) = 1 for all i
        Recursion: β_t(i) = Σ_j A_{ij} * P(x_{t+1} | S_{t+1} = j) * β_{t+1}(j)
        
        Parameters:
        -----------
        X : np.ndarray, shape (T, n_features)
            Observation sequence
        scale_factors : np.ndarray, shape (T,)
            Scaling factors from forward algorithm
            
        Returns:
        --------
        beta : np.ndarray, shape (T, n_states)
            Backward probabilities
        """
        T = X.shape[0]
        beta = np.zeros((T, self.n_states))
        
        # Initialization (t=T)
        beta[-1, :] = 1.0
        
        # Recursion (t=T-1 to 1, working backward)
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    beta[t, i] += (self.A[i, j] * 
                                  self._emission_probability(X[t+1], j) * 
                                  beta[t+1, j])
            
            # Apply same scaling as forward algorithm
            if scale_factors[t+1] > 0:
                beta[t] /= scale_factors[t+1]
        
        return beta
    
    def _compute_gamma_xi(self, X: np.ndarray, alpha: np.ndarray, 
                         beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        E-step: Compute expected sufficient statistics
        
        γ_t(i) = P(S_t = i | X, λ) = [α_t(i) * β_t(i)] / P(X | λ)
        ξ_t(i,j) = P(S_t = i, S_{t+1} = j | X, λ)
        
        Parameters:
        -----------
        X : np.ndarray, shape (T, n_features)
            Observation sequence
        alpha : np.ndarray, shape (T, n_states)
            Forward probabilities
        beta : np.ndarray, shape (T, n_states)
            Backward probabilities
            
        Returns:
        --------
        gamma : np.ndarray, shape (T, n_states)
            State occupation probabilities
        xi : np.ndarray, shape (T-1, n_states, n_states)
            State transition probabilities
        """
        T = X.shape[0]
        
        # Compute gamma (state occupation probability)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10
        
        # Compute xi (state transition probability)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                  self._emission_probability(X[t+1], j) * 
                                  beta[t+1, j])
            
            # Normalize
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum
        
        return gamma, xi
    
    def _m_step(self, X_list: List[np.ndarray], gamma_list: List[np.ndarray], 
                xi_list: List[np.ndarray]):
        """
        M-step: Update parameters to maximize expected log-likelihood
        
        Updates:
        - π_i = γ_1(i)
        - A_{ij} = [Σ_t ξ_t(i,j)] / [Σ_t γ_t(i)]
        - μ_j = [Σ_t γ_t(j) * x_t] / [Σ_t γ_t(j)]
        - Σ_j = [Σ_t γ_t(j) * (x_t - μ_j)(x_t - μ_j)^T] / [Σ_t γ_t(j)]
        
        Parameters:
        -----------
        X_list : List[np.ndarray]
            List of observation sequences
        gamma_list : List[np.ndarray]
            List of gamma values for each sequence
        xi_list : List[np.ndarray]
            List of xi values for each sequence
        """
        # Update initial distribution
        self.pi = np.mean([gamma[0] for gamma in gamma_list], axis=0)
        self.pi /= self.pi.sum()
        
        # Update transition matrix
        numerator = np.zeros((self.n_states, self.n_states))
        denominator = np.zeros(self.n_states)
        
        for xi, gamma in zip(xi_list, gamma_list):
            numerator += xi.sum(axis=0)
            denominator += gamma[:-1].sum(axis=0)
        
        # Add small constant to avoid division by zero
        self.A = numerator / (denominator[:, np.newaxis] + 1e-10)
        
        # Check for NaN and replace with uniform if needed
        if not np.all(np.isfinite(self.A)):
            print("  Warning: NaN detected in transition matrix, using uniform distribution")
            self.A = np.ones((self.n_states, self.n_states)) / self.n_states
        
        # Ensure row-stochastic property
        row_sums = self.A.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        self.A = self.A / row_sums
        
        # Update emission parameters
        for j in range(self.n_states):
            # Compute weighted mean
            numerator_mean = np.zeros(self.n_features)
            denominator_mean = 0
            
            for X, gamma in zip(X_list, gamma_list):
                numerator_mean += np.sum(gamma[:, j:j+1] * X, axis=0)
                denominator_mean += gamma[:, j].sum()
            
            self.means[j] = numerator_mean / (denominator_mean + 1e-10)
            
            # Compute weighted covariance
            if self.covariance_type == 'diag':
                numerator_cov = np.zeros(self.n_features)
                
                for X, gamma in zip(X_list, gamma_list):
                    diff = X - self.means[j]
                    numerator_cov += np.sum(gamma[:, j:j+1] * (diff ** 2), axis=0)
                
                # Stronger regularization and minimum variance
                self.covars[j] = np.maximum(numerator_cov / (denominator_mean + 1e-10), 1e-4) + 1e-3
            else:
                numerator_cov = np.zeros((self.n_features, self.n_features))
                
                for X, gamma in zip(X_list, gamma_list):
                    diff = X - self.means[j]
                    numerator_cov += np.dot((gamma[:, j:j+1] * diff).T, diff)
                
                self.covars[j] = numerator_cov / (denominator_mean + 1e-10)
                # Regularize
                self.covars[j] += 1e-3 * np.eye(self.n_features)
            
            # Validate - if NaN detected, reset to reasonable values
            if not np.all(np.isfinite(self.means[j])):
                print(f"  Warning: NaN in mean for state {j}, resetting")
                self.means[j] = np.zeros(self.n_features)
            if not np.all(np.isfinite(self.covars[j])):
                print(f"  Warning: NaN in covariance for state {j}, resetting")
                if self.covariance_type == 'diag':
                    self.covars[j] = np.ones(self.n_features)
                else:
                    self.covars[j] = np.eye(self.n_features)
    
    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray], 
            n_iter: int = 50, tol: float = 1e-4, verbose: bool = True) -> 'GaussianHMM':
        """
        Train HMM using Baum-Welch (EM) algorithm
        
        Parameters:
        -----------
        X_list : List[np.ndarray]
            List of observation sequences (one per subject)
            Each array has shape (T_i, n_features)
        y_list : List[np.ndarray]
            List of label sequences for initialization
            Each array has shape (T_i,)
        n_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence threshold for log-likelihood improvement
        verbose : bool
            Print progress information
            
        Returns:
        --------
        self : GaussianHMM
            Fitted model
        """
        # Concatenate all sequences for initialization
        X_concat = np.vstack(X_list)
        y_concat = np.hstack(y_list)
        
        # Initialize parameters
        self._initialize_parameters(X_concat, y_concat)
        
        if verbose:
            print(f"Initialized HMM with {self.n_states} states, {self.n_features} features")
            print(f"Covariance type: {self.covariance_type}")
            print(f"Training on {len(X_list)} sequences, {X_concat.shape[0]} total samples")
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        
        for iteration in range(n_iter):
            # E-step: compute forward-backward for all sequences
            alpha_list = []
            beta_list = []
            gamma_list = []
            xi_list = []
            scale_factors_list = []
            total_log_likelihood = 0
            
            for X in X_list:
                # Forward pass
                alpha, log_likelihood = self._forward(X)
                alpha_list.append(alpha)
                total_log_likelihood += log_likelihood
                
                # Compute scale factors for backward pass
                scale_factors = np.zeros(X.shape[0])
                for t in range(X.shape[0]):
                    scale_factors[t] = alpha[t].sum()
                scale_factors_list.append(scale_factors)
                
                # Backward pass
                beta = self._backward(X, scale_factors)
                beta_list.append(beta)
                
                # Compute gamma and xi
                gamma, xi = self._compute_gamma_xi(X, alpha, beta)
                gamma_list.append(gamma)
                xi_list.append(xi)
            
            self.log_likelihoods.append(total_log_likelihood)
            
            # Check convergence
            improvement = total_log_likelihood - prev_log_likelihood
            
            if verbose:
                print(f"Iteration {iteration+1}/{n_iter}: Log-likelihood = {total_log_likelihood:.2f}, "
                      f"Improvement = {improvement:.4f}")
            
            if improvement < tol and iteration > 0:
                if verbose:
                    print(f"Converged after {iteration+1} iterations")
                self.converged = True
                break
            
            # M-step: update parameters
            self._m_step(X_list, gamma_list, xi_list)
            
            prev_log_likelihood = total_log_likelihood
        
        if not self.converged and verbose:
            print(f"Reached maximum iterations ({n_iter}) without convergence")
        
        return self
    
    def viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm: find most likely state sequence
        
        S* = argmax_S P(S | X, λ)
        
        Uses dynamic programming:
        δ_t(j) = max_{S_1,...,S_{t-1}} P(S_1,...,S_t=j, x_1,...,x_t | λ)
        
        Parameters:
        -----------
        X : np.ndarray, shape (T, n_features)
            Observation sequence
            
        Returns:
        --------
        states : np.ndarray, shape (T,)
            Most likely state sequence
        log_probability : float
            Log probability of the most likely path
        """
        T = X.shape[0]
        
        # Initialize delta and psi (backpointers)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization (t=1)
        for j in range(self.n_states):
            delta[0, j] = np.log(self.pi[j] + 1e-10) + np.log(self._emission_probability(X[0], j) + 1e-10)
        
        # Recursion (t=2 to T)
        for t in range(1, T):
            for j in range(self.n_states):
                # Find best previous state
                temp = delta[t-1] + np.log(self.A[:, j] + 1e-10)
                psi[t, j] = np.argmax(temp)
                delta[t, j] = temp[psi[t, j]] + np.log(self._emission_probability(X[t], j) + 1e-10)
        
        # Termination: find best final state
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        log_probability = delta[-1, states[-1]]
        
        # Backtracking (t=T-1 to 1)
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states, log_probability
    
    def predict(self, X_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict most likely state sequences for multiple observation sequences
        
        Parameters:
        -----------
        X_list : List[np.ndarray]
            List of observation sequences
            
        Returns:
        --------
        predictions : List[np.ndarray]
            List of predicted state sequences
        """
        predictions = []
        for X in X_list:
            states, _ = self.viterbi(X)
            predictions.append(states)
        return predictions
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of observation sequence
        
        Parameters:
        -----------
        X : np.ndarray, shape (T, n_features)
            Observation sequence
            
        Returns:
        --------
        log_likelihood : float
            Log P(X | λ)
        """
        _, log_likelihood = self._forward(X)
        return log_likelihood
    
    def get_transition_matrix(self) -> np.ndarray:
        """Return the learned transition matrix A"""
        return self.A.copy()
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution π* such that π* = π* A
        
        This represents the long-term probability of being in each state
        assuming the Markov chain has converged.
        
        Returns:
        --------
        stationary : np.ndarray, shape (n_states,)
            Stationary distribution
        """
        # Check if A contains NaN or Inf
        if not np.all(np.isfinite(self.A)):
            # Return uniform distribution if A is not valid
            return np.ones(self.n_states) / self.n_states
        
        try:
            # Find eigenvector corresponding to eigenvalue 1
            eigenvalues, eigenvectors = np.linalg.eig(self.A.T)
            
            # Find index of eigenvalue closest to 1
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            
            # Get corresponding eigenvector
            stationary = np.real(eigenvectors[:, idx])
            
            # Normalize to get probability distribution
            if np.sum(np.abs(stationary)) > 1e-10:
                stationary = np.abs(stationary) / np.sum(np.abs(stationary))
            else:
                stationary = np.ones(self.n_states) / self.n_states
                
            return stationary
        except:
            # Fallback to uniform distribution on error
            return np.ones(self.n_states) / self.n_states
        
        return stationary
