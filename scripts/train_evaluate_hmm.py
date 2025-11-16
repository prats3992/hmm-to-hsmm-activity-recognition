"""
HMM Training and Evaluation Pipeline

Implements complete workflow:
1. Load preprocessed data
2. Run assumption verification tests
3. Train Gaussian HMM using Baum-Welch
4. Decode sequences using Viterbi
5. Evaluate performance with metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import json
import pickle
from typing import Dict, List, Tuple
import sys
import time

# Import custom modules
sys.path.append('.')
from scripts.data_loader import StochasticDataLoader
from scripts.hmm_model import GaussianHMM
from scripts.assumption_tests import AssumptionVerifier


class HMMEvaluator:
    """
    Complete evaluation pipeline for HMM activity recognition
    """
    
    def __init__(self, output_dir: str = 'outputs/results'):
        self.output_dir = output_dir
        self.activity_names = ['LAYING', 'SITTING', 'STANDING', 
                              'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        self.loader = StochasticDataLoader()
        self.label_mapping = None  # Will store integer->string mapping
        
    def prepare_sequences(self, method: str = 'anova') -> Tuple:
        """
        Load and prepare sequential data for HMM training
        
        Parameters:
        -----------
        method : str
            Feature selection method ('anova', 'mutual_info', 'random_forest')
            
        Returns:
        --------
        train_sequences : List[Tuple]
            List of (observations, labels, subject_id) for training
        test_sequences : List[Tuple]
            List of (observations, labels, subject_id) for testing
        n_features : int
            Number of features
        """
        print("\n" + "="*80)
        print("LOADING AND PREPARING SEQUENTIAL DATA")
        print("="*80)
        
        # Load sequences grouped by subject
        train_seq, test_seq = self.loader.prepare_for_hmm(method=method)
        
        # Store label mapping for later use
        self.label_mapping = self.loader.activity_mapping
        # Update activity_names to match encoded order
        self.activity_names = [self.label_mapping[i] for i in range(len(self.label_mapping))]
        
        print(f"\nFeature selection method: {method.upper()}")
        print(f"Training sequences: {len(train_seq)} subjects")
        print(f"Test sequences: {len(test_seq)} subjects")
        
        # Compute statistics
        train_lengths = [seq[0].shape[0] for seq in train_seq]
        test_lengths = [seq[0].shape[0] for seq in test_seq]
        n_features = train_seq[0][0].shape[1]
        
        print(f"\nSequence lengths:")
        print(f"  Training - Mean: {np.mean(train_lengths):.1f}, "
              f"Min: {np.min(train_lengths)}, Max: {np.max(train_lengths)}")
        print(f"  Test - Mean: {np.mean(test_lengths):.1f}, "
              f"Min: {np.min(test_lengths)}, Max: {np.max(test_lengths)}")
        print(f"\nFeature dimensionality: {n_features}")
        print(f"Activity encoding: {self.label_mapping}")
        
        return train_seq, test_seq, n_features
    
    def run_assumption_tests(self, train_sequences: List[Tuple], 
                            feature_names: List[str]):
        """
        Run comprehensive assumption verification tests
        
        Parameters:
        -----------
        train_sequences : List[Tuple]
            Training sequences
        feature_names : List[str]
            Names of features
        """
        print("\n" + "="*80)
        print("RUNNING ASSUMPTION VERIFICATION TESTS (Section 4.3)")
        print("="*80)
        
        # Concatenate all training data for testing
        X_concat = np.vstack([seq[0] for seq in train_sequences])
        y_concat = np.hstack([seq[1] for seq in train_sequences])
        
        # Run tests
        verifier = AssumptionVerifier(output_dir=self.output_dir)
        test_results = verifier.run_all_tests(
            X_concat, y_concat, self.activity_names, feature_names
        )
        
        # Save results
        results_file = f'{self.output_dir}/assumption_test_results.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(test_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n→ Saved test results to {results_file}")
        
        return test_results
    
    def train_hmm(self, train_sequences: List[Tuple], n_features: int,
                  covariance_type: str = 'diag', n_iter: int = 50,
                  verbose: bool = True) -> GaussianHMM:
        """
        Train HMM using Baum-Welch algorithm (Section 4.4)
        
        Parameters:
        -----------
        train_sequences : List[Tuple]
            Training sequences (observations, labels, subject_id)
        n_features : int
            Feature dimensionality
        covariance_type : str
            'diag' or 'full'
        n_iter : int
            Maximum EM iterations
        verbose : bool
            Print training progress
            
        Returns:
        --------
        hmm : GaussianHMM
            Trained model
        """
        print("\n" + "="*80)
        print("TRAINING HMM WITH BAUM-WELCH ALGORITHM (Section 4.4)")
        print("="*80)
        
        # Extract observations and labels
        X_list = [seq[0] for seq in train_sequences]
        y_list = [seq[1] for seq in train_sequences]
        
        # Initialize model
        hmm = GaussianHMM(
            n_states=6,
            n_features=n_features,
            covariance_type=covariance_type,
            random_state=42
        )
        
        # Train
        start_time = time.time()
        hmm.fit(X_list, y_list, n_iter=n_iter, tol=1e-4, verbose=verbose)
        train_time = time.time() - start_time
        
        print(f"\n→ Training completed in {train_time:.2f} seconds")
        print(f"→ Converged: {hmm.converged}")
        print(f"→ Final log-likelihood: {hmm.log_likelihoods[-1]:.2f}")
        
        # Plot training curve
        self._plot_training_curve(hmm.log_likelihoods)
        
        # Save model
        model_file = f'{self.output_dir}/trained_hmm_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(hmm, f)
        print(f"→ Saved model to {model_file}")
        
        return hmm
    
    def _plot_training_curve(self, log_likelihoods: List[float]):
        """Plot EM training convergence curve"""
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(log_likelihoods) + 1)
        plt.plot(iterations, log_likelihoods, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Log-Likelihood', fontsize=12)
        plt.title('Baum-Welch Training Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_convergence.png', dpi=300, bbox_inches='tight')
        print(f"→ Saved training curve to {self.output_dir}/training_convergence.png")
        plt.close()
    
    def decode_sequences(self, hmm: GaussianHMM, 
                        test_sequences: List[Tuple],
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode test sequences using Viterbi algorithm (Section 4.5)
        
        Parameters:
        -----------
        hmm : GaussianHMM
            Trained model
        test_sequences : List[Tuple]
            Test sequences (observations, labels, subject_id)
        verbose : bool
            Print progress
            
        Returns:
        --------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        """
        print("\n" + "="*80)
        print("DECODING SEQUENCES WITH VITERBI ALGORITHM (Section 4.5)")
        print("="*80)
        
        y_true_list = []
        y_pred_list = []
        
        start_time = time.time()
        
        for i, (X, y, subject) in enumerate(test_sequences):
            # Decode using Viterbi
            states, log_prob = hmm.viterbi(X)
            
            y_true_list.append(y)
            y_pred_list.append(states)
            
            if verbose and (i + 1) % 3 == 0:
                print(f"  Decoded {i+1}/{len(test_sequences)} sequences...", end='\r')
        
        decode_time = time.time() - start_time
        
        # Concatenate all predictions
        y_true = np.hstack(y_true_list)
        y_pred = np.hstack(y_pred_list)
        
        print(f"\n→ Decoding completed in {decode_time:.2f} seconds")
        print(f"→ Total test samples: {len(y_true)}")
        
        return y_true, y_pred
    
    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                           hmm: GaussianHMM) -> Dict:
        """
        Comprehensive performance evaluation
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        hmm : GaussianHMM
            Trained model
            
        Returns:
        --------
        metrics : Dict
            Performance metrics
        """
        print("\n" + "="*80)
        print("PERFORMANCE EVALUATION")
        print("="*80)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(6)
        )
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Activity': self.activity_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        print("\nPer-Class Metrics:")
        print(metrics_df.to_string(index=False))
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        print(f"\nMacro Average:")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall: {recall_macro:.4f}")
        print(f"  F1-Score: {f1_macro:.4f}")
        
        print(f"\nWeighted Average:")
        print(f"  Precision: {precision_weighted:.4f}")
        print(f"  Recall: {recall_weighted:.4f}")
        print(f"  F1-Score: {f1_weighted:.4f}")
        
        # Confusion matrix
        self._plot_confusion_matrix(y_true, y_pred)
        
        # Transition matrix analysis
        self._analyze_transition_matrix(hmm)
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'per_class': metrics_df.to_dict('records'),
            'macro_avg': {
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1_score': float(f1_macro)
            },
            'weighted_avg': {
                'precision': float(precision_weighted),
                'recall': float(recall_weighted),
                'f1_score': float(f1_weighted)
            }
        }
        
        metrics_file = f'{self.output_dir}/hmm_performance_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n→ Saved metrics to {metrics_file}")
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.activity_names,
                   yticklabels=self.activity_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.activity_names,
                   yticklabels=self.activity_names,
                   ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix_hmm.png', dpi=300, bbox_inches='tight')
        print(f"→ Saved confusion matrix to {self.output_dir}/confusion_matrix_hmm.png")
        plt.close()
    
    def _analyze_transition_matrix(self, hmm: GaussianHMM):
        """Analyze and visualize learned transition matrix"""
        A = hmm.get_transition_matrix()
        
        print("\n" + "="*80)
        print("TRANSITION MATRIX ANALYSIS")
        print("="*80)
        
        # Display transition matrix
        df_transition = pd.DataFrame(A, 
                                     columns=self.activity_names,
                                     index=self.activity_names)
        print("\nLearned Transition Matrix A:")
        print(df_transition.to_string())
        
        # Self-transition probabilities (diagonal)
        self_transitions = np.diag(A)
        print("\nSelf-Transition Probabilities (staying in same activity):")
        for i, activity in enumerate(self.activity_names):
            print(f"  {activity}: {self_transitions[i]:.4f}")
        
        # Most likely transitions
        print("\nTop 5 Most Likely Transitions (off-diagonal):")
        transitions = []
        for i in range(6):
            for j in range(6):
                if i != j:  # Exclude self-transitions
                    transitions.append((self.activity_names[i], 
                                      self.activity_names[j], 
                                      A[i, j]))
        transitions.sort(key=lambda x: x[2], reverse=True)
        
        for from_state, to_state, prob in transitions[:5]:
            print(f"  {from_state} → {to_state}: {prob:.4f}")
        
        # Stationary distribution
        stationary = hmm.get_stationary_distribution()
        print("\nStationary Distribution (long-term state probabilities):")
        for i, activity in enumerate(self.activity_names):
            print(f"  {activity}: {stationary[i]:.4f}")
        
        # Plot transition matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(A, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=self.activity_names,
                   yticklabels=self.activity_names,
                   ax=ax, cbar_kws={'label': 'Transition Probability'})
        ax.set_xlabel('To State', fontsize=12)
        ax.set_ylabel('From State', fontsize=12)
        ax.set_title('Learned HMM Transition Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/learned_transition_matrix.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n→ Saved transition matrix plot to {self.output_dir}/learned_transition_matrix.png")
        plt.close()
        
        # Save transition matrix
        df_transition.to_csv(f'{self.output_dir}/learned_transition_matrix.csv')
        print(f"→ Saved transition matrix to {self.output_dir}/learned_transition_matrix.csv")
    
    def run_complete_pipeline(self, feature_method: str = 'anova',
                             covariance_type: str = 'diag',
                             n_iter: int = 50,
                             run_tests: bool = True):
        """
        Execute complete HMM training and evaluation pipeline
        
        Parameters:
        -----------
        feature_method : str
            Feature selection method
        covariance_type : str
            'diag' or 'full'
        n_iter : int
            Maximum EM iterations
        run_tests : bool
            Whether to run assumption tests
        """
        print("\n" + "="*80)
        print("HMM TRAINING AND EVALUATION PIPELINE")
        print("Sections 4.2 (Model), 4.3 (Assumptions), 4.4 (Baum-Welch), 4.5 (Viterbi)")
        print("="*80)
        
        # Step 1: Prepare data
        train_seq, test_seq, n_features = self.prepare_sequences(method=feature_method)
        
        # Load feature names
        with open('outputs/results/selected_features.json', 'r') as f:
            feature_data = json.load(f)
        feature_names = feature_data[feature_method]
        
        # Step 2: Assumption tests (optional, can be slow)
        if run_tests:
            self.run_assumption_tests(train_seq, feature_names)
        else:
            print("\n→ Skipping assumption tests (set run_tests=True to enable)")
        
        # Step 3: Train HMM
        hmm = self.train_hmm(train_seq, n_features, 
                            covariance_type=covariance_type,
                            n_iter=n_iter, verbose=True)
        
        # Step 4: Decode test sequences
        y_true, y_pred = self.decode_sequences(hmm, test_seq, verbose=True)
        
        # Step 5: Evaluate performance
        metrics = self.evaluate_performance(y_true, y_pred, hmm)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nFinal Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_avg']['f1_score']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_avg']['f1_score']:.4f}")
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print("="*80 + "\n")
        
        return hmm, metrics


def main():
    """Run complete HMM evaluation pipeline"""
    
    evaluator = HMMEvaluator(output_dir='outputs/results')
    
    # Run pipeline with ANOVA features
    print("Running pipeline with ANOVA-selected features...")
    hmm, metrics = evaluator.run_complete_pipeline(
        feature_method='anova',
        covariance_type='diag',
        n_iter=50,
        run_tests=False  # Set to False to skip assumption tests (faster)
    )
    
    return hmm, metrics


if __name__ == '__main__':
    hmm, metrics = main()
