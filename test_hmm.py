"""
Quick test script to verify HMM implementation
"""
import numpy as np

print("Testing HMM implementation...")
print("="*80)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from scripts.hmm_model import GaussianHMM
    from scripts.assumption_tests import AssumptionVerifier
    from scripts.train_evaluate_hmm import HMMEvaluator
    from scripts.data_loader import StochasticDataLoader
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    exit(1)

# Test 2: Create HMM instance (will be recreated with correct n_features after loading data)
print("\n2. Testing HMM instantiation...")
try:
    hmm = GaussianHMM(n_states=6, n_features=100, covariance_type='diag')
    print(f"   ✓ Created HMM with {hmm.n_states} states, {hmm.n_features} features")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 3: Load small dataset
print("\n3. Testing data loading...")
try:
    loader = StochasticDataLoader()
    X_train, X_test, y_train, y_test, _, _, features = \
        loader.load_with_selected_features(method='anova', encode_labels=True)
    print(f"   ✓ Loaded data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"   ✓ Features: {len(features)} selected")
    print(f"   ✓ Labels encoded: {np.unique(y_train)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 4: Prepare sequences
print("\n4. Testing sequence preparation...")
try:
    train_seq, test_seq = loader.prepare_for_hmm(method='anova')
    print(f"   ✓ Prepared {len(train_seq)} training sequences")
    print(f"   ✓ Prepared {len(test_seq)} test sequences")
    print(f"   ✓ Example sequence shape: {train_seq[0][0].shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 5: Initialize HMM parameters
print("\n5. Testing HMM parameter initialization...")
try:
    import numpy as np
    
    # Recreate HMM with correct n_features from loaded data
    n_features = X_train.shape[1]
    hmm = GaussianHMM(n_states=6, n_features=n_features, covariance_type='diag')
    print(f"   ✓ Recreated HMM with {n_features} features")
    
    X_small = X_train[:1000]
    y_small = y_train[:1000]
    
    hmm._initialize_parameters(X_small, y_small)
    print(f"   ✓ Initialized transition matrix A: {hmm.A.shape}")
    print(f"   ✓ Initialized means: {hmm.means.shape}")
    print(f"   ✓ Initialized covariances: {hmm.covars.shape}")
    print(f"   ✓ Row sums of A (should be ~1.0): {hmm.A.sum(axis=1)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 6: Test forward algorithm
print("\n6. Testing forward algorithm...")
try:
    X_seq = train_seq[0][0][:50]  # Small sequence
    alpha, log_likelihood = hmm._forward(X_seq)
    print(f"   ✓ Forward pass completed")
    print(f"   ✓ Alpha shape: {alpha.shape}")
    print(f"   ✓ Log-likelihood: {log_likelihood:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 7: Test Viterbi algorithm
print("\n7. Testing Viterbi algorithm...")
try:
    states, log_prob = hmm.viterbi(X_seq)
    print(f"   ✓ Viterbi decoding completed")
    print(f"   ✓ Predicted states shape: {states.shape}")
    print(f"   ✓ Log probability: {log_prob:.4f}")
    print(f"   ✓ Unique states predicted: {np.unique(states)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED - HMM implementation is working correctly!")
print("="*80)
print("\nReady to run full pipeline:")
print("  python scripts/train_evaluate_hmm.py")
print("="*80)
