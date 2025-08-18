import pandas as pd
import numpy as np
import os
import time
import json

# --- Classical ML Imports ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Qiskit Imports (Updated for Qiskit 1.x) ---
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler

from qiskit_machine_learning.algorithms.classifiers import VQC

# Set a random seed for reproducibility
seed = 42
algorithm_globals.random_seed = seed

def load_and_prepare_data(data_path, n_features_pca):
    """ Loads, splits, and preprocesses the data, including PCA. """
    print("--- Loading and preparing data ---")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    TARGET_COLUMN = 'RG'

    train_df = df[((df['ID'] >= 10000) & (df['ID'] <= 11133)) | ((df['ID'] >= 20000) & (df['ID'] <= 21022))].copy()
    test_df = df[((df['ID'] >= 12000) & (df['ID'] <= 12132)) | ((df['ID'] >= 22000) & (df['ID'] <= 22384))].copy()

    non_feature_cols = ['ID', 'filename', TARGET_COLUMN, 'family']
    non_feature_cols = [col for col in non_feature_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COLUMN].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Original feature count: {X_train.shape[1]}")

    print(f"Applying PCA to reduce dimensions to {n_features_pca}...")
    pca = PCA(n_components=n_features_pca, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA complete. Explained variance with {n_features_pca} components: {explained_variance:.2f}%")
    
    return X_train_pca, y_train, X_test_pca, y_test, explained_variance

def main():
    """ Main function for the Quantum Model pipeline. """
    start_time = time.time()
    
    # --- 1. SETUP ---
    print("--- Quantum Model Pipeline ---")
    DATA_PATH = 'data/ransomware_features.csv'
    RESULTS_PATH = 'results'
    NUM_QUBITS = 8  # Number of features after PCA = number of qubits
    MAX_ITER = 100  # Max iterations for the optimizer

    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # --- 2. LOAD & PREPARE DATA ---
    X_train, y_train, X_test, y_test, explained_variance = load_and_prepare_data(
        data_path=DATA_PATH,
        n_features_pca=NUM_QUBITS
    )
    
    # --- 3. DEFINE THE VQC MODEL ---
    print("\n--- Defining the VQC Model ---")
    
    # A. Feature Map: Encodes classical data into quantum states.
    # ZZFeatureMap is a common choice for VQC.
    feature_map = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)
    
    # B. Ansatz: The parameterized quantum circuit that is trained.
    # RealAmplitudes is a hardware-efficient ansatz.
    ansatz = RealAmplitudes(num_qubits=NUM_QUBITS, reps=3) # reps is the number of layers
    
    # C. Optimizer: The classical algorithm that updates the ansatz parameters.
    optimizer = COBYLA(maxiter=MAX_ITER)
    
    # D. Sampler: The backend primitive to run the circuits.
    sampler = Sampler()

    # Callback function to store intermediate results
    objective_func_vals = []
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        print(f"Iteration {len(objective_func_vals)}: Cost = {obj_func_eval:.4f}", end='\r')

    # E. Combine everything into the VQC model
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback_graph,
        sampler=sampler,
    )
    print("VQC model defined successfully.")

    # --- 4. TRAIN THE MODEL ---
    print(f"\n--- Training the VQC model for {MAX_ITER} iterations... ---")
    training_start_time = time.time()

    # For reproducibility, we need to clear previous weights if the script is run multiple times
    vqc.initial_point = None 
    
    vqc.fit(X_train, y_train)

    training_end_time = time.time()
    print(f"\nTraining finished in {training_end_time - training_start_time:.2f} seconds.")

    # --- 5. EVALUATE THE MODEL ---
    print("\n--- Evaluating the trained VQC model on the test set ---")
    y_pred = vqc.predict(X_test)
    y_proba = vqc.predict_proba(X_test)[:, 1]

    # --- 6. SAVE AND DISPLAY RESULTS ---
    print("\n--- VQC Model Performance ---")
    results = {
        'Model': 'Hybrid VQC',
        'Num Qubits': NUM_QUBITS,
        'Optimizer': 'COBYLA',
        'Max Iterations': MAX_ITER,
        'Explained Variance (%)': round(explained_variance, 2),
        'Accuracy (%)': round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision (%)': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall (%)': round(recall_score(y_test, y_pred) * 100, 2),
        'F1-Score (%)': round(f1_score(y_test, y_pred) * 100, 2),
        'AUC': round(roc_auc_score(y_test, y_proba), 4)
    }

    # Print results to console
    for key, value in results.items():
        print(f"{key}: {value}")
        
    # Save results to a JSON file
    results_json_path = os.path.join(RESULTS_PATH, 'quantum_model_performance.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nQuantum model results saved to {results_json_path}")

    end_time = time.time()
    print(f"\n--- Entire script finished in {end_time - start_time:.2f} seconds ---")


if __name__ == '__main__':
    main()
