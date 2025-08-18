import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Preprocessing and Metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def main():
    """
    Main function to run the classical model training and evaluation pipeline.
    """
    start_time = time.time()
    
    # --- 1. SETUP ---
    print("--- Step 1: Setting up paths and parameters ---")
    DATA_PATH = 'data/ransomware_features.csv'
    FIGURES_PATH = 'figures'
    RESULTS_PATH = 'results'
    SAVE_DPI = 600

    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print(f"Data will be loaded from: {DATA_PATH}")
    print(f"Figures will be saved to: {FIGURES_PATH}")
    print(f"Results will be saved to: {RESULTS_PATH}\n")

    # --- 2. LOAD DATA ---
    print("--- Step 2: Loading and inspecting data ---")
    try:
        df = pd.read_csv(DATA_PATH)
        # *** SỬA LỖI: Làm sạch tên cột (loại bỏ khoảng trắng thừa) ***
        df.columns = df.columns.str.strip()
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        # *** SỬA LỖI: In ra các tên cột để kiểm tra ***
        print(f"Column names: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Exiting.")
        return

    # *** SỬA LỖI: Xác định tên cột nhãn chính xác ***
    TARGET_COLUMN = 'RG'
    if TARGET_COLUMN not in df.columns:
        print(f"\nError: Target column '{TARGET_COLUMN}' not found in the dataframe.")
        print("Please check the column names printed above and update the 'TARGET_COLUMN' variable.")
        return

    # --- 3. SPLIT DATA ---
    print("\n--- Step 3: Splitting data into training and testing sets ---")
    train_df = df[((df['ID'] >= 10000) & (df['ID'] <= 11133)) | ((df['ID'] >= 20000) & (df['ID'] <= 21022))].copy()
    test_df = df[((df['ID'] >= 12000) & (df['ID'] <= 12132)) | ((df['ID'] >= 22000) & (df['ID'] <= 22384))].copy()

    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    print(f"Training set class distribution:\n{train_df[TARGET_COLUMN].value_counts(normalize=True)}")
    print(f"Testing set class distribution:\n{test_df[TARGET_COLUMN].value_counts(normalize=True)}")
    
    # --- 4. PREPARE DATA FOR MODELING ---
    print("\n--- Step 4: Preparing data for modeling (feature selection and scaling) ---")
    non_feature_cols = ['ID', 'filename', TARGET_COLUMN, 'family']
    # Loại bỏ các cột không tồn tại khỏi danh sách này một cách an toàn
    non_feature_cols = [col for col in non_feature_cols if col in df.columns]

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    all_zero_features_in_train = [col for col in feature_cols if train_df[col].sum() == 0]
    print(f"Found and removing {len(all_zero_features_in_train)} features that are all-zero in the training set.")
    
    feature_cols = [col for col in feature_cols if col not in all_zero_features_in_train]
    
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]

    print(f"Final feature count: {len(feature_cols)}")
    print(f"X_train shape: {X_train.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features have been scaled using StandardScaler.")

    # --- 5. TRAIN AND EVALUATE MODELS ---
    print("\n--- Step 5: Training and evaluating classical models ---")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }

    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train)
        
        print(f"--- Evaluating {name} ---")
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba),
            'model': model
        }

    # --- 6. SAVE AND DISPLAY RESULTS ---
    print("\n--- Step 6: Saving and displaying results ---")
    results_df = pd.DataFrame(results).T.drop(columns=['model'])
    # Convert percentages, keep AUC as is
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        results_df[col] = results_df[col] * 100
        results_df.rename(columns={col: f"{col} (%)"}, inplace=True)
    
    print("\nPerformance Comparison on the Test Set:")
    print(results_df.round(2))
    
    results_csv_path = os.path.join(RESULTS_PATH, 'classical_models_performance.csv')
    results_df.round(4).to_csv(results_csv_path)
    print(f"\nResults table saved to {results_csv_path}")

    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    for name, result in results.items():
        y_proba = result['model'].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = result['AUC']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curves for Classical Baseline Models')
    plt.legend(loc='lower right')
    
    roc_figure_path = os.path.join(FIGURES_PATH, 'classical_roc_curves.png')
    plt.savefig(roc_figure_path, dpi=SAVE_DPI, bbox_inches='tight')
    print(f"ROC curve plot saved to {roc_figure_path}")
    
    end_time = time.time()
    print(f"\n--- Script finished in {end_time - start_time:.2f} seconds ---")


if __name__ == '__main__':
    main()
