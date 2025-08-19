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
    DATA_PATH = 'data/ransomware_features.csv'
    FIGURES_PATH = 'figures'
    RESULTS_PATH = 'results' 
    SAVE_DPI = 600

    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # --- 2. LOAD DATA ---
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    TARGET_COLUMN = 'RG'

    # --- 3. SPLIT DATA ---
    train_df = df[((df['ID'] >= 10000) & (df['ID'] <= 11133)) | ((df['ID'] >= 20000) & (df['ID'] <= 21022))].copy()
    test_df = df[((df['ID'] >= 12000) & (df['ID'] <= 12132)) | ((df['ID'] >= 22000) & (df['ID'] <= 22384))].copy()
    
    # --- 4. PREPARE DATA FOR MODELING ---
    non_feature_cols = ['ID', 'filename', TARGET_COLUMN, 'family']
    non_feature_cols = [col for col in non_feature_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 5. TRAIN AND EVALUATE MODELS ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        results[name] = {'model': model, 'AUC': roc_auc_score(y_test, y_proba)}

    # --- 6. GENERATE HIGH-QUALITY PLOT ---
    print("\nGenerating high-quality ROC curve plot...")
    
    # --- STYLING ENHANCEMENTS ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8)) # Kích thước lớn

    # Màu sắc đậm, tương phản cao
    colors = {
        "Logistic Regression": "#0d3b66", # Xanh Sapphire Đậm
        "Random Forest": "#c0392b",       # Đỏ Lựu Đậm
        "XGBoost": "#27ae60"              # Xanh Ngọc Bích
    }
    # ---------------------------

    for name, result in results.items():
        y_proba = result['model'].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = result['AUC']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', 
                 color=colors.get(name, 'gray'), 
                 linewidth=3) # Đường kẻ dày hơn

    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)', linewidth=2.5) # Đường tham chiếu cũng dày hơn
    
    # --- FONT SIZE ENHANCEMENTS ---
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.title('ROC Curves for Classical Baseline Models', fontsize=18, weight='bold')
    plt.legend(loc='lower right', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # -----------------------------
    
    plt.grid(True)
    roc_figure_path = os.path.join(FIGURES_PATH, 'classical_roc_curves.png')
    plt.savefig(roc_figure_path, dpi=SAVE_DPI, bbox_inches='tight')
    print(f"High-quality ROC curve plot saved to {roc_figure_path}")
    
    end_time = time.time()
    print(f"\n--- Script finished in {end_time - start_time:.2f} seconds ---")

if __name__ == '__main__':
    main()
