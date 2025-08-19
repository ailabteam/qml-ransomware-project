# src/plot_final_results.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_results():
    FIGURES_PATH = 'figures'
    os.makedirs(FIGURES_PATH, exist_ok=True)
    SAVE_DPI = 600

    # Data from our experiments
    data = {
        'Num Qubits': [4, 8, 12],
        'Recall (%)': [42.60, 40.00, 55.06],
        'AUC': [0.5994, 0.4624, 0.5374],
        'Explained Variance (%)': [19.14, 29.07, 35.50]
    }
    df = pd.DataFrame(data)
    
    # Best classical baseline recall
    best_classical_recall = 97.66

    # Create the plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Recall on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Number of Qubits (PCA Components)')
    ax1.set_ylabel('Recall Score (%)', color=color)
    ax1.plot(df['Num Qubits'], df['Recall (%)'], marker='o', linestyle='-', color=color, label='VQC Recall')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(df['Num Qubits']) # Ensure ticks are at 4, 8, 12
    ax1.set_ylim(0, 100)

    # Plot the classical baseline
    ax1.axhline(y=best_classical_recall, color='r', linestyle='--', label=f'Best Classical Recall ({best_classical_recall}%)')

    # Create a secondary y-axis for Explained Variance
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Explained Variance (%)', color=color)
    ax2.plot(df['Num Qubits'], df['Explained Variance (%)'], marker='s', linestyle=':', color=color, label='Explained Variance')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')

    plt.title('VQC Performance and Explained Variance vs. Qubit Count')
    fig.tight_layout()
    
    # Save the figure
    output_path = os.path.join(FIGURES_PATH, 'final_recall_vs_qubits.png')
    plt.savefig(output_path, dpi=SAVE_DPI)
    print(f"Final plot saved to: {output_path}")

if __name__ == '__main__':
    plot_results()
