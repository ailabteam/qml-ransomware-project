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
        'Explained Variance (%)': [19.14, 29.07, 35.50]
    }
    df = pd.DataFrame(data)
    
    # Best classical baseline recall
    best_classical_recall = 97.66

    # Use a style and context appropriate for a paper
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Recall on the primary y-axis
    color_recall = '#0d3b66'      # Dark Sapphire
    ax1.set_xlabel('Number of Qubits (PCA Components)', fontsize=16)
    ax1.set_ylabel('Recall Score (\%)', color=color_recall, fontsize=16, weight='bold')
    ax1.plot(df['Num Qubits'], df['Recall (%)'], marker='o', markersize=10, linestyle='-', 
             color=color_recall, linewidth=3.5, label='VQC Recall')
    ax1.tick_params(axis='y', labelcolor=color_recall, labelsize=14)
    ax1.set_xticks(df['Num Qubits'])
    ax1.set_xticklabels(df['Num Qubits'], fontsize=14)
    ax1.set_ylim(0, 110)

    # Plot the classical baseline
    color_baseline = '#c0392b'    # Dark Pomegranate
    ax1.axhline(y=best_classical_recall, color=color_baseline, linestyle='--', linewidth=3,
                label=f'Best Classical Recall ({best_classical_recall}\%)')

    # Create a secondary y-axis for Explained Variance
    ax2 = ax1.twinx()
    color_variance = '#5a6a7b'    # Slate Gray
    ax2.set_ylabel('Cumulative Explained Variance (\%)', color=color_variance, fontsize=16)
    
    # *** DÒNG ĐÃ SỬA LỖI ***
    ax2.plot(df['Num Qubits'], df['Explained Variance (%)'], marker='s', markersize=8, linestyle=':', 
             color=color_variance, linewidth=3, label='Explained Variance')
    # *********************

    ax2.tick_params(axis='y', labelcolor=color_variance, labelsize=14)
    ax2.set_ylim(0, 110)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=14)

    plt.title('VQC Performance vs. Qubit Count', fontsize=18, weight='bold')
    fig.tight_layout()
    
    output_path = os.path.join(FIGURES_PATH, 'final_recall_vs_qubits.png')
    plt.savefig(output_path, dpi=SAVE_DPI)
    print(f"High-quality final plot saved to: {output_path}")

if __name__ == '__main__':
    plot_results()
