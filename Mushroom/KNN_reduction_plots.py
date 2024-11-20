import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV and prepare aggregated statistics for reduction analysis
    """
    # Load results
    results = pd.DataFrame(pd.read_csv(csv_path))

    # Extract configuration parameters and reduction method
    results[['Algorithm', 'k', 'distance_metric', 'weighting_method', 'voting_policy', 'reduction']] = \
        results['Model'].str.split(', ', expand=True)

    # Create aggregated results
    aggregated_results = results.groupby(['reduction']).agg({
        'Accuracy': ['mean', 'std'],
        'Time': 'mean',
        'F1': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    aggregated_results.columns = [
        'reduction', 'mean_accuracy', 'std_accuracy', 'mean_time',
        'mean_f1', 'std_f1'
    ]

    return results, aggregated_results


def create_plots_folder(base_path: str):
    """Create folder for plots if it doesn't exist"""
    Path(base_path).mkdir(parents=True, exist_ok=True)

def calculate_storage_percentages(sample_counts_df: pd.DataFrame):
    """
    Calculate the average percentage of storage reduction for each reduction method compared to the "None" reduction.
    """
    storage_percentages = {'NONE': 100}

    for reduction_method in sample_counts_df['Reduction Method'].unique():
        if reduction_method == "NONE":
            continue

        reduction_samples = sample_counts_df[sample_counts_df['Reduction Method'] == reduction_method]['Training Samples']
        none_samples = sample_counts_df[sample_counts_df['Reduction Method'] == "NONE"]['Training Samples']

        avg_reduction_samples = reduction_samples.mean()
        avg_none_samples = none_samples.mean()

        storage_percentage = avg_reduction_samples / avg_none_samples * 100
        storage_percentages[reduction_method] = storage_percentage

    return storage_percentages


def plot_accuracy_storage_comparison(results: pd.DataFrame, sample_counts: pd.DataFrame, plots_path: str):
    """Plot comparison of reduction methods' accuracies and storage percentages"""

    storage_percentages = calculate_storage_percentages(sample_counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy Distribution Plot
    sns.boxplot(x='reduction', y='Accuracy', data=results, ax=ax1)
    ax1.set_title('Accuracy Distribution by Reduction Method')
    ax1.set_xlabel('Reduction Method')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)

    # Storage Percentages Plot
    ax2.bar(storage_percentages.keys(), storage_percentages.values())
    ax2.set_xlabel('Reduction Method')
    ax2.set_ylabel('Storage Percentage (%)')
    ax2.set_title('Storage Percentages per Reduction Method')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'accuracy_storage_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_time_comparison(aggregated_results: pd.DataFrame, plots_path: str):
    """Plot time comparison across reduction methods"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='reduction', y='mean_time', data=aggregated_results)
    plt.title('Average Execution Time by Reduction Method')
    plt.xlabel('Reduction Method')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'reduction_time_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()


def create_comparison_plots(results: pd.DataFrame, plots_path: str):
    """Create additional comparison plots"""
    # Accuracy vs Time trade-off
    plt.figure(figsize=(12, 8))
    for reduction in results['reduction'].unique():
        reduction_data = results[results['reduction'] == reduction]
        plt.scatter(reduction_data['Time'], reduction_data['Accuracy'],
                    alpha=0.6, label=reduction)
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Time-Accuracy Trade-off by Reduction Method')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'reduction_time_accuracy_tradeoff.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # F1 vs Accuracy correlation
    plt.figure(figsize=(12, 8))
    for reduction in results['reduction'].unique():
        reduction_data = results[results['reduction'] == reduction]
        plt.scatter(reduction_data['F1'], reduction_data['Accuracy'],
                    alpha=0.6, label=reduction)
    plt.xlabel('F1 Score')
    plt.ylabel('Accuracy')
    plt.title('F1 Score vs Accuracy Correlation by Reduction Method')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'reduction_f1_accuracy_correlation.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_efficiency_comparison(results: pd.DataFrame, sample_counts: pd.DataFrame, plots_path: str):
    """
    Plot accuracy per training sample for each reduction method.
    This shows how efficiently each method uses its training samples to achieve accuracy.
    """
    # Calculate average accuracy for each reduction method
    avg_accuracy = results.groupby('reduction')['Accuracy'].mean()

    # Calculate average number of training samples for each reduction method
    avg_samples = sample_counts.groupby('Reduction Method')['Training Samples'].mean()

    # Calculate efficiency (accuracy per sample)
    efficiency = {}
    for method in avg_accuracy.index:
        # Match the method name in sample_counts (which uses uppercase)
        samples_method = method.upper()
        if samples_method in avg_samples.index:
            # Multiply by 100 to make the values more readable
            efficiency[method] = (avg_accuracy[method] * 100) / avg_samples[samples_method]

    # Create the plot
    plt.figure(figsize=(10, 6))
    methods = list(efficiency.keys())
    efficiencies = list(efficiency.values())

    bars = plt.bar(methods, efficiencies)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2e}',
                 ha='center', va='bottom')

    plt.title('Accuracy per Training Sample by Reduction Method')
    plt.xlabel('Reduction Method')
    plt.ylabel('Efficiency (Accuracy % per Training Sample)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(plots_path, 'reduction_efficiency_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    dataset_path = '..\\Mushroom'
else:
    dataset_path = 'Mushroom'
# Paths
csv_path = os.path.join(dataset_path,'knn_reduction_results.csv')
counts_path = os.path.join(dataset_path,'knn_reduction_counts.csv')
plots_path = os.path.join(dataset_path,'plots_and_tables\\knn_reduction')

# Create plots folder
create_plots_folder(plots_path)

# Load and prepare data
results, aggregated_results = load_and_prepare_data(csv_path)
sample_counts = pd.DataFrame(pd.read_csv(counts_path))

# Generate plots
plot_accuracy_storage_comparison(results, sample_counts, plots_path)
plot_time_comparison(aggregated_results, plots_path)
create_comparison_plots(results, plots_path)
plot_efficiency_comparison(results, sample_counts, plots_path)

print(f"Plots successfully saved in folder Mushroom/plots_and_tables/knn_reduction\n")
