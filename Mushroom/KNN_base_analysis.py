import pandas as pd
import numpy as np
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_model_performance(csv_path, alpha=0.05):
    """
    Analyze model performance using Friedman and potentially Nemenyi tests.

    Parameters:
    csv_path (str): Path to the CSV file containing model performance data
    alpha (float): Significance level for statistical tests

    Returns:
    tuple: (top_models_df, friedman_result, nemenyi_matrix, significant_differences)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Calculate average accuracy for each model
    avg_performance = df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
    top_models = avg_performance.head(8).index.tolist()

    # Filter data for Top models
    top_df = df[df['Model'].isin(top_models)]

    # Create a pivot table for the Friedman test
    pivot_df = top_df.pivot(
        index='Dataset/Fold',
        columns='Model',
        values='Accuracy'
    )

    # Perform Friedman test
    friedman_statistic, friedman_p_value = stats.friedmanchisquare(
        *[pivot_df[model] for model in top_models]
    )

    # Only perform Nemenyi test if Friedman test is significant
    significant_differences = friedman_p_value < alpha
    nemenyi_result = None
    if significant_differences:
        nemenyi_result = posthoc_nemenyi_friedman(pivot_df)

    # Create summary DataFrame for Top models
    summary_stats = pd.DataFrame({
        'Mean Accuracy': top_df.groupby('Model')['Accuracy'].mean(),
        'Std Accuracy': top_df.groupby('Model')['Accuracy'].std(),
        'Mean F1': top_df.groupby('Model')['F1'].mean(),
        'Mean Time': top_df.groupby('Model')['Time'].mean()
    }).round(4)

    summary_stats = summary_stats.loc[top_models]  # Preserve order

    return summary_stats, (friedman_statistic, friedman_p_value), nemenyi_result, significant_differences


def visualize_results(summary_stats, friedman_result, nemenyi_matrix, significant_differences):
    """
    Create visualizations for the statistical analysis results.
    """
    if significant_differences and nemenyi_matrix is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Plot mean accuracy with error bars
    models = summary_stats.index
    means = summary_stats['Mean Accuracy']
    stds = summary_stats['Std Accuracy']

    ax1.bar(models, means)
    ax1.errorbar(models, means, yerr=stds, fmt='none', color='black', capsize=5)
    ax1.set_title('Mean Accuracy of Top Models')
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')

    # Plot Nemenyi test results as a heatmap only if significant differences were found
    if significant_differences and nemenyi_matrix is not None:
        sns.heatmap(nemenyi_matrix, annot=True, cmap='RdYlGn_r', ax=ax2)
        ax2.set_title('Nemenyi Test p-values\n(lower values indicate significant differences)')

    plt.tight_layout()
    return fig


def generate_report(summary_stats, friedman_result, nemenyi_matrix, significant_differences, alpha, output_path):
    """
    Generate a detailed text report of the statistical analysis results.
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("Statistical Analysis Report\n")
        f.write("=========================\n\n")

        # Top Models Section
        f.write("Top Models Performance Summary\n")
        f.write("--------------------------\n")
        for model in summary_stats.index:
            f.write(f"\nModel: {model}\n")
            f.write(f"Mean Accuracy: {summary_stats.loc[model, 'Mean Accuracy']:.4f}\n")
            f.write(f"Std Accuracy: {summary_stats.loc[model, 'Std Accuracy']:.4f}\n")
            f.write(f"Mean F1 Score: {summary_stats.loc[model, 'Mean F1']:.4f}\n")
            f.write(f"Mean Time: {summary_stats.loc[model, 'Mean Time']:.4f}\n")

        # Friedman Test Results
        f.write("\nFriedman Test Results\n")
        f.write("--------------------\n")
        f.write(f"Test Statistic: {friedman_result[0]:.4f}\n")
        f.write(f"P-value: {friedman_result[1]:.4f}\n")
        f.write(f"Significance level (alpha): {alpha}\n")

        if significant_differences:
            # Nemenyi Test Results
            f.write("\nNemenyi Test P-values\n")
            f.write("-------------------\n")
            f.write("Lower values indicate more significant differences between models\n\n")

            # Format Nemenyi matrix as a table
            models = nemenyi_matrix.index
            f.write("Model Pairs" + " " * 30 + "P-value\n")
            f.write("-" * 50 + "\n")

            # Write all pairwise comparisons
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model1, model2 = models[i], models[j]
                    p_value = nemenyi_matrix.iloc[i, j]
                    pair_name = f"{model1} vs {model2}"
                    f.write(f"{pair_name:<40} {p_value:.4f}\n")
        else:
            f.write("\nNo significant differences were found between the models ")
            f.write(f"at the {alpha} significance level.\n")
            f.write("Nemenyi post-hoc test was not performed.\n")


def main(csv_path, plot_output_path=None, report_output_path=None, alpha=0.05):
    """
    Main function to run the analysis and save results.
    """
    # Perform analysis
    summary_stats, friedman_result, nemenyi_matrix, significant_differences = \
        analyze_model_performance(csv_path, alpha)

    # Print results to console
    print("\nTop Models Summary Statistics:")
    print(summary_stats)
    print("\nFriedman Test Results:")
    print(f"Statistic: {friedman_result[0]:.4f}")
    print(f"P-value: {friedman_result[1]:.4f}")
    print(f"Significance level (alpha): {alpha}")

    if significant_differences:
        print("\nSignificant differences found between models.")
        print("\nNemenyi Test Results (p-values):")
        print(nemenyi_matrix.round(4))
    else:
        print(f"\nNo significant differences were found between the models at the {alpha} significance level.")
        print("Nemenyi post-hoc test was not performed.")

    # Create and save visualizations
    fig = visualize_results(summary_stats, friedman_result, nemenyi_matrix, significant_differences)
    if plot_output_path:
        fig.savefig(plot_output_path)

    # Generate and save text report
    if report_output_path:
        generate_report(summary_stats, friedman_result, nemenyi_matrix,
                        significant_differences, alpha, report_output_path)

    return summary_stats, friedman_result, nemenyi_matrix, significant_differences


# Example usage:
if __name__ == "__main__":
    # Replace with your file paths
    csv_path = "knn_base_results.csv"
    plot_output_path = "plots_and_tables\\knn_base\\statistical_analysis_results.png"
    report_output_path = "plots_and_tables\\knn_base\\statistical_analysis_report.txt"
else:
    csv_path = 'Mushroom\\knn_base_results.csv'
    plot_output_path = "Mushroom\\plots_and_tables\\knn_base\\statistical_analysis_results.png"
    report_output_path = "Mushroom\\plots_and_tables\\knn_base\\statistical_analysis_report.txt"

alpha = 0.2  # Set significance level
main(csv_path, plot_output_path, report_output_path, alpha)
print(f"Results saved in file Mushroom/plots_and_tables/knn_base/statistical_analysis_report.txt\n")