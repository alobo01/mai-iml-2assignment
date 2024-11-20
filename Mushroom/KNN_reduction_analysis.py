import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class ReductionMethodAnalyzer:
    def __init__(self, csv_path: str, alpha: float):
        """Initialize the analyzer with the CSV data."""
        self.df = pd.read_csv(csv_path)
        self.parse_model_column()
        self.alpha = alpha

    def parse_model_column(self):
        """Parse the Model column to extract the reduction method."""
        # Split the Model column into its components
        hyperparams = self.df['Model'].str.split(',', expand=True)
        self.df['reduction_method'] = hyperparams[5].str.strip()

    def create_pivot_table(self) -> pd.DataFrame:
        """Create a pivot table for statistical analysis."""
        return self.df.pivot_table(
            values='Accuracy',
            index='Dataset/Fold',
            columns='reduction_method',
            aggfunc='first'
        )

    def perform_friedman_test(self, pivot_df: pd.DataFrame) -> Tuple[float, float]:
        """Perform Friedman test on the pivot table."""
        return stats.friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])

    def perform_bonferroni_test(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform Bonferroni-corrected Wilcoxon signed-rank tests against NONE reduction.
        """
        control_data = pivot_df['NONE']
        results = {}
        n_comparisons = len(pivot_df.columns) - 1  # Subtract control

        for method in pivot_df.columns:
            if method != 'NONE':
                # Calculate effect size (median difference)
                effect_size = np.median(pivot_df[method] - control_data)

                # Perform Wilcoxon test
                statistic, p_value = stats.wilcoxon(
                    control_data,
                    pivot_df[method],
                    alternative='two-sided'
                )

                # Calculate percentage of improvement/degradation
                diff_percentage = ((pivot_df[method].mean() - control_data.mean())
                                   / control_data.mean() * 100)

                # Apply Bonferroni correction
                adjusted_p = min(p_value * n_comparisons, 1.0)

                results[method] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'adjusted_p': adjusted_p,
                    'effect_size': effect_size,
                    'diff_percentage': diff_percentage
                }

        return pd.DataFrame(results).T

    def analyze_reduction_methods(self) -> Dict:
        """Analyze the impact of different reduction methods."""
        pivot_df = self.create_pivot_table()

        # Perform Friedman test
        friedman_stat, friedman_p = self.perform_friedman_test(pivot_df)
        significant_differences = friedman_p < self.alpha

        # Only perform post-hoc test if significant differences are found
        post_hoc = None
        if significant_differences:
            post_hoc = self.perform_bonferroni_test(pivot_df)

        # Calculate summary statistics
        summary = self.df.groupby('reduction_method')['Accuracy'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)

        # Calculate execution time statistics
        time_stats = self.df.groupby('reduction_method')['Time'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)

        return {
            'summary': summary,
            'time_stats': time_stats,
            'friedman_result': (friedman_stat, friedman_p),
            'post_hoc': post_hoc,
            'significant_differences': significant_differences
        }

    def visualize_results(self, results: Dict) -> Tuple[plt.Figure, plt.Figure]:
        """Create visualizations for the analysis results."""
        # Figure 1: Accuracy comparison and statistical significance
        if results['significant_differences'] and results['post_hoc'] is not None:
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        summary = results['summary']
        methods = summary.index

        # Bar plot of mean accuracy
        ax1.bar(range(len(methods)), summary['mean'])
        ax1.errorbar(range(len(methods)), summary['mean'],
                     yerr=summary['std'], fmt='none', color='black', capsize=5)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.set_title('Mean Accuracy by Reduction Method')
        ax1.set_ylabel('Accuracy')

        # Plot of statistical significance (only if significant differences were found)
        if results['significant_differences'] and results['post_hoc'] is not None:
            post_hoc = results['post_hoc']
            significance_plot = -np.log10(post_hoc['adjusted_p'])
            ax2.bar(range(len(post_hoc)), significance_plot)
            ax2.axhline(y=-np.log10(self.alpha), color='r', linestyle='--',
                        label=f'p={self.alpha} threshold')
            ax2.set_xticks(range(len(post_hoc)))
            ax2.set_xticklabels(post_hoc.index, rotation=45, ha='right')
            ax2.set_title('Statistical Significance vs NONE\n(-log10 adjusted p-value)')
            ax2.set_ylabel('-log10(adjusted p-value)')
            ax2.legend()

        plt.tight_layout()

        # Figure 2: Performance metrics
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot percentage difference from control (only if post-hoc results exist)
        if results['post_hoc'] is not None:
            diff_percentages = results['post_hoc']['diff_percentage']
            colors = ['g' if x > 0 else 'r' for x in diff_percentages]
            ax3.bar(range(len(diff_percentages)), diff_percentages, color=colors)
            ax3.set_xticks(range(len(diff_percentages)))
            ax3.set_xticklabels(diff_percentages.index, rotation=45, ha='right')
            ax3.set_title('Percentage Difference from NONE')
            ax3.set_ylabel('Difference (%)')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Plot execution times
        time_stats = results['time_stats']
        ax4.bar(range(len(time_stats)), time_stats['mean'])
        ax4.errorbar(range(len(time_stats)), time_stats['mean'],
                     yerr=time_stats['std'], fmt='none', color='black', capsize=5)
        ax4.set_xticks(range(len(time_stats)))
        ax4.set_xticklabels(time_stats.index, rotation=45, ha='right')
        ax4.set_title('Mean Execution Time by Reduction Method')
        ax4.set_ylabel('Time (seconds)')

        plt.tight_layout()

        return fig1, fig2

    def generate_report(self, results: Dict, output_path: str):
        """Generate a detailed text report of the statistical analysis results."""
        with open(output_path, 'w') as f:
            # Write header
            f.write("Statistical Analysis Report - Reduction Methods\n")
            f.write("=" * 50 + "\n\n")

            # Summary Statistics
            f.write("Summary Statistics (Accuracy)\n")
            f.write("--------------------------\n")
            f.write(results['summary'].to_string())
            f.write("\n\n")

            # Execution Time Statistics
            f.write("Execution Time Statistics\n")
            f.write("-----------------------\n")
            f.write(results['time_stats'].to_string())
            f.write("\n\n")

            # Friedman Test Results
            f.write("Friedman Test Results\n")
            f.write("--------------------\n")
            stat, p = results['friedman_result']
            f.write(f"Test Statistic: {stat:.4f}\n")
            f.write(f"P-value: {p:.4f}\n")
            f.write(f"Significance level (alpha): {self.alpha}\n\n")

            if results['significant_differences']:
                f.write("Significant differences were found between the reduction methods.\n\n")

                # Post-hoc Test Results
                f.write("Post-hoc Test Results (Bonferroni)\n")
                f.write("--------------------------------\n")
                f.write("Results compared to control (NONE):\n\n")
                f.write(results['post_hoc'].round(4).to_string())
            else:
                f.write(f"No significant differences were found between the reduction methods ")
                f.write(f"at the {self.alpha} significance level.\n")
                f.write("Post-hoc test was not performed.\n")


def main(csv_path: str, output_dir: str = None, alpha: float = 0.1):
    """Main function to run the analysis."""
    analyzer = ReductionMethodAnalyzer(csv_path, alpha)
    results = analyzer.analyze_reduction_methods()

    # Print results to console
    print("\nSummary Statistics (Accuracy):")
    print(results['summary'])

    print("\nExecution Time Statistics:")
    print(results['time_stats'])

    print("\nFriedman Test Results:")
    stat, p = results['friedman_result']
    print(f"Statistic: {stat:.4f}")
    print(f"p-value: {p:.4f}")
    print(f"Significance level (alpha): {alpha}")

    if results['significant_differences']:
        print("\nSignificant differences found between reduction methods.")
        print("\nPost-hoc Test Results (Bonferroni):")
        print(results['post_hoc'].round(4))
    else:
        print(f"\nNo significant differences were found between the reduction methods ")
        print(f"at the {alpha} significance level.")
        print("Post-hoc test was not performed.")

    # Create and save visualizations and report
    if output_dir:
        fig1, fig2 = analyzer.visualize_results(results)
        fig1.savefig(f"{output_dir}/reduction_accuracy_analysis.png")
        fig2.savefig(f"{output_dir}/reduction_performance_analysis.png")
        analyzer.generate_report(results, f"{output_dir}/reduction_analysis.txt")
        plt.close('all')

    return results


if __name__ == "__main__":
    csv_path = "knn_reduction_results.csv"
    output_dir = "plots_and_tables\\knn_reduction"
else:
    csv_path = "Hepatitis\\knn_reduction_results.csv"
    output_dir = "Hepatitis\\plots_and_tables\\knn_reduction"

results = main(csv_path, output_dir)

print(f"Results successfully saved in folder Mushroom/plots_and_tables/knn_reduction\n")