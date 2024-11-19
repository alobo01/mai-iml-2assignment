import pandas as pd
import numpy as np
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List


class KNNHyperparameterAnalyzer:
    def __init__(self, csv_path: str, alpha: float):
        """Initialize the analyzer with the CSV data."""
        self.df = pd.read_csv(csv_path)
        self.parse_model_column()
        self.alpha = alpha

    def parse_model_column(self):
        """Parse the Model column into separate hyperparameter columns."""
        # Split the Model column into its components
        hyperparams = self.df['Model'].str.split(',', expand=True)

        # Clean and rename columns
        self.df['k'] = hyperparams[1].str.strip().astype(int)
        self.df['distance_metric'] = hyperparams[2].str.strip()
        self.df['weighting_method'] = hyperparams[3].str.strip()
        self.df['voting_policy'] = hyperparams[4].str.strip()

    def create_pivot_table(self, group_col: str) -> pd.DataFrame:
        """Create a pivot table for statistical analysis."""
        return self.df.pivot_table(
            values='Accuracy',
            index='Dataset/Fold',
            columns=group_col,
            aggfunc='first'
        )

    def perform_friedman_test(self, pivot_df: pd.DataFrame) -> Tuple[float, float]:
        """Perform Friedman test on the pivot table."""
        return stats.friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])

    def perform_nemenyi_test(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
        """Perform Nemenyi post-hoc test."""
        return posthoc_nemenyi_friedman(pivot_df)

    def perform_bonferroni_test(self, pivot_df: pd.DataFrame, control: str) -> pd.DataFrame:
        """
        Perform Bonferroni-corrected Wilcoxon signed-rank tests against a control.

        Adds effect size and percentage difference calculations.
        """
        control_data = pivot_df[control]
        results = {}
        n_comparisons = len(pivot_df.columns) - 1  # Subtract control

        for column in pivot_df.columns:
            if column != control:
                # Calculate effect size (median difference)
                effect_size = np.median(pivot_df[column] - control_data)

                # Perform Wilcoxon test
                statistic, p_value = stats.wilcoxon(
                    control_data,
                    pivot_df[column],
                    alternative='two-sided'
                )

                # Calculate percentage of improvement/degradation
                diff_percentage = ((pivot_df[column].mean() - control_data.mean())
                                   / control_data.mean() * 100)

                # Apply Bonferroni correction
                adjusted_p = min(p_value * n_comparisons, 1.0)

                results[column] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'adjusted_p': adjusted_p,
                    'effect_size': effect_size,
                    'diff_percentage': diff_percentage
                }

        return pd.DataFrame(results).T

    def analyze_hyperparameter(self,
                               param_name: str,
                               test_type: str = 'nemenyi',
                               control: str = None) -> Dict:
        """
        Analyze a specific hyperparameter using either Nemenyi or Bonferroni test.
        """
        pivot_df = self.create_pivot_table(param_name)
        friedman_stat, friedman_p = self.perform_friedman_test(pivot_df)
        significant_differences = friedman_p < self.alpha

        # Only perform post-hoc tests if significant differences are found
        post_hoc = None
        if significant_differences:
            if test_type == 'nemenyi':
                post_hoc = self.perform_nemenyi_test(pivot_df)
            else:  # bonferroni
                post_hoc = self.perform_bonferroni_test(pivot_df, control)

        # Calculate summary statistics
        summary = self.df.groupby(param_name)['Accuracy'].agg(['mean', 'std']).round(4)

        return {
            'summary': summary,
            'friedman_result': (friedman_stat, friedman_p),
            'post_hoc': post_hoc,
            'significant_differences': significant_differences
        }

    def visualize_results(self,
                          param_name: str,
                          results: Dict,
                          test_type: str = 'nemenyi',
                          control: str = None) -> plt.Figure:
        """Create visualizations for the analysis results."""
        if results['significant_differences'] and results['post_hoc'] is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        # Plot mean accuracy with error bars
        summary = results['summary']
        ax1.bar(range(len(summary)), summary['mean'])
        ax1.errorbar(range(len(summary)), summary['mean'],
                     yerr=summary['std'], fmt='none', color='black', capsize=5)
        ax1.set_xticks(range(len(summary)))
        ax1.set_xticklabels(summary.index, rotation=45, ha='right')
        ax1.set_title(f'Mean Accuracy by {param_name}')
        ax1.set_ylabel('Accuracy')

        # Plot post-hoc test results only if significant differences were found
        if results['significant_differences'] and results['post_hoc'] is not None:
            if test_type == 'nemenyi':
                sns.heatmap(results['post_hoc'], annot=True, cmap='RdYlGn_r', ax=ax2)
                ax2.set_title('Nemenyi Test p-values\n(lower values indicate significant differences)')
            else:  # bonferroni
                post_hoc = results['post_hoc']
                ax2.bar(range(len(post_hoc)), -np.log10(post_hoc['adjusted_p']))
                ax2.axhline(y=-np.log10(self.alpha), color='r', linestyle='--',
                            label=f'p={self.alpha} threshold')
                ax2.set_xticks(range(len(post_hoc)))
                ax2.set_xticklabels(post_hoc.index, rotation=45, ha='right')
                ax2.set_title('Bonferroni Test Results\n(-log10 adjusted p-value)')
                ax2.set_ylabel('-log10(adjusted p-value)')
                ax2.legend()

                # Percentage difference plot
                diff_percentages = post_hoc['diff_percentage']
                colors = ['g' if x > 0 else 'r' for x in diff_percentages]
                ax3.bar(range(len(diff_percentages)), diff_percentages, color=colors)
                ax3.set_xticks(range(len(diff_percentages)))
                ax3.set_xticklabels(diff_percentages.index, rotation=45, ha='right')
                ax3.set_title('Percentage Difference from Control')
                ax3.set_ylabel('Difference (%)')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_report(self,
                        param_name: str,
                        results: Dict,
                        test_type: str,
                        control: str,
                        output_path: str):
        """
        Generate a detailed text report of the statistical analysis results.
        """
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"Statistical Analysis Report - {param_name}\n")
            f.write("=" * 50 + "\n\n")

            # Summary Statistics
            f.write("Summary Statistics\n")
            f.write("-----------------\n")
            f.write(results['summary'].to_string())
            f.write("\n\n")

            # Friedman Test Results
            f.write("Friedman Test Results\n")
            f.write("--------------------\n")
            stat, p = results['friedman_result']
            f.write(f"Test Statistic: {stat:.4f}\n")
            f.write(f"P-value: {p:.4f}\n")
            f.write(f"Significance level (alpha): {self.alpha}\n\n")

            if results['significant_differences']:
                f.write("Significant differences were found between the configurations.\n\n")

                # Post-hoc Test Results
                f.write(f"Post-hoc Test Results ({test_type})\n")
                f.write("-" * 30 + "\n")
                if test_type == 'nemenyi':
                    f.write("Pairwise p-values (lower values indicate more significant differences):\n\n")
                    f.write(results['post_hoc'].round(4).to_string())
                else:  # bonferroni
                    f.write(f"Results compared to control ({control}):\n\n")
                    f.write(results['post_hoc'].round(4).to_string())

                    # Add explanation of new columns
                    f.write("\n\nAdditional Metrics Explanation:\n")
                    f.write("- effect_size: Median difference from the control configuration\n")
                    f.write(
                        "- diff_percentage: Percentage difference in mean accuracy from the control configuration\n")
            else:
                f.write(f"No significant differences were found between the configurations ")
                f.write(f"at the {self.alpha} significance level.\n")
                f.write("Post-hoc test was not performed.\n")


# Rest of the script remains the same (main function and other methods)
def main(csv_path: str, output_dir: str = None, alpha: float = 0.05):
    """Main function to run all analyses."""
    analyzer = KNNHyperparameterAnalyzer(csv_path, alpha=alpha)

    # Define analyses to perform
    analyses = [
        ('k', 'nemenyi', None),
        ('distance_metric', 'nemenyi', None),
        ('weighting_method', 'bonferroni', 'equal_weight'),
        ('voting_policy', 'bonferroni', 'majority_class')
    ]

    results = {}
    for param_name, test_type, control in analyses:
        print(f"\nAnalyzing {param_name}...")
        results[param_name] = analyzer.analyze_hyperparameter(param_name, test_type, control)

        # Print results to console
        print(f"\n{param_name} Summary Statistics:")
        print(results[param_name]['summary'])
        print(f"\nFriedman Test Results:")
        stat, p = results[param_name]['friedman_result']
        print(f"Statistic: {stat:.4f}")
        print(f"p-value: {p:.4f}")
        print(f"Significance level (alpha): {analyzer.alpha}")

        if results[param_name]['significant_differences']:
            print("\nSignificant differences found between configurations.")
            print(f"\nPost-hoc Test Results ({test_type}):")
            print(results[param_name]['post_hoc'].round(4))
        else:
            print(f"\nNo significant differences were found between the configurations ")
            print(f"at the {analyzer.alpha} significance level.")
            print("Post-hoc test was not performed.")

        # Create and save visualization
        fig = analyzer.visualize_results(param_name, results[param_name], test_type, control)
        if output_dir:
            # Save plot
            fig.savefig(f"{output_dir}/{param_name}_analysis.png")
            # Generate and save report
            analyzer.generate_report(param_name, results[param_name], test_type, control,
                                     f"{output_dir}/{param_name}_analysis.txt")
        plt.close()

    return results


if __name__ == "__main__":
    csv_path = "knn_base_results.csv"
    output_dir = "plots_and_tables\\knn_base\\hyperparameter_analysis"
else:
    csv_path = "Mushroom\\knn_base_results.csv"
    output_dir = "Mushroom\\plots_and_tables\\knn_base\\hyperparameter_analysis"

results = main(csv_path, output_dir, alpha=0.2)
print(f"Results and plots saved in folder Mushroom/plots_and_tables/knn_base/hyperparameter_analysis\n")