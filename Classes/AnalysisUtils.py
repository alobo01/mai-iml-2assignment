import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scikit_posthocs import posthoc_nemenyi


class AnalysisUtils:
    """
    A utility class for data analysis and visualization operations.

    This class provides methods for loading, preparing, and visualizing data
    with a focus on hyperparameter and metric analysis.
    """

    @staticmethod
    def load_and_prepare_data(csv_path: str,
                              params: List[str],
                              agg_funcs: Dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load CSV data and prepare aggregated statistics.

        Args:
            csv_path (str): Path to the CSV file
            params (List[str]): Parameters to group by
            agg_funcs (Dict[str, str]): Aggregation functions for each variable

        Returns:
            tuple: Original results and aggregated results DataFrames
        """
        # Load results
        results = pd.read_csv(csv_path)

        # Create aggregated results
        aggregated_results = results.groupby(params).agg(agg_funcs).reset_index()

        # Flatten column names
        variables = list(agg_funcs.keys())
        aggregated_results.columns = params + variables

        return results, aggregated_results

    @staticmethod
    def create_plots_folder(base_path: str) -> None:
        """
        Create a folder for plots if it doesn't exist.

        Args:
            base_path (str): Directory path for storing plots
        """
        Path(base_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_pairplot(cls,
                        data: pd.DataFrame,
                        params: List[str],
                        metric: str,
                        agg_func: str,
                        plots_path: str,
                        transposed: bool = False) -> None:
        """
        Create a comprehensive pairplot matrix for model hyperparameters.

        Args:
            data (pd.DataFrame): Input DataFrame
            params (List[str]): Parameters to analyze
            metric (str): Performance metric to visualize
            agg_func (str): Aggregation function for metric
            plots_path (str): Path to save the plot
            transposed (bool): Decides if lower triangular will be transposed
        """
        save_path = os.path.join(plots_path, 'hyperparameter_pairplot_matrix.png')
        n_params = len(params)

        # Create figure
        fig, axes = plt.subplots(n_params, n_params, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Color maps
        accuracy_cmap = 'YlOrRd'
        time_cmap = 'YlGnBu'

        data['Time'] = data['Time'].multiply(100)

        # Process each pair of parameters
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                param1 = params[i]
                param2 = params[j]

                # Label configuration
                xlabels = i == n_params - 1
                ylabels = j == 0

                # Set labels
                if xlabels:
                    ax.set_xlabel(param2)
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])

                if ylabels:
                    ax.set_ylabel(param1)
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                # Diagonal - Histograms
                if i == j:
                    cls._plot_diagonal_histogram(data, ax, param1, metric)

                # Upper triangle - Time heatmaps
                elif i < j:
                    cls._plot_time_heatmap(data, ax, param1, param2, xlabels, ylabels)

                # Lower triangle - Metric heatmaps
                else:
                    cls._plot_metric_heatmap(data, ax, param1, param2, metric, agg_func, xlabels, ylabels, transposed)

                # Rotate x-axis labels for bottom row
                if i == n_params - 1:
                    ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)

        plt.suptitle(f'Hyperparameter Relationships Matrix\n{metric} and Time Analysis',
                     fontsize=16, y=1.02)

        # Save and close the plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    @staticmethod
    def _plot_diagonal_histogram(data: pd.DataFrame,
                                 ax: plt.Axes,
                                 param: str,
                                 metric: str) -> None:
        """
        Plot histogram for diagonal of pairplot matrix.

        Args:
            data (pd.DataFrame): Input DataFrame
            ax (plt.Axes): Matplotlib axis to plot on
            param (str): Parameter to group by
            metric (str): Metric to visualize
        """
        param_groups = data.groupby(param)[metric]
        unique_values = data[param].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))

        for idx, (value, group) in enumerate(param_groups):
            ax.hist(group, alpha=0.7, color=colors[idx],
                    label=f'{value}', bins=20)

        ax.set_title(f'{metric} Distribution by {param}')
        ax.set_xlabel(f'{metric}')
        ax.set_ylabel('Count')
        ax.legend()

    @staticmethod
    def _plot_time_heatmap(data: pd.DataFrame,
                           ax: plt.Axes,
                           param1: str,
                           param2: str,
                           xlabels: bool,
                           ylabels: bool) -> None:
        """
        Plot time heatmap for upper triangle of pairplot matrix.

        Args:
            data (pd.DataFrame): Input DataFrame
            ax (plt.Axes): Matplotlib axis to plot on
            param1 (str): First parameter
            param2 (str): Second parameter
            xlabels (bool): Whether to show x-labels
            ylabels (bool): Whether to show y-labels
        """
        pivot_data = data.pivot_table(
            values='Time',
            index=param1,
            columns=param2,
            aggfunc='min'
        )

        sns.heatmap(pivot_data, ax=ax, xticklabels=xlabels, yticklabels=ylabels, cmap='YlGnBu',
                    annot=True, fmt='.2f', cbar=False)
        ax.set_title(f'Average Time')

    @staticmethod
    def _plot_metric_heatmap(data: pd.DataFrame,
                             ax: plt.Axes,
                             param1: str,
                             param2: str,
                             metric: str,
                             agg_func: str,
                             xlabels: bool,
                             ylabels: bool,
                             transposed: bool) -> None:
        """
        Plot metric heatmap for lower triangle of pairplot matrix.

        Args:
            data (pd.DataFrame): Input DataFrame
            ax (plt.Axes): Matplotlib axis to plot on
            param1 (str): First parameter
            param2 (str): Second parameter
            metric (str): Performance metric to visualize
            agg_func (str): Aggregation function for metric
            xlabels (bool): Whether to show x-labels
            ylabels (bool): Whether to show y-labels
        """
        if not transposed:
            pivot_data = data.pivot_table(
                values=metric,
                index=param1,
                columns=param2,
                aggfunc=agg_func
            )
        else:
            pivot_data = data.pivot_table(
                values=metric,
                index=param2,
                columns=param1,
                aggfunc=agg_func
            )

        sns.heatmap(pivot_data, ax=ax, xticklabels=xlabels, yticklabels=ylabels, cmap='YlOrRd',
                    annot=True, fmt='.3f', cbar=False)
        ax.set_title(f'{agg_func} {metric}')

    @staticmethod
    def plot_custom_heatmap(df: pd.DataFrame, plots_path: str) -> None:
        """
        Create a custom heatmap showing correlations, distributions, and scatter plots.

        Args:
            df (pd.DataFrame): Input DataFrame to visualize
            plots_path: Path to plots folder
        """
        save_path = os.path.join(plots_path, 'metrics_correlations_matrix.png')

        # Calculate Pearson Correlation Matrix
        corr_matrix = df.corr()
        metrics = df.columns
        n = len(metrics)
        fig, axes = plt.subplots(n, n, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i > j:  # Lower triangular: Pairwise Pearson Correlation
                    sns.heatmap([[corr_matrix.iloc[i, j]]],
                                annot=True, fmt=".2f", cbar=False,
                                xticklabels=False, yticklabels=False,
                                cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
                elif i == j:  # Diagonal: Metric Distribution
                    sns.histplot(df[metrics[i]], kde=True, ax=ax, color="skyblue")
                    ax.set_ylabel("")
                else:  # Upper triangular: Scatter Plots
                    ax.scatter(df[metrics[j]], df[metrics[i]], alpha=0.6, color="purple", s=10)
                    ax.set_ylabel("")
                    ax.set_xlabel("")

                if i != n - 1:
                    ax.set_xticks([])
                if j != 0:
                    ax.set_yticks([])

        # Add metric names as labels only for the edges
        for ax, label in zip(axes[-1, :], metrics):
            ax.set_xlabel(label, rotation=45, ha="right")
        for ax, label in zip(axes[:, 0], metrics):
            ax.set_ylabel(label, rotation=0, ha="right", labelpad=30)

        plt.suptitle("Custom Clustering Metric Matrix", fontsize=16, y=0.92)

        # Save and close the plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    @classmethod
    def perform_statistical_comparison(
            cls,
            data: pd.DataFrame,
            parameter: str,
            metric: str,
            test_type: str = 'pairwise',
            control_value: Any = None,
            alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison of a metric across different parameter values.

        Args:
            data (pd.DataFrame): Input DataFrame
            parameter (str): Parameter to compare
            metric (str): Performance metric to analyze
            test_type (str): Type of comparison - 'pairwise' or 'control'
            control_value (Any, optional): Reference value for control comparison
            alpha (float): Significance level for statistical tests

        Returns:
            Dict containing statistical test results
        """
        # Validate inputs
        if parameter not in data.columns:
            raise ValueError(f"Parameter {parameter} not found in DataFrame")
        if metric not in data.columns:
            raise ValueError(f"Metric {metric} not found in DataFrame")

        # Get unique parameter values
        param_values = data[parameter].unique()

        # Prepare data for Friedman test
        grouped_data = [
            data[data[parameter] == val][metric].values
            for val in param_values
        ]

        # Perform Friedman test
        friedman_statistic, friedman_pvalue = stats.friedmanchisquare(*grouped_data)

        # Check for significant differences
        if friedman_pvalue >= alpha:
            return {
                'test': 'Friedman',
                'statistic': friedman_statistic,
                'p_value': friedman_pvalue,
                'significant': False,
                'interpretation': 'No significant differences among groups'
            }

        # If significant, perform post-hoc test
        if test_type == 'pairwise':
            # Nemenyi post-hoc test
            posthoc_results = posthoc_nemenyi(grouped_data)

            # Create detailed pairwise comparison results
            results = {}
            for i in range(len(param_values)):
                for j in range(i + 1, len(param_values)):
                    val1, val2 = param_values[i], param_values[j]
                    results[f'{val1}_vs_{val2}'] = {
                        'group1': val1,
                        'group2': val2,
                        'p_value': posthoc_results.iloc[i, j],
                        'significant': posthoc_results.iloc[i, j] < (
                                    alpha / ((len(param_values) * (len(param_values) - 1)) / 2)),
                        'interpretation': (
                            'Significant difference' if posthoc_results.iloc[i, j] < (
                                        alpha / ((len(param_values) * (len(param_values) - 1)) / 2))
                            else 'No significant difference'
                        )
                    }


        elif test_type == 'control':

            if control_value is None:
                raise ValueError("Control value must be specified for control comparison")

            # Find index and data for control value

            control_index = list(param_values).index(control_value)

            control_group = grouped_data[control_index]

            # Perform Bonferroni-corrected Wilcoxon signed-rank test

            results = {}

            for i, val in enumerate(param_values):

                if val == control_value:
                    continue

                # Perform Wilcoxon signed-rank test

                statistic, p_value = stats.wilcoxon(control_group, grouped_data[i])

                # Bonferroni correction

                corrected_p_value = p_value * (len(param_values) - 1)

                corrected_p_value = min(corrected_p_value, 1.0)  # Cap at 1.0

                results[f'control_{control_value}_vs_{val}'] = {

                    'control_group': control_value,

                    'compared_group': val,

                    'statistic': statistic,

                    'p_value': corrected_p_value,

                    'significant': corrected_p_value < alpha,

                    'interpretation': (

                        'Significant difference' if corrected_p_value < alpha

                        else 'No significant difference'

                    )

                }

        else:
            raise ValueError("Invalid test type. Choose 'pairwise' or 'control'")

        # Add Friedman test details to the results
        return {
            'friedman_test': {
                'statistic': friedman_statistic,
                'p_value': friedman_pvalue,
                'significant': True,
                'interpretation': 'Significant differences among groups'
            },
            'posthoc_results': results
        }

    @classmethod
    def generate_statistical_report(
            cls,
            comparison_results: Dict[str, Any],
            output_path: str = None
    ) -> pd.DataFrame:
        """
        Generate a comprehensive statistical comparison report.

        Args:
            comparison_results (Dict[str, Any]): Results from statistical comparisons
            output_path (str, optional): Path to save the report

        Returns:
            pd.DataFrame with statistical comparison details
        """
        # Check if no significant differences were found
        if not comparison_results.get('friedman_test', {}).get('significant', False):
            report_df = pd.DataFrame({
                'Test': ['Friedman'],
                'Statistic': [comparison_results['friedman_test']['statistic']],
                'P-Value': [comparison_results['friedman_test']['p_value']],
                'Significant': [False],
                'Interpretation': ['No significant differences among groups']
            })

            # Optional: Save to CSV
            if output_path:
                report_df.to_csv(output_path, index=False)

            return report_df

        # Convert posthoc results to DataFrame
        report_data = []
        for comparison, results in comparison_results.get('posthoc_results', {}).items():
            report_data.append({
                'Comparison': comparison,
                'Group 1': results.get('group1', results.get('control_group', '')),
                'Group 2': results.get('group2', results.get('compared_group', '')),
                'P-Value': results['p_value'],
                'Significant': results['significant'],
                'Interpretation': results['interpretation']
            })

        report_df = pd.DataFrame(report_data)

        # Add Friedman test details
        friedman_row = pd.DataFrame({
            'Comparison': ['Friedman Test'],
            'Group 1': ['-'],
            'Group 2': ['-'],
            'P-Value': [comparison_results['friedman_test']['p_value']],
            'Significant': [True],
            'Interpretation': ['Significant differences among groups']
        })

        report_df = pd.concat([friedman_row, report_df], ignore_index=True)

        # Optional: Save to CSV
        if output_path:
            report_df.to_csv(output_path, index=False)

        return report_df

    @classmethod
    def summarize_statistical_comparisons(
            cls,
            comparison_reports: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create a summary of all statistical comparisons.

        Args:
            comparison_reports (Dict[str, pd.DataFrame]): Dictionary of comparison reports

        Returns:
            pd.DataFrame: Summary of significant differences across comparisons
        """
        summary_data = []

        for comparison_id, report in comparison_reports.items():
            if report.empty:
                continue

            # Determine if Friedman test showed significant differences
            friedman_row = report[report['Comparison'] == 'Friedman Test']
            is_friedman_significant = not friedman_row.empty and friedman_row['Significant'].iloc[0]

            # If Friedman test was not significant, add minimal information
            if not is_friedman_significant:
                summary_data.append({
                    'Comparison': comparison_id,
                    'Friedman P-Value': friedman_row['P-Value'].iloc[0] if not friedman_row.empty else None,
                    'Significant Comparisons': 0,
                    'Significance Percentage': 0,
                    'Most Significant Comparison': 'No significant differences',
                    'Lowest P-Value': None
                })
                continue

            # Count significant pairwise/control comparisons
            significant_rows = report[report['Significant'] == True]
            significant_rows = significant_rows[significant_rows['Comparison'] != 'Friedman Test']

            total_comparisons = report[report['Comparison'] != 'Friedman Test'].shape[0]
            significant_count = significant_rows.shape[0]

            summary_data.append({
                'Comparison': comparison_id,
                'Friedman P-Value': friedman_row['P-Value'].iloc[0],
                'Total Comparisons': total_comparisons,
                'Significant Comparisons': significant_count,
                'Significance Percentage': (
                                                       significant_count / total_comparisons) * 100 if total_comparisons > 0 else 0,
                'Most Significant Comparison': (
                    significant_rows['Comparison'].iloc[0]
                    if not significant_rows.empty
                    else 'No significant differences'
                ),
                'Lowest P-Value': (
                    significant_rows['P-Value'].min()
                    if not significant_rows.empty
                    else None
                )
            })

        return pd.DataFrame(summary_data)

    @classmethod
    def bulk_statistical_comparisons(
            cls,
            data: pd.DataFrame,
            comparison_configs: List[Tuple[str, str, str, Optional[Any]]],
            output_dir: Optional[str] = None,
            alpha: float = 0.05
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform bulk statistical comparisons based on multiple configuration tuples.

        Args:
            data (pd.DataFrame): Input DataFrame
            comparison_configs (List[Tuple]): List of comparison configurations
                Each tuple contains:
                - parameter (str): Parameter to compare
                - metric (str): Performance metric
                - test_type (str): 'pairwise' or 'control'
                - control_value (Optional[Any]): Control value for 'control' test type
            output_dir (Optional[str]): Directory to save individual reports
            alpha (float): Significance level for statistical tests

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of statistical comparison reports
        """
        # Validate output directory if provided
        if output_dir:
            cls.create_plots_folder(output_dir)

        # Store results for each comparison
        comparison_reports = {}

        # Perform statistical comparisons for each configuration
        for param, metric, test_type, control_value in comparison_configs:
            # Generate a unique identifier for the comparison
            comparison_id = f"{param}_{metric}"

            try:
                # Perform statistical comparison
                comparison_results = cls.perform_statistical_comparison(
                    data=data,
                    parameter=param,
                    metric=metric,
                    test_type=test_type,
                    control_value=control_value,
                    alpha=alpha
                )

                # Generate report
                report = cls.generate_statistical_report(comparison_results)

                # Store the report
                comparison_reports[comparison_id] = comparison_results

                # Optionally save to CSV
                if output_dir:
                    output_path = os.path.join(output_dir, f"{comparison_id}_statistical_report.csv")
                    report.to_csv(output_path, index=False)

            except Exception as e:
                print(f"Error in comparison {param}_{metric}: {e}")
                comparison_reports[comparison_id] = pd.DataFrame()  # Empty DataFrame for failed comparisons

        return comparison_reports

    @classmethod
    def generate_comprehensive_report(
            cls,
            comparison_reports: Dict[str, Dict[str, Any]],
            output_dir: str
    ) -> None:
        """
        Generate comprehensive statistical reports for each parameter.

        Args:
            comparison_reports (Dict[str, Dict[str, Any]]): Statistical comparison results
            output_dir (str): Directory to save report files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Group comparison reports by their parameter
        parameter_metrics = {}
        for comparison_id, report_data in comparison_reports.items():
            param, metric = comparison_id.split('_', 1)
            if param not in parameter_metrics:
                parameter_metrics[param] = []
            parameter_metrics[param].append((metric, report_data))

        # Generate a report for each parameter
        for param, metric_reports in parameter_metrics.items():
            report_path = os.path.join(output_dir, f"{param}_statistical_report.txt")

            with open(report_path, 'w') as f:
                f.write(f"Statistical Analysis Report for Parameter: {param}\n")
                f.write("=" * 50 + "\n\n")

                # Friedman Test Results Table
                f.write("FRIEDMAN TEST RESULTS\n")
                f.write("-" * 20 + "\n")
                f.write("{:<20} {:<15} {:<15} {:<15}\n".format("Metric", "Statistic", "P-Value", "Significant"))
                f.write("-" * 65 + "\n")

                # Store metrics with significant differences for detailed Nemenyi tests
                significant_metrics = []

                for metric, comparison_result in metric_reports:
                    print(metric)
                    print(comparison_result)
                    # Check if Friedman test was significant
                    friedman_test = comparison_result.get('friedman_test', {})
                    statistic = friedman_test.get('statistic', 'N/A')
                    p_value = friedman_test.get('p_value', 'N/A')
                    significant = friedman_test.get('significant', False)

                    f.write("{:<20} {:<15.4f} {:<15.4f} {:<15}\n".format(
                        metric,
                        statistic if isinstance(statistic, (int, float)) else 0,
                        p_value if isinstance(p_value, (int, float)) else 0,
                        str(significant)
                    ))

                    # Track metrics with significant differences
                    if significant:
                        significant_metrics.append((metric, comparison_result))

                # Detailed Nemenyi Test Results
                if significant_metrics:
                    f.write("\n\nNEMENYI POST-HOC TEST RESULTS\n")
                    f.write("=" * 30 + "\n")

                    for metric, comparison_result in significant_metrics:
                        f.write(f"\nMetric: {metric}\n")
                        f.write("-" * 20 + "\n")

                        # Nemenyi Results Table
                        f.write("{:<20} {:<20} {:<15} {:<15}\n".format(
                            "Group 1", "Group 2", "P-Value", "Significant"
                        ))
                        f.write("-" * 70 + "\n")

                        posthoc_results = comparison_result.get('posthoc_results', {})
                        for comparison, result in posthoc_results.items():
                            f.write("{:<20} {:<20} {:<15.4f} {:<15}\n".format(
                                str(result.get('group1', 'N/A')),
                                str(result.get('group2', 'N/A')),
                                result.get('p_value', 0),
                                str(result.get('significant', False))
                            ))

    @classmethod
    def bulk_report_generation(
            cls,
            data: pd.DataFrame,
            comparison_configs: List[Tuple[str, str, str, Optional[Any]]],
            output_dir: str,
            test_method: str = 'friedman',
            alpha: float = 0.05
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform bulk statistical comparisons and generate reports.

        Args:
            data (pd.DataFrame): Input DataFrame
            comparison_configs (List[Tuple]): Comparison configurations
            output_dir (str): Directory to save reports
            test_method (str): Statistical test method
            alpha (float): Significance level

        Returns:
            Dict of statistical comparison reports
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Perform bulk statistical comparisons using AnalysisUtils
        comparison_reports = cls.bulk_statistical_comparisons(
            data=data,
            comparison_configs=comparison_configs,
            output_dir=None,  # We'll handle reporting separately
            alpha=alpha
        )

        # Generate comprehensive reports
        cls.generate_comprehensive_report(
            comparison_reports,
            output_dir
        )

        return comparison_reports