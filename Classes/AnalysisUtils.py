import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


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
                        plots_path: str) -> None:
        """
        Create a comprehensive pairplot matrix for model hyperparameters.

        Args:
            data (pd.DataFrame): Input DataFrame
            params (List[str]): Parameters to analyze
            metric (str): Performance metric to visualize
            agg_func (str): Aggregation function for metric
            plots_path (str): Path to save the plot
        """
        save_path = os.path.join(plots_path, 'hyperparameter_pairplot_matrix.png')
        n_params = len(params)

        # Create figure
        fig, axes = plt.subplots(n_params, n_params, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Color maps
        accuracy_cmap = 'YlOrRd'
        time_cmap = 'YlGnBu'

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

                # Lower triangle - Accuracy heatmaps
                else:
                    cls._plot_accuracy_heatmap(data, ax, param1, param2, metric, agg_func, xlabels, ylabels)

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
    def _plot_accuracy_heatmap(data: pd.DataFrame,
                               ax: plt.Axes,
                               param1: str,
                               param2: str,
                               metric: str,
                               agg_func: str,
                               xlabels: bool,
                               ylabels: bool) -> None:
        """
        Plot accuracy heatmap for lower triangle of pairplot matrix.

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
        pivot_data = data.pivot_table(
            values=metric,
            index=param1,
            columns=param2,
            aggfunc=agg_func
        )

        sns.heatmap(pivot_data, ax=ax, xticklabels=xlabels, yticklabels=ylabels, cmap='YlOrRd',
                    annot=True, fmt='.3f', cbar=False)
        ax.set_title(f'{agg_func} {metric}')

    @staticmethod
    def plot_custom_heatmap(df: pd.DataFrame) -> None:
        """
        Create a custom heatmap showing correlations, distributions, and scatter plots.

        Args:
            df (pd.DataFrame): Input DataFrame to visualize
        """
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
        plt.show()

    @classmethod
    def perform_statistical_comparison(
            cls,
            data: pd.DataFrame,
            parameter: str,
            metric: str,
            test_type: str = 'pairwise',
            control_value: Any = None,
            alpha: float = 0.05,
            test_method: str = 'mannwhitneyu'
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
            test_method (str): Statistical test to use ('mannwhitneyu' or 't-test')

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

        # Select appropriate statistical test
        if test_method == 'mannwhitneyu':
            test_func = cls._mann_whitney_test
        elif test_method == 't-test':
            test_func = cls._t_test
        else:
            raise ValueError("Invalid test method. Choose 'mannwhitneyu' or 't-test'")

        # Perform comparisons based on test type
        if test_type == 'pairwise':
            return cls._pairwise_comparison(
                data, parameter, metric, param_values,
                test_func, alpha
            )
        elif test_type == 'control':
            if control_value is None:
                raise ValueError("Control value must be specified for control comparison")
            return cls._control_comparison(
                data, parameter, metric, control_value,
                test_func, alpha
            )
        else:
            raise ValueError("Invalid test type. Choose 'pairwise' or 'control'")

    @staticmethod
    def _mann_whitney_test(group1: np.ndarray, group2: np.ndarray, alpha: float) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test between two groups.

        Args:
            group1 (np.ndarray): First group of values
            group2 (np.ndarray): Second group of values
            alpha (float): Significance level

        Returns:
            Dict with test statistics and interpretation
        """
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        return {
            'test': 'Mann-Whitney U',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'interpretation': (
                'Significant difference' if p_value < alpha
                else 'No significant difference'
            )
        }

    @staticmethod
    def _t_test(group1: np.ndarray, group2: np.ndarray, alpha: float) -> Dict[str, Any]:
        """
        Perform independent t-test between two groups.

        Args:
            group1 (np.ndarray): First group of values
            group2 (np.ndarray): Second group of values
            alpha (float): Significance level

        Returns:
            Dict with test statistics and interpretation
        """
        statistic, p_value = stats.ttest_ind(group1, group2)

        return {
            'test': 'Independent t-test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'interpretation': (
                'Significant difference' if p_value < alpha
                else 'No significant difference'
            )
        }

    @classmethod
    def _pairwise_comparison(
            cls,
            data: pd.DataFrame,
            parameter: str,
            metric: str,
            param_values: List[Any],
            test_func: callable,
            alpha: float
    ) -> Dict[str, Any]:
        """
        Perform pairwise comparisons for all parameter values.

        Args:
            data (pd.DataFrame): Input DataFrame
            parameter (str): Parameter to compare
            metric (str): Performance metric
            param_values (List[Any]): Unique parameter values
            test_func (callable): Statistical test function
            alpha (float): Significance level

        Returns:
            Dict of pairwise comparison results
        """
        results = {}

        for i in range(len(param_values)):
            for j in range(i + 1, len(param_values)):
                val1, val2 = param_values[i], param_values[j]

                # Extract metric values for each parameter value
                group1 = data[data[parameter] == val1][metric].values
                group2 = data[data[parameter] == val2][metric].values

                # Perform statistical test
                comparison_result = test_func(group1, group2, alpha)

                # Store results with a descriptive key
                results[f'{val1}_vs_{val2}'] = {
                    **comparison_result,
                    'group1': val1,
                    'group2': val2
                }

        return results

    @classmethod
    def _control_comparison(
            cls,
            data: pd.DataFrame,
            parameter: str,
            metric: str,
            control_value: Any,
            test_func: callable,
            alpha: float
    ) -> Dict[str, Any]:
        """
        Compare all parameter values against a control value.

        Args:
            data (pd.DataFrame): Input DataFrame
            parameter (str): Parameter to compare
            metric (str): Performance metric
            control_value (Any): Reference value to compare against
            test_func (callable): Statistical test function
            alpha (float): Significance level

        Returns:
            Dict of control comparison results
        """
        results = {}

        # Get control group metrics
        control_group = data[data[parameter] == control_value][metric].values

        # Compare other parameter values against control
        for val in data[parameter].unique():
            if val == control_value:
                continue

            # Extract metric values for current parameter value
            current_group = data[data[parameter] == val][metric].values

            # Perform statistical test
            comparison_result = test_func(control_group, current_group, alpha)

            # Store results
            results[f'control_{control_value}_vs_{val}'] = {
                **comparison_result,
                'control_group': control_value,
                'compared_group': val
            }

        return results

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
        # Convert results to DataFrame
        report_data = []
        for comparison, results in comparison_results.items():
            report_data.append({
                'Comparison': comparison,
                'Test': results['test'],
                'Statistic': results['statistic'],
                'P-Value': results['p_value'],
                'Significant': results['significant'],
                'Interpretation': results['interpretation']
            })

        report_df = pd.DataFrame(report_data)

        # Optional: Save to CSV
        if output_path:
            report_df.to_csv(output_path, index=False)

        return report_df

    @classmethod
    def bulk_statistical_comparisons(
            cls,
            data: pd.DataFrame,
            comparison_configs: List[Tuple[str, str, str, Optional[Any]]],
            output_dir: Optional[str] = None,
            test_method: str = 'mannwhitneyu',
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
            test_method (str): Statistical test to use ('mannwhitneyu' or 't-test')
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
                    test_method=test_method,
                    alpha=alpha
                )

                # Generate report
                report = cls.generate_statistical_report(comparison_results)

                # Store the report
                comparison_reports[comparison_id] = report

                # Optionally save to CSV
                if output_dir:
                    output_path = os.path.join(output_dir, f"{comparison_id}_statistical_report.csv")
                    report.to_csv(output_path, index=False)

            except Exception as e:
                print(f"Error in comparison {param}_{metric}: {e}")
                comparison_reports[comparison_id] = pd.DataFrame()  # Empty DataFrame for failed comparisons

        return comparison_reports

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

            # Count significant and non-significant comparisons
            significant_count = report[report['Significant'] == True].shape[0]
            total_comparisons = report.shape[0]

            # Extract main insights
            significant_rows = report[report['Significant'] == True]

            summary_data.append({
                'Comparison': comparison_id,
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