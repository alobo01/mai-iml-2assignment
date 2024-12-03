import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scikit_posthocs import posthoc_nemenyi
from torch.onnx._internal.fx._pass import Analysis

from Classes.ViolinPlotsUtils import ViolinPlotter


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
        fig, axes = plt.subplots(n_params, n_params, figsize=(10, 10))
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
                    cls._plot_metric_heatmap(data, ax, param1, param2, metric, agg_func, xlabels, ylabels)

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
        if len(unique_values) < 6:
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
        transposed = len(param1) > len(param2)
        if not transposed:
            pivot_data = data.pivot_table(
                values='Time',
                index=param1,
                columns=param2,
                aggfunc='min'
            )
        else:
            pivot_data = data.pivot_table(
                values='Time',
                index=param2,
                columns=param1,
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
                             ylabels: bool) -> None:
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
        transposed = len(param1) > len(param2)
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
        fig, axes = plt.subplots(n, n, figsize=(10, 10))
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

        plt.suptitle("Metric Correlations", fontsize=16, y=0.92)

        # Save and close the plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    @staticmethod
    def extract_best_runs(df: pd.DataFrame, metrics: List[str] = ['ARI', 'NMI', 'DBI', 'Silhouette', 'CHS']):
        """
        Load a CSV file and extract the rows with maximum values for specified metrics.

        Parameters:
        df (pd.DataFrame): DataFrame with extracted metrics

        Returns:
        dict: A dictionary with metric names as keys and corresponding rows with best values as values
        """


        # Dictionary to store results
        max_metrics_dict = {}

        # Find rows with best values for each metric
        for metric in metrics:
            # Find the row with the best value for the current metric
            max_row = df.loc[df[metric].idxmax()] if metric != 'DBI' else df.loc[df[metric].idxmin()]
            max_metrics_dict[metric] = max_row.to_dict()

        return max_metrics_dict

    @staticmethod
    def totalAnalysis(results_dataframe: pd.DataFrame, plots_path: str, features_explored: List[str] = ["n_clusters"], metrics: List[str] = ['ARI', 'NMI', 'DBI', 'Silhouette', 'CHS']):


        # Violin Analysis
        ViolinPlotter.createViolinPlots(results_dataframe, features_explored, metrics, plotsPath = os.path.join(plots_path,"violinPlots"))

        max_metrics_dict = AnalysisUtils.extract_best_runs(results_dataframe, metrics=metrics)

        # Best Result Plots / tables

        # PCA Plots
