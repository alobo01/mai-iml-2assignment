import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        fig, axes = plt.subplots(n_params, n_params, figsize=(8, 8))
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
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    @classmethod
    def create_separate_pairplots(cls,
                                  data: pd.DataFrame,
                                  params: List[str],
                                  metric: str,
                                  agg_func: str,
                                  plots_path: str) -> None:
        """
        Create separate plots for each pair of parameters.

        Args:
            data (pd.DataFrame): Input DataFrame
            params (List[str]): Parameters to analyze
            metric (str): Performance metric to visualize
            agg_func (str): Aggregation function for metric
            plots_path (str): Path to save the plots
        """
        n_params = len(params)

        # Ensure folder exists
        os.makedirs(plots_path, exist_ok=True)

        # Ensure Time column is scaled appropriately
        if  data['Time'].max()<1:
            data['Time'] = data['Time'].multiply(100)

        # Iterate over each pair of parameters
        for i in range(n_params):
            for j in range(n_params):
                param1 = params[i]
                param2 = params[j]

                # Generate a separate figure for each plot
                fig, ax = plt.subplots(figsize=(6, 6))

                # Diagonal - Histograms
                if i == j:
                    cls._plot_diagonal_histogram(data, ax, param1, metric)
                    plot_title = f"Histogram_{param1}"

                # Upper triangle - Time heatmaps
                elif i < j:
                    cls._plot_time_heatmap(data, ax, param1, param2, xlabels=True, ylabels=True)
                    plot_title = f"Time_Heatmap_{param1}_vs_{param2}"

                # Lower triangle - Metric heatmaps
                else:
                    cls._plot_metric_heatmap(data, ax, param1, param2, metric, agg_func, xlabels=True, ylabels=True)
                    plot_title = f"Metric_Heatmap_{param1}_vs_{param2}"

                # Configure plot titles and labels
                ax.set_title(plot_title, fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)

                # Save each plot with a unique name
                save_path = os.path.join(plots_path, f"{plot_title}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
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
        fig, axes = plt.subplots(n, n, figsize=(6, 6))
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
                    ax.scatter(df[metrics[j]], df[metrics[i]], alpha=0.6, color="purple", s=2)
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
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    @staticmethod
    def extract_best_runs(results_df: pd.DataFrame, metrics: List[str] = ['ARI', 'NMI', 'DBI', 'Silhouette', 'CHS']):
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
            max_row = results_df.loc[results_df[metric].idxmax()] if metric != 'DBI' else results_df.loc[results_df[metric].idxmin()]
            max_metrics_dict[metric] = max_row.to_dict()

        return max_metrics_dict

    @staticmethod
    def generate_best_runs_table(best_runs: dict, best_runs_path: str, features_explored: List[str]):
        """
        Generates a table with best runs for each metric and saves it to a text file.

        Parameters:
        best_runs (dict): Dictionary containing the best rows for each metric
        best_runs_path (str): Path to save the output text file
        features_explored (List[str]): List of features used in the exploration

        Returns:
        str: Formatted table as a string
        """
        # Create a table header
        table_header = "| Metric | " + " | ".join(features_explored) + " | Best Value |"
        table_separator = "|" + "---|" * (len(features_explored) + 2)

        # Initialize table rows
        table_rows = [table_header, table_separator]

        # Populate table rows
        for metric, best_row in best_runs.items():
            # Extract feature values for this run
            feature_values = [str(best_row.get(feature, 'N/A')) for feature in features_explored]

            # Create row with metric, feature values, and best metric value
            row = f"| {metric} | " + " | ".join(feature_values) + f" | {best_row.get(metric, 'N/A')} |"
            table_rows.append(row)

        # Convert table to string
        table_str = "\n".join(table_rows)

        # Print the table
        print(table_str)

        # Ensure the directory exists
        os.makedirs(best_runs_path, exist_ok=True)
        file_path = os.path.join(best_runs_path, 'best_runs_table.md')

        # Write table to file
        with open(file_path, 'w') as f:
            f.write(table_str)

        return table_str

    @staticmethod
    def plot_best_runs(best_runs: dict, labels_df: pd.DataFrame, pca_dataset_df: pd.DataFrame,
                       umap_dataset_df: pd.DataFrame, best_runs_path: str):
        """
        Generate scatter plots for best runs using both PCA-transformed and UMAP-transformed data,
        each with separate legends. The plots will be saved with names indicating PCA or UMAP.

        Parameters:
        best_runs (dict): Dictionary of best runs for each metric.
        labels_df (pd.DataFrame): DataFrame containing cluster labels for each algorithm.
        pca_dataset_df (pd.DataFrame): PCA-transformed dataset with original class labels.
        umap_dataset_df (pd.DataFrame): UMAP-transformed dataset with original class labels.
        best_runs_path (str): Path to store the generated plots.
        """
        # Ensure the directory exists
        os.makedirs(best_runs_path, exist_ok=True)

        # Define marker shapes for true labels
        marker_shapes = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']

        # Color palette for clusters
        color_palette = plt.cm.get_cmap('tab10')

        # Iterate through best runs for each metric
        for metric, run in best_runs.items():
            # Get the algorithm name for this run
            algorithm = run.get('Algorithm', 'Unknown')

            # -------------------
            # PCA Visualization
            # -------------------
            plt.figure(figsize=(8, 6))

            # Get cluster labels for this run
            cluster_labels = labels_df[algorithm].values

            # Get true labels
            true_labels = pca_dataset_df['Class'].values

            # Unique true labels and clusters
            unique_true_labels = np.unique(true_labels)
            unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude -1

            # Plot noisy points if they exist
            noisy_mask = cluster_labels == -1
            if noisy_mask.any():
                plt.scatter(
                    pca_dataset_df.iloc[noisy_mask, 1],
                    pca_dataset_df.iloc[noisy_mask, 2],
                    c='lightgray',
                    marker='X',
                    alpha=0.5,
                    label='Noisy Points'
                )

            # Plots for colored clusters
            for cluster_idx, cluster in enumerate(unique_clusters):
                cluster_mask = cluster_labels == cluster

                for true_label_idx, true_label in enumerate(unique_true_labels):
                    true_label_mask = true_labels == true_label
                    mask = cluster_mask & true_label_mask

                    plt.scatter(
                        pca_dataset_df.iloc[mask, 1],  # First PCA component
                        pca_dataset_df.iloc[mask, 2],  # Second PCA component
                        c=[color_palette(cluster_idx)],  # Cluster color
                        marker=marker_shapes[true_label_idx % len(marker_shapes)],  # True label marker
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5,
                        label=f'Cluster {cluster}' if true_label_idx == 0 else ''
                    )

            plt.title(f'Best Run for {metric} - {algorithm} (PCA)')
            plt.xlabel('First PCA Component')
            plt.ylabel('Second PCA Component')

            from matplotlib.lines import Line2D

            # Create color legend handles
            color_handles = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_palette(i), markersize=10,
                       label=f'Cluster {cluster}')
                for i, cluster in enumerate(unique_clusters)
            ]

            # Create shape legend handles
            shape_handles = [
                Line2D([0], [0], marker=marker_shapes[i % len(marker_shapes)], color='k',
                       markerfacecolor='gray', markersize=10,
                       label=f'True Label {true_label}')
                for i, true_label in enumerate(unique_true_labels)
            ]

            # Add noisy points to legend only if they exist
            if noisy_mask.any():
                noisy_handle = Line2D([0], [0], marker='X', color='lightgray',
                                      markerfacecolor='lightgray', markersize=10,
                                      label='Noisy Points')
                color_handles.append(noisy_handle)

            first_legend = plt.legend(handles=color_handles, title='Clusters',
                                      loc='center left', bbox_to_anchor=(1.02, 0.5))
            plt.gca().add_artist(first_legend)
            plt.legend(handles=shape_handles, title='True Labels',
                       loc='center left', bbox_to_anchor=(1.02, 0.1))

            plt.tight_layout()

            # Save the PCA plot
            pca_plot_filename = os.path.join(best_runs_path, f'best_run_{metric}_pca.png')
            plt.savefig(pca_plot_filename, dpi=100, bbox_inches='tight')
            plt.close()

            # -------------------
            # UMAP Visualization
            # -------------------
            plt.figure(figsize=(8, 6))

            # For UMAP, we assume the same clusters apply
            cluster_labels = labels_df[algorithm].values
            true_labels = umap_dataset_df['Class'].values

            unique_true_labels = np.unique(true_labels)
            unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude -1

            # Plot noisy points for UMAP if they exist
            noisy_mask = cluster_labels == -1
            if noisy_mask.any():
                plt.scatter(
                    umap_dataset_df.iloc[noisy_mask, 1],
                    umap_dataset_df.iloc[noisy_mask, 2],
                    c='lightgray',
                    marker='X',
                    alpha=0.5,
                    label='Noisy Points'
                )

            # Plots for colored clusters
            for cluster_idx, cluster in enumerate(unique_clusters):
                cluster_mask = cluster_labels == cluster

                for true_label_idx, true_label in enumerate(unique_true_labels):
                    true_label_mask = true_labels == true_label
                    mask = cluster_mask & true_label_mask

                    plt.scatter(
                        umap_dataset_df.iloc[mask, 1],  # First UMAP component
                        umap_dataset_df.iloc[mask, 2],  # Second UMAP component
                        c=[color_palette(cluster_idx)],
                        marker=marker_shapes[true_label_idx % len(marker_shapes)],
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5,
                        label=f'Cluster {cluster}' if true_label_idx == 0 else ''
                    )


            plt.title(f'Best Run for {metric} - {algorithm} (UMAP)')
            plt.xlabel('First UMAP Component')
            plt.ylabel('Second UMAP Component')

            # Recreate legend handles for UMAP
            color_handles = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_palette(i), markersize=10,
                       label=f'Cluster {cluster}')
                for i, cluster in enumerate(unique_clusters)
            ]

            shape_handles = [
                Line2D([0], [0], marker=marker_shapes[i % len(marker_shapes)], color='k',
                       markerfacecolor='gray', markersize=10,
                       label=f'True Label {true_label}')
                for i, true_label in enumerate(unique_true_labels)
            ]

            # Add noisy points to legend only if they exist
            if noisy_mask.any():
                noisy_handle = Line2D([0], [0], marker='X', color='lightgray',
                                      markerfacecolor='lightgray', markersize=10,
                                      label='Noisy Points')
                color_handles.append(noisy_handle)

            first_legend = plt.legend(handles=color_handles, title='Clusters',
                                      loc='center left', bbox_to_anchor=(1.02, 0.5))
            plt.gca().add_artist(first_legend)
            plt.legend(handles=shape_handles, title='True Labels',
                       loc='center left', bbox_to_anchor=(1.02, 0.1))

            plt.tight_layout()

            # Save the UMAP plot
            umap_plot_filename = os.path.join(best_runs_path, f'best_run_{metric}_umap.png')
            plt.savefig(umap_plot_filename, dpi=100, bbox_inches='tight')
            plt.close()

    @staticmethod
    def totalAnalysis(results_df: pd.DataFrame, labels_df: pd.DataFrame, pca_dataset_df: pd.DataFrame, umap_dataset_df: pd.DataFrame, plots_path: str, features_explored: List[str], metrics: List[str] = ['ARI', 'NMI', 'DBI', 'Silhouette', 'CHS']):

        # Pair-plot for Hyperparameter Analysis
        AnalysisUtils.create_pairplot(
            data=results_df,
            params=features_explored,
            metric='ARI',  # Using ARI as primary performance metric
            agg_func='max',
            plots_path=plots_path
        )

        # Separate Pair-plot for Hyperparameter Analysis
        AnalysisUtils.create_separate_pairplots(
            data=results_df,
            params=features_explored,
            metric='ARI',  # Using ARI as primary performance metric
            agg_func='max',
            plots_path=os.path.join(plots_path,"pairPlots")
        )

        # 2. Create Custom Heatmap for Metric Correlations
        AnalysisUtils.plot_custom_heatmap(results_df[metrics], plots_path=plots_path)

        # Violin Analysis
        ViolinPlotter.createViolinPlots(results_df, features_explored, metrics, plotsPath = os.path.join(plots_path, "violinPlots"))

        # Best Result Plots / tables
        best_runs_path = os.path.join(plots_path, "bestRuns")
        best_runs = AnalysisUtils.extract_best_runs(results_df, metrics=metrics)
        AnalysisUtils.generate_best_runs_table(best_runs, best_runs_path, features_explored)

        # PCA Plots
        AnalysisUtils.plot_best_runs(best_runs, labels_df, pca_dataset_df, umap_dataset_df, best_runs_path)

    @staticmethod
    def predicted_k_vs_iterations(results_df: pd.DataFrame, plots_path: str):
        grouped = results_df.groupby("max_iterations").agg({
            "Predicted k": "mean",
            "max_clusters": "first"  # Assuming max_clusters doesn't vary within iterations
        }).reset_index()

        plt.figure(figsize=(12, 8))

        unique_iterations = sorted(results_df["max_iterations"].unique())
        for max_iter in unique_iterations:
            subset = results_df[results_df["max_iterations"] == max_iter]
            x_positions = np.full(len(subset), max_iter)  # Fixed x positions for each max_iterations
            plt.scatter(
                x_positions, subset["Predicted k"],
                color="gray", alpha=0.7, edgecolor="black",
                label="Actual Predicted k" if max_iter == unique_iterations[0] else None
            )

        # Overlay the mean Predicted k and max_clusters as lines
        plt.plot(grouped["max_iterations"], grouped["Predicted k"], label="Mean Predicted k", marker="o", linestyle="-",
                 color="blue")
        plt.plot(grouped["max_iterations"], grouped["max_clusters"], label="Max Clusters", marker="s", linestyle="--",
                 color="green")

        # Customize the plot
        plt.xticks(unique_iterations)
        plt.xlabel("Max Iterations")
        plt.ylabel("Predicted k / Max Clusters")
        plt.title("X-Means Analysis: Points for Predicted k with Mean and Max Clusters")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # Move legend to the side
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        save_path = os.path.join(plots_path, "xmeans_analysis_predicted_k_vs_iterations.png")
        plt.savefig(save_path, dpi=300)
        plt.close()  # Close the plot to free up memory
        print(f"Plot saved at: {save_path}")

    @staticmethod
    def max_k_vs_actual_k(results_df: pd.DataFrame, plots_path: str):
        """
        Generates a plot of max_clusters (x-axis) vs. the mean of Predicted_k (line) and individual predictions (points).

        Parameters:
            results_df (pd.DataFrame): DataFrame containing 'max_clusters' and 'Predicted_k'.
            plots_path (str): Path to save the generated plot.

        Returns:
            None
        """
        # Group by 'max_clusters' and calculate mean of 'Predicted_k'
        grouped = results_df.groupby('max_clusters')['Predicted k'].mean()

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(grouped.index, grouped.values, label='Mean Predicted k', marker='o', color='blue', linewidth=2)
        plt.scatter(results_df['max_clusters'], results_df['Predicted k'], color='red', alpha=0.6,
                    label='Individual Predicted k', zorder=5)

        # Customizations
        plt.title('Max Clusters vs Predicted k', fontsize=14)
        plt.xlabel('Max Clusters', fontsize=12)
        plt.ylabel('Predicted k', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)

        # Save the plot
        save_path = os.path.join(plots_path, "xmeans_analysis_max_k_vs_actual_k.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved at: {save_path}")

    @staticmethod
    def xmeans_analysis(results_df: pd.DataFrame, plots_path: str):
        #AnalysisUtils.predicted_k_vs_iterations(results_df, plots_path)
        AnalysisUtils.max_k_vs_actual_k(results_df, plots_path)



