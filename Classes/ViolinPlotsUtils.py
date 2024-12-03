import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ViolinPlotter:
    def __init__(self, dataframe):
        """
        Initialize the ViolinPlotter with a pandas DataFrame.

        :param dataframe: pandas DataFrame containing the data.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.dataframe = dataframe

    @staticmethod
    def create_violin_plot(dataframe, x_col, y_col, add_jitter=True, jitter_alpha=0.7):
        """
        Create a violin plot with optional jittered points.

        :param dataframe: pandas DataFrame containing the data.
        :param x_col: Column name for the x-axis (categorical or discrete data).
        :param y_col: Column name for the y-axis (numerical data).
        :param add_jitter: Whether to add jittered points to the violin plot.
        :param jitter_alpha: Transparency level for the jittered points.
        :return: The created matplotlib figure.
        """
        if x_col not in dataframe.columns or y_col not in dataframe.columns:
            raise ValueError(f"Columns '{x_col}' or '{y_col}' not found in DataFrame.")

        # Create a dedicated figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the violin plot
        sns.violinplot(data=dataframe, x=x_col, y=y_col, inner=None, ax=ax, color="skyblue", linewidth=1)

        # Add jittered points
        if add_jitter:
            sns.stripplot(data=dataframe, x=x_col, y=y_col, color='black', alpha=jitter_alpha, jitter=True, ax=ax)

        # Plot customization
        ax.set_title(f"{y_col} by {x_col}", fontsize=14)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        return fig

    @staticmethod
    def createViolinPlots(df: pd.DataFrame, x_cols: List[str], y_cols: List[str], plotsPath: str = "violinPlots", filePreffix: str = "violin", add_jitter=True, jitter_alpha=0.2):
        """
        Create multiple violin plots and save them as files.

        :param df: pandas DataFrame containing the data.
        :param x_cols: List of column names for the x-axis.
        :param y_cols: List of column names for the y-axis.
        :param plotsPath: Directory to save the plots.
        :param filePreffix: Prefix for the saved plot filenames.
        :param add_jitter: Whether to add jittered points to the violin plot.
        :param jitter_alpha: Transparency level for the jittered points.
        """
        os.makedirs(plotsPath, exist_ok=True)
        for x_col in x_cols:
            for y_col in y_cols:
                if x_col in df.columns and y_col in df.columns:
                    # Generate the plot and save it
                    fig = ViolinPlotter.create_violin_plot(df, x_col, y_col, add_jitter=add_jitter, jitter_alpha=jitter_alpha)
                    file_name = f"{filePreffix}_{x_col}_vs_{y_col}.png"
                    fig.savefig(os.path.join(plotsPath, file_name))
                    plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    # Example DataFrame with at least 3 x_cols and 3 y_cols
    data = {
        'Category1': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Category2': ['X', 'X', 'Y', 'Y', 'Z', 'Z'],
        'Category3': ['P', 'Q', 'P', 'Q', 'P', 'Q'],
        'Values1': [3, 5, 7, 8, 2, 3],
        'Values2': [10, 20, 30, 40, 50, 60],
        'Values3': [15, 25, 35, 45, 55, 65]
    }
    df = pd.DataFrame(data)

    # Specify x_cols and y_cols
    x_cols = ['Category1', 'Category2', 'Category3']
    y_cols = ['Values1', 'Values2', 'Values3']

    # Create multiple violin plots and save them as files
    ViolinPlotter.createViolinPlots(df, x_cols, y_cols, add_jitter=True)
