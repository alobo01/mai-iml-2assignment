


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

    def create_violin_plot(self, x_col, y_col, add_jitter=True, jitter_alpha=0.7):
        """
        Create a violin plot with optional jittered points.

        :param x_col: Column name for the x-axis (categorical or discrete data).
        :param y_col: Column name for the y-axis (numerical data).
        :param add_jitter: Whether to add jittered points to the violin plot.
        :param jitter_alpha: Transparency level for the jittered points.
        """
        if x_col not in self.dataframe.columns or y_col not in self.dataframe.columns:
            raise ValueError(f"Columns '{x_col}' or '{y_col}' not found in DataFrame.")

        # Create the violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.dataframe, x=x_col, y=y_col, inner=None, color="skyblue", linewidth=1)

        # Add jittered points
        if add_jitter:
            sns.stripplot(data=self.dataframe, x=x_col, y=y_col, color='black', alpha=jitter_alpha, jitter=True)

        # Plot customization
        plt.title(f"Violin Plot with Jitter: {y_col} by {x_col}", fontsize=14)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    # Example DataFrame
    data = {
        'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Values': [3, 5, 7, 8, 2, 3]
    }
    df = pd.DataFrame(data)

    # Create an instance of ViolinPlotter
    plotter = ViolinPlotter(df)

    # Create a violin plot
    plotter.create_violin_plot('Category', 'Values')
