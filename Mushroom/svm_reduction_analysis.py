import os
import pandas as pd
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    # Extract method names (everything after the last comma)
    df['Method'] = df['Model'].apply(lambda x: x.split(',')[-1].strip())
    # Pivot the data to get methods as columns and folds as rows
    accuracy_matrix = df.pivot(index='Dataset/Fold',
                               columns='Method',
                               values='Accuracy')
    return df, accuracy_matrix


def save_mean_metrics(df, dataset_path):
    # Calculate mean metrics for each method
    mean_metrics = df.groupby('Method').agg({
        'Accuracy': ['mean', 'std'],
        'F1': ['mean', 'std'],
        'Time': ['mean', 'std']
    }).round(4)

    # Rename columns for better readability
    mean_metrics.columns = ['Mean_Accuracy', 'Std_Accuracy',
                            'Mean_F1', 'Std_F1',
                            'Mean_Time', 'Std_Time']

    # Save to file
    file_path = os.path.join(dataset_path, "plots_and_tables/svm_reduction/svm_results_mean_metrics.txt")
    with open(file_path, 'w') as f:
        f.write("Mean Metrics by Method\n")
        f.write("=====================\n\n")
        f.write(mean_metrics.to_string())

    return mean_metrics


def perform_friedman_test(accuracy_matrix):
    statistic, p_value = stats.friedmanchisquare(*[accuracy_matrix[col] for col in accuracy_matrix.columns])
    return statistic, p_value


def perform_nemenyi_test(accuracy_matrix):
    return posthoc_nemenyi_friedman(accuracy_matrix)


def create_boxplot(df, dataset_path):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Method', y='Accuracy', data=df)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(dataset_path, "plots_and_tables/svm_reduction/svm_results_boxplot.png")
    plt.savefig(plot_path)
    plt.close()


def create_heatmap(nemenyi_matrix, dataset_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi_matrix, annot=True, cmap='RdYlBu', center=0.5)
    plt.tight_layout()
    plot_path = os.path.join(dataset_path, "plots_and_tables/svm_reduction/svm_results_heatmap.png")
    plt.savefig(plot_path)
    plt.close()


def write_results(dataset_path, friedman_stat, friedman_p, accuracy_matrix, nemenyi_matrix=None):
    file_path = os.path.join(dataset_path, "plots_and_tables/svm_reduction/svm_results_analysis.txt")
    with open(file_path, 'w') as f:
        f.write("Statistical Analysis of SVM Results\n")
        f.write("==================================\n\n")

        # Write summary statistics
        f.write("Summary Statistics:\n")
        f.write("-----------------\n")
        summary = accuracy_matrix.agg(['mean', 'std', 'min', 'max']).round(4)
        f.write(summary.to_string())
        f.write("\n\n")

        # Write Friedman test results
        f.write("Friedman Test Results:\n")
        f.write("---------------------\n")
        f.write(f"Statistic: {friedman_stat:.4f}\n")
        f.write(f"P-value: {friedman_p:.4f}\n\n")

        if nemenyi_matrix is not None:
            f.write("Nemenyi Test Results:\n")
            f.write("--------------------\n")
            f.write("P-values matrix:\n")
            f.write(nemenyi_matrix.round(4).to_string())
            f.write("\n\n")

        # Write interpretation
        f.write("Interpretation:\n")
        f.write("--------------\n")
        if friedman_p < 0.05:
            f.write("The Friedman test shows significant differences between methods (p < 0.05).\n")
            if nemenyi_matrix is not None:
                f.write("The Nemenyi test was performed to identify specific differences between methods.\n")
                sig_pairs = []
                for i in range(len(nemenyi_matrix.columns)):
                    for j in range(i + 1, len(nemenyi_matrix.columns)):
                        if nemenyi_matrix.iloc[i, j] < 0.05:
                            sig_pairs.append(f"{nemenyi_matrix.columns[i]} vs {nemenyi_matrix.columns[j]}")
                if sig_pairs:
                    f.write("\nSignificant differences found between:\n")
                    for pair in sig_pairs:
                        f.write(f"- {pair}\n")
        else:
            f.write("The Friedman test shows no significant differences between methods (p >= 0.05).\n")

    with open(file_path, 'r') as file:
        logs = file.read()
        print(logs)


if __name__ == "__main__":
    dataset_path = '..\\Mushroom'
else:
    dataset_path = 'Mushroom'

data_path = os.path.join(dataset_path, 'svm_mushroom_results_reduced.csv')
# Load and prepare data
df, accuracy_matrix = load_and_prepare_data(data_path)

# Always save mean metrics regardless of other tests
mean_metrics = save_mean_metrics(df, dataset_path)

# Create boxplot regardless of other tests
create_boxplot(df, dataset_path)

accuracy_values = df['Accuracy']

# Check if we should perform statistical tests
if np.ptp(accuracy_values) < 0.1:  # np.ptp() gives the range (max - min) of values
    print("Skipping Friedman test due to low accuracy variation.")

else:
    # Perform Friedman test
    friedman_stat, friedman_p = perform_friedman_test(accuracy_matrix)

    # If Friedman test is significant, perform Nemenyi test
    nemenyi_matrix = None
    if friedman_p < 0.05:
        nemenyi_matrix = perform_nemenyi_test(accuracy_matrix)
        create_heatmap(nemenyi_matrix, 'svm_results')

    # Write results to file
    write_results(dataset_path, friedman_stat, friedman_p, accuracy_matrix, nemenyi_matrix)
