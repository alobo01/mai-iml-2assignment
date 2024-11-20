import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from Mushroom.svm_base_analysis import analyze_model_performance, main
from classes.SVM import SVM

"""
This script performs a comprehensive analysis of SVM models on the Mushroom dataset using different 
kernels, C values, and data reduction methods. The analysis follows these main steps:

1. Evaluates 16 different SVM configurations (4 kernels × 4 C values) across 10 folds
2. Identifies the top 10 performing configurations based on accuracy
3. Applies Friedman-Nemenyi statistical test to determine significant differences between models
4. Selects the best performing model and evaluates it with different data reduction techniques

The reduction techniques evaluated are:
- EENTH (Enhanced Edited Nearest Neighbor with Triangle Height)
- GCNN (Generalized Condensed Nearest Neighbor)
- DROP3 (Decremental Reduction Optimization Procedure 3)
"""


def load_fold_data(fold_number: int, dataset_path: str, reduction_method: Optional[str] = None) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads training and test data for a specific fold, with optional data reduction.

    Args:
        fold_number: Index of the fold to load (0-9)
        dataset_path: Root directory containing the dataset files
        reduction_method: Optional data reduction technique ('EENTH', 'GCNN', or 'DROP3')

    Returns:
        Tuple containing:
        - train_features: DataFrame of training features
        - train_labels: Series of training labels
        - test_features: DataFrame of test features
        - test_labels: Series of test labels
    """
    if reduction_method:
        # Load reduced training data from the ReducedFolds directory
        train_path = os.path.join(dataset_path, 'ReducedFolds')
        train_file = os.path.join(train_path, f'mushroom.fold.{fold_number:06d}.train.{reduction_method}.csv')
    else:
        # Load original training data from preprocessed_csvs directory
        train_path = os.path.join(dataset_path, 'preprocessed_csvs')
        train_file = os.path.join(train_path, f'mushroom.fold.{fold_number:06d}.train.csv')

    # Load and clean training data
    train_data = pd.read_csv(train_file)
    train_data = train_data.drop('Unnamed: 0', axis=1)
    if reduction_method: train_data = train_data.drop('Unnamed: 0.1', axis=1)

    # Load and clean test data (always from original dataset)
    test_file = os.path.join(dataset_path, 'preprocessed_csvs', f'mushroom.fold.{fold_number:06d}.test.csv')
    test_data = pd.read_csv(test_file)
    test_data = test_data.drop('Unnamed: 0', axis=1)

    # Split features and labels
    train_features = train_data.drop('class', axis=1)
    train_labels = train_data['class']
    test_features = test_data.drop('class', axis=1)
    test_labels = test_data['class']

    return train_features, train_labels, test_features, test_labels


def previous_analysis(dataset_path_f):
    """
    Performs initial analysis to evaluate different SVM configurations.

    Tests all combinations of:
    - C values: [0.1, 1, 10, 100]
    - Kernels: ['linear', 'rbf', 'poly', 'sigmoid']

    For each configuration:
    - Trains and evaluates the model on all 10 folds
    - Records accuracy, training time, and F1 score

    Args:
        dataset_path_f: Path to the dataset directory

    Returns:
        Tuple containing:
        - results: 5x5 array with average accuracies for each configuration
        - metrics: DataFrame with detailed performance metrics for each fold
    """
    c_values = [0.1, 1, 10, 100]
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    results = np.zeros((5, 5), dtype=object)
    results[0, 1:] = c_values
    results[1:, 0] = kernels
    metrics = []

    for j in range(4):
        for i in range(4):
            prev_accuracy = np.zeros(10)
            for n in range(10):
                x_train, y_train, x_test, y_test = load_fold_data(n, dataset_path_f)
                svm_classifier = SVM(train_data=x_train, train_labels=y_train, kernel=kernels[i], C=c_values[j],
                                     gamma='auto')
                svm_classifier.train()
                evaluation = svm_classifier.evaluate(x_test, y_test)
                prev_accuracy[n] = evaluation[0]

                model = f"SVM, kernel={kernels[i]}, C={c_values[j]:.1f}"
                metrics.append({
                    'Model': model,
                    'Dataset/Fold': f"Mushroom/{n}",
                    'Accuracy': evaluation[0],
                    'Time': evaluation[1],
                    'F1': evaluation[2]
                })
                #print(f'Model {model}, for mushroom/{n} trained and saved.')
            results[i + 1, j + 1] = np.mean(prev_accuracy)

    print('Previous analysis done in order to find the best parameters for the dataset.')
    return results, pd.DataFrame(metrics)


def find_top_ten(results_f):
    """
    Identifies the top 10 performing SVM configurations based on average accuracy.

    Args:
        results_f: 10x10 array containing average accuracies for each configuration

    Returns:
        Tuple containing:
        - kernel_tags: Array of the top 10 kernel types
        - c_value_tags: Array of the corresponding C values
    """
    kernel_tags = np.zeros(10, dtype='object')
    c_value_tags = np.zeros(10, dtype='float')
    data_region = results_f[1:, 1:].copy()

    print(f'Optimal parameters are: ')

    for n in range(10):
        max_index = np.argmax(data_region)
        max_coords = np.unravel_index(max_index, data_region.shape)
        kernel_tags[n] = results_f[max_coords[0] + 1, 0]
        c_value_tags[n] = results_f[0, max_coords[1] + 1]
        data_region[max_coords] = np.min(data_region) - 1
        print(f'{kernel_tags[n]} and {c_value_tags[n]}')

    return kernel_tags, c_value_tags


def filter_top_models(prev_results_dataframe, kernel_def_fff, c_value_def_fff):
    """
    Extracts performance metrics for the top 10 SVM configurations.

    Args:
        prev_results_dataframe: DataFrame containing all performance metrics
        kernel_def_fff: Array of top 10 kernel types
        c_value_def_fff: Array of top 10 C values

    Returns:
        DataFrame containing metrics only for the top 10 configurations
    """
    filtered_rows = []
    for i in range(10):
        model_pattern = f"SVM, kernel={kernel_def_fff[i]}, C={c_value_def_fff[i]:.1f}"
        matching_rows = prev_results_dataframe[prev_results_dataframe['Model'] == model_pattern]
        filtered_rows.append(matching_rows)
    filtered_dataframe = pd.concat(filtered_rows, ignore_index=True)
    return filtered_dataframe


def filter_top_model(prev_results_dataframe, kernel_def_fff, c_value_def_fff):
    """
    Extracts performance metrics for a single SVM configuration.

    Args:
        prev_results_dataframe: DataFrame containing all performance metrics
        kernel_def_fff: Kernel type
        c_value_def_fff: C value

    Returns:
        DataFrame containing metrics only for the specified configuration
    """
    model_pattern = f"SVM, kernel={kernel_def_fff}, C={c_value_def_fff:.1f}"
    matching_rows = prev_results_dataframe[prev_results_dataframe['Model'] == model_pattern]
    return matching_rows


def total_analysis(kernel_def_f, c_value_def_f, dataset_path_ff):
    """
    Evaluates the best performing SVM configuration using different data reduction methods.

    Tests the model with:
    - Original dataset (no reduction)
    - EENTH reduction
    - GCNN reduction
    - DROP3 reduction

    Args:
        kernel_def_f: Selected kernel type
        c_value_def_f: Selected C value
        dataset_path_ff: Path to dataset directory

    Returns:
        DataFrame containing performance metrics for all combinations of folds and reduction methods
    """
    c_value_def_f = float(c_value_def_f)
    reduction_methods_f = ['EENTH', 'GCNN', 'DROP3']
    metrics = []

    #print(f"Testing configurations across 10 folds with different reduction methods.")
    for reduction_method in reduction_methods_f:
        reduction_desc = reduction_method if reduction_method else "None"

        for fold in range(10):
            model = f"SVM, kernel={kernel_def_f}, C={c_value_def_f}"
            x_train, y_train, x_test, y_test = load_fold_data(fold, dataset_path_ff, reduction_method)
            svm_classifier = SVM(train_data=x_train, train_labels=y_train, kernel=kernel_def_f, C=c_value_def_f,
                                 gamma='auto')
            svm_classifier.train()
            evaluation = svm_classifier.evaluate(x_test, y_test)

            metrics.append({
                'Model': model + ', ' + reduction_desc,
                'Dataset/Fold': f"Mushroom/{fold}",
                'Accuracy': evaluation[0],
                'Time': evaluation[1],
                'F1': evaluation[2]
            })
            #print(f'Model {model}, for mushroom/{fold} and {reduction_desc} trained and saved.')

    return pd.DataFrame(metrics)

if __name__ == "__main__":
    dataset_path = '..\\Mushroom'
else:
    dataset_path = 'Mushroom'
# Main execution flow

# 1. Initial analysis of all SVM configurations
prev_results = previous_analysis(dataset_path)
txt_path = os.path.join(dataset_path, "pre_analysis.txt")
np.savetxt(txt_path, prev_results[0], fmt="%s", delimiter=" , ")

# 2. Extract top 10 performing configurations
kernel_def, c_value_def = find_top_ten(prev_results[0])
best_ten_algo = filter_top_models(prev_results[1], kernel_def, c_value_def)
csv_path = os.path.join(dataset_path, "svm_mushroom_results_best10.csv")
best_ten_algo.to_csv(csv_path, index=False)

# 3. Statistical analysis using Friedman-Nemenyi test
csv_path = os.path.join(dataset_path, "svm_mushroom_results_best10.csv")
output_path = os.path.join(dataset_path, "plots_and_tables\\svm_base\\statistical_analysis_results.png")
report_output_path = os.path.join(dataset_path, "plots_and_tables\\svm_base\\statistical_analysis_results.txt")
alpha = 0.1  # Significance level for statistical tests

if not analyze_model_performance(csv_path, output_path, report_output_path, alpha):
    print("It is concluded that there is no statistical difference between models.")
else:
    print("It is concluded that there is statistical difference between models.")
    main(csv_path,output_path,report_output_path,alpha)

# 4. Select and analyze best performing configuration
best_SVM_algo = filter_top_model(prev_results[1], kernel_def[0], c_value_def[0])
csv_path_2 = os.path.join(dataset_path, "svm_mushroom_results_best1.csv")
best_SVM_algo.to_csv(csv_path_2, index=False)

# Add "NONE" label to indicate no reduction method used
df = best_SVM_algo.copy()
df['Model'] = df['Model'] + ", NONE"
print(c_value_def[0])

# 5. Evaluate best configuration with different reduction methods
best_algo_reduced = total_analysis(kernel_def[0], float(c_value_def[0]), dataset_path)
best_algo_reduced_and_non_red = pd.concat([df, best_algo_reduced], ignore_index=True)
csv_path_3 = os.path.join(dataset_path, "svm_mushroom_results_reduced.csv")
best_algo_reduced_and_non_red.to_csv(csv_path_3, index=False)

# Note: Further analysis of reduction methods can be performed by running svm_reduction_analysis.py