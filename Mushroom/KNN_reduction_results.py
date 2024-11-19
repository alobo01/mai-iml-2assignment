import pandas as pd
import numpy as np
from classes.KNN import KNNAlgorithm, apply_weighting_method
from typing import Dict, List, Tuple, Optional
import time
from sklearn.metrics import f1_score
from tqdm import tqdm
import os


def load_fold_data(fold_number: int, dataset_path: str, reduction_method: Optional[str] = None) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load fold data with optional reduction method

    Args:
        fold_number: The fold number to load
        dataset_path: Base path to the dataset
        reduction_method: Optional reduction method ('EENTH', 'GCNN' or 'DROP3')

    Returns:
        Tuple of (train_features, train_labels, test_features, test_labels)
    """
    if reduction_method:
        train_path = os.path.join(dataset_path, 'ReducedFolds')
        train_file = os.path.join(train_path, f'mushroom.fold.{fold_number:06d}.train.{reduction_method}.csv')
    else:
        train_path = os.path.join(dataset_path, 'preprocessed_csvs')
        train_file = os.path.join(train_path, f'mushroom.fold.{fold_number:06d}.train.csv')

    train_data = pd.read_csv(train_file)
    train_data = train_data.drop('Unnamed: 0', axis=1)
    if reduction_method: train_data = train_data.drop('Unnamed: 0.1', axis=1)

    test_file = os.path.join(dataset_path, 'preprocessed_csvs', f'mushroom.fold.{fold_number:06d}.test.csv')
    test_data = pd.read_csv(test_file)
    test_data = test_data.drop('Unnamed: 0', axis=1)

    train_features = train_data.drop('class', axis=1)
    train_labels = train_data['class']
    test_features = test_data.drop('class', axis=1)
    test_labels = test_data['class']

    return train_features, train_labels, test_features, test_labels


def evaluate_knn_configuration(
        weighted_train_features: pd.DataFrame,
        train_labels: pd.Series,
        weighted_test_features: pd.DataFrame,
        test_labels: pd.Series,
        config: Dict
) -> Tuple[float, float, float]:
    """
    Evaluate a single KNN configuration and return accuracy, training/evaluation time, and F1 score
    """
    knn = KNNAlgorithm(
        k=config['k'],
        distance_metric=config['distance_metric'],
        weighting_method='equal_weight',
        voting_policy=config['voting_policy']
    )

    start_time = time.time()
    knn.fit(weighted_train_features, train_labels)
    predictions = knn.predict(weighted_test_features)
    total_time = time.time() - start_time

    accuracy = knn.score(weighted_test_features, test_labels)
    f1 = f1_score(test_labels, predictions, average='weighted')

    return accuracy, total_time, f1


def get_weighted_features(train_features: pd.DataFrame,
                          train_labels: pd.Series,
                          test_features: pd.DataFrame,
                          weighting_method: str,
                          k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply feature weighting to both train and test features
    """
    weighted_train = apply_weighting_method(train_features, train_labels, weighting_method, k)
    return weighted_train, test_features


def run_reduction_experiments(dataset_path: str, knn_config: Dict):
    """
    Run experiments for a specific KNN configuration with different reduction techniques

    Args:
        dataset_path: Path to the dataset
        knn_config: Dictionary containing KNN configuration (k, distance_metric, weighting_method, voting_policy)
    """
    reduction_methods = [None, 'EENTH', 'GCNN', 'DROP3']
    results = []
    sample_counts = []

    print(f"Testing reduction methods across 10 folds...")
    for reduction_method in reduction_methods:
        reduction_desc = reduction_method if reduction_method else "None"

        for fold in tqdm(range(10), desc=f"Processing fold ({reduction_desc})"):
            train_features, train_labels, test_features, test_labels = load_fold_data(
                fold, dataset_path, reduction_method
            )

            weighted_train, weighted_test = get_weighted_features(
                train_features, train_labels, test_features,
                knn_config['weighting_method'], knn_config['k']
            )

            accuracy, train_time, f1 = evaluate_knn_configuration(
                weighted_train, train_labels,
                weighted_test, test_labels,
                knn_config
            )

            reduction_suffix = f"{reduction_method}" if reduction_method else "NONE"
            model_name = (f"KNN, {knn_config['k']}, {knn_config['distance_metric']}, "
                          f"{knn_config['weighting_method']}, {knn_config['voting_policy']}, {reduction_suffix}")

            results.append({
                'Model': model_name,
                'Dataset/Fold': f"Mushroom/{fold}",
                'Accuracy': accuracy,
                'Time': train_time,
                'F1': f1
            })

            sample_counts.append({
                'Fold': fold,
                'Reduction Method': reduction_suffix,
                'Training Samples': len(train_features)
            })

    results_df = pd.DataFrame(results)
    sample_counts_df = pd.DataFrame(sample_counts)
    return results_df, sample_counts_df


if __name__ == "__main__":
    dataset_path = '..\\Mushroom'
else:
    dataset_path = 'Mushroom'

# Define a specific KNN configuration to test with reduction methods
knn_config = {
    'k': 7,
    'distance_metric': 'manhattan_distance',
    'weighting_method': 'equal_weight',
    'voting_policy': 'majority_class'
}

results, counts = run_reduction_experiments(dataset_path, knn_config)
results.to_csv('knn_reduction_results.csv', index=False)
counts.to_csv('knn_reduction_counts.csv', index=False)