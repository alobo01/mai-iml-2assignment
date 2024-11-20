import pandas as pd
import numpy as np
from classes.KNN import KNNAlgorithm, apply_weighting_method
import itertools
from typing import Dict, List, Tuple
import time
from sklearn.metrics import f1_score
from tqdm import tqdm
import os


def load_fold_data(fold_number: int, dataset_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load fold data without reduction

    Args:
        fold_number: The fold number to load
        dataset_path: Base path to the dataset

    Returns:
        Tuple of (train_features, train_labels, test_features, test_labels)
    """
    train_path = os.path.join(dataset_path, 'preprocessed_csvs')
    train_file = os.path.join(train_path, f'mushroom.fold.{fold_number:06d}.train.csv')

    train_data = pd.read_csv(train_file)
    train_data = train_data.drop('Unnamed: 0', axis=1)

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


def run_knn_experiments(dataset_path: str):
    """
    Run experiments for all KNN configurations without reduction techniques
    """
    config_space = {
        'k': [1, 3, 5, 7],
        'distance_metric': ['euclidean_distance', 'manhattan_distance', 'clark_distance'],
        'weighting_method': ['equal_weight', 'information_gain_weight', 'reliefF_weight'],
        'voting_policy': ['majority_class', 'inverse_distance_weighted', 'shepard']
    }

    results = []

    print("Testing KNN configurations across 10 folds...")
    for fold in tqdm(range(10), desc="Processing folds"):
        train_features, train_labels, test_features, test_labels = load_fold_data(fold, dataset_path)

        for weighting_method in config_space['weighting_method']:
            for k in config_space['k']:
                weighted_train, weighted_test = get_weighted_features(
                    train_features, train_labels, test_features, weighting_method, k
                )

                for distance_metric, voting_policy in itertools.product(
                        config_space['distance_metric'],
                        config_space['voting_policy']
                ):
                    config = {
                        'k': k,
                        'distance_metric': distance_metric,
                        'weighting_method': weighting_method,
                        'voting_policy': voting_policy
                    }

                    accuracy, train_time, f1 = evaluate_knn_configuration(
                        weighted_train, train_labels,
                        weighted_test, test_labels,
                        config
                    )

                    model_name = f"KNN, {k}, {distance_metric}, {weighting_method}, {voting_policy}"

                    results.append({
                        'Model': model_name,
                        'Dataset/Fold': f"Mushroom/{fold}",
                        'Accuracy': accuracy,
                        'Time': train_time,
                        'F1': f1
                    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    dataset_path = '..\\Mushroom'
else:
    dataset_path = 'Mushroom'
results = run_knn_experiments(dataset_path)
results.to_csv('knn_base_results.csv', index=False)