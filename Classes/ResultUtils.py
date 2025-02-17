import itertools
import os
import sys
import time
from typing import Iterable
import numpy as np

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from Classes.EvaluationUtils import EvaluationUtils


class ResultUtils:

    @staticmethod
    def flatten_grid(grid):
        """
        Flatten a nested grid to handle configurations with two levels of nesting,
        and repeat each configuration based on the 'Repetitions' parameter.

        Parameters:
        - grid: dict, configuration grid with optional nested dictionaries

        Returns:
        - list of dicts, flattened configurations with Repetitions applied
        """
        base_keys = [k for k in grid if not isinstance(grid[k], dict) and k != 'Repetitions']
        nested_keys = [k for k in grid if isinstance(grid[k], dict) and k != 'Repetitions']

        base_combinations = [dict(zip(base_keys, values)) for values in
                             itertools.product(*[grid[k] for k in base_keys])]

        final_combinations = []

        for base_combination in base_combinations:
            if nested_keys:
                for nested_key in nested_keys:
                    for sub_key, sub_values in grid[nested_key].items():
                        for sub_param, sub_value in sub_values.items():
                            if isinstance(sub_values, Iterable):
                                for sub_value_value in sub_value:
                                    combined = base_combination.copy()
                                    combined[nested_key] = sub_key
                                    combined[sub_param] = sub_value_value
                                    final_combinations.append(combined)
                            else:
                                combined = base_combination.copy()
                                combined[nested_key] = sub_key
                                combined[sub_param] = sub_value
                                final_combinations.append(combined)
            else:
                # No nested keys; use base combinations directly.
                final_combinations.append(base_combination)

        # Handle Repetitions if present
        if 'Repetitions' in grid:
            repetitions = grid['Repetitions']
            expanded_combinations = []
            for comb in final_combinations:
                for repetition_index in range(repetitions):
                    expanded_comb = comb.copy()
                    expanded_comb['Repetition'] = repetition_index  # Track repetition index
                    expanded_combinations.append(expanded_comb)
            return expanded_combinations

        return final_combinations

    @staticmethod
    def progress_bar(current, total, bar_length=50):
        """
        Displays a progress bar.

        :param current: Current iteration index.
        :param total: Total number of iterations.
        :param bar_length: Length of the progress bar.
        """
        fraction = current / total
        arrow = "=" * int(fraction * bar_length - 1) + ">" if fraction < 1 else "=" * bar_length
        padding = " " * (bar_length - len(arrow))
        end = "\r" if current < total else "\n"
        sys.stdout.write(f"\r[{arrow}{padding}] {fraction:.0%}")
        sys.stdout.flush()

    @staticmethod
    def getResults(algorithm_name, model, X, class_labels):
        """
        Run clustering, evaluate results, and return performance metrics and labels.
        """
        try:
            # Start timing
            start_time = time.time()

            # Fit the model and predict labels
            cluster_labels = model.fit(X)
            # Stop timing
            end_time = time.time()
            execution_time = end_time - start_time

            # Filter out the noisy points (labels == -1)

            mask = cluster_labels != -1 # Create a mask to exclude rows where labels == -1
            filtered_labels = cluster_labels[mask]
            filtered_x = X[mask]
            filtered_classes = class_labels[mask].to_numpy()

            # Evaluate clustering performance
            metrics = EvaluationUtils.evaluate(filtered_x, filtered_classes, filtered_labels)

            # Add predicted number of clusters (for XMeans)
            metrics['Predicted k'] = len(np.unique(cluster_labels))

            # Add execution time to metrics
            metrics['Time'] = execution_time
            return metrics, cluster_labels

        except Exception as e:
            print(f"Error during evaluation of {algorithm_name}: {e}")
            return None, None

    @staticmethod
    def runGrid(grid, model_class, X, class_labels, results_file, labels_file):
        """
        Run a grid of configurations and save results and labels.

        Parameters:
        - grid: dict, configuration grid (supports nested grids)
        - model_class: class, clustering model class (must have fit and predict methods)
        - X: numpy.ndarray, feature matrix
        - class_labels: pandas.Series, true class labels
        - results_file: str, path to save the results CSV file
        - labels_file: str, path to save the labels CSV file
        """
        results = []
        labels_df = pd.DataFrame()

        # Flatten the grid to handle nested configurations
        flattened_configs = ResultUtils.flatten_grid(grid)
        total_configs = len(flattened_configs)
        for index,config in enumerate(flattened_configs,start=1):
            algorithm_name = f"{model_class.__name__}({', '.join(f'{k}={v}' for k, v in config.items() if k!='Repetition')})"
            try:
                # Instantiate the model with the flattened configuration
                model = model_class(**{k: v for k, v in config.items() if k != 'Repetition'})

                # Handle multiple runs
                repetition = config['Repetition']
                full_algorithm_name = f"{algorithm_name}_{repetition}"

                # Get results using getResults
                metrics, cluster_labels = ResultUtils.getResults(full_algorithm_name, model, X, class_labels)
                if metrics is not None:
                    # Append results
                    results.append({
                        'Algorithm': full_algorithm_name,
                        **config,
                        **metrics
                    })

                    # Add labels to DataFrame
                    labels_df = pd.concat([labels_df, pd.DataFrame({full_algorithm_name: cluster_labels})], axis=1)
                ResultUtils.progress_bar(index,total_configs)

            except Exception as e:
                print(f"Error with configuration {config}: {e}")

        # Save results and labels
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        pd.DataFrame(results).to_csv(results_file, index=False)
        labels_df.to_csv(labels_file, index=False)

        print(f"\nResults saved to {results_file}")
        print(f"Cluster labels saved to {labels_file}")
