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

            # Filter out rows where labels == -1
            # # Create a mask to exclude rows where labels == -1
            # mask = labels != -1
            # filtered_labels = labels[mask]
            # filtered_X = X[mask]
            # filtered_classes = classes[mask]
            #
            # # Reset index for clean output (optional)
            # filtered_X = filtered_X.reset_index(drop=True)
            # filtered_classes = filtered_classes.reset_index(drop=True)

            # Stop timing
            end_time = time.time()
            execution_time = end_time - start_time

            # Evaluate clustering performance
            metrics = EvaluationUtils.evaluate(X, class_labels, cluster_labels)

            # Add predicted number of clusters (for XMeans)
            metrics['Predicted k'] = len(np.unique(cluster_labels))

            # Add execution time to metrics
            metrics['Time'] = execution_time

            return metrics, cluster_labels
        except Exception as e:
            print(f"Error during evaluation of {algorithm_name}: {e}")
            return None, None

    @staticmethod
    def run_single_config(config, model_class, X, class_labels):
        """
        Run a single configuration and return the results.

        Parameters:
        - config: dict, single configuration from the grid
        - model_class: class, clustering model class
        - X: numpy.ndarray, feature matrix
        - class_labels: pandas.Series, true class labels

        Returns:
        - dict with results and labels
        """
        algorithm_name = f"{model_class.__name__}({', '.join(f'{k}={v}' for k, v in config.items() if k != 'Repetition')})"
        try:
            model = model_class(**{k: v for k, v in config.items() if k != 'Repetition'})
            metrics, cluster_labels = ResultUtils.getResults(algorithm_name, model, X, class_labels)
            if metrics is not None:
                return {
                    "success": True,
                    "results": {
                        'Algorithm': algorithm_name,
                        **config,
                        **metrics
                    },
                    "labels": pd.DataFrame({algorithm_name: cluster_labels})
                }
        except Exception as e:
            print(f"Error with configuration {config}: {e}")

        return {"success": False}

    @staticmethod
    def runGrid(grid, model_class, X, class_labels, results_file, labels_file):
        """
        Run a grid of configurations in parallel and save results and labels.
        """
        flattened_configs = ResultUtils.flatten_grid(grid)
        total_configs = len(flattened_configs)

        results = []
        labels_df = pd.DataFrame()

        with ProcessPoolExecutor() as executor:
            future_to_config = {
                executor.submit(
                    ResultUtils.run_single_config, config, model_class, X, class_labels
                ): config for config in flattened_configs
            }

            for i, future in enumerate(as_completed(future_to_config), start=1):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result["success"]:
                        results.append(result["results"])
                        labels_df = pd.concat([labels_df, result["labels"]], axis=1)
                except Exception as e:
                    print(f"Error processing configuration {config}: {e}")

                ResultUtils.progress_bar(i, total_configs)

        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        pd.DataFrame(results).to_csv(results_file, index=False)
        labels_df.to_csv(labels_file, index=False)

        print(f"Results saved to {results_file}")
        print(f"Cluster labels saved to {labels_file}")
