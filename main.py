import argparse
import os
import subprocess
import sys

# Define valid algorithms and datasets
VALID_ALGORITHMS = ["kmeans", "spectral_clustering", "fuzzy", "xmeans", "global_kmeans", "optics"]
VALID_DATASETS = ["Hepatitis", "Mushroom", "Pen-based"]  # Add all supported datasets here

def is_valid_algorithm(algorithm):
    return algorithm.lower() in VALID_ALGORITHMS

def is_valid_dataset(dataset):
    return dataset in VALID_DATASETS and os.path.isdir(dataset)

def run_preprocessing(dataset):
    """Run preprocessing for the given dataset."""
    preprocessing_script = os.path.join(dataset, "Preprocessing", "preprocessing.py")
    if not os.path.isfile(preprocessing_script):
        print(f"Error: Preprocessing script not found in {dataset}.")
        sys.exit(1)
    print(f"Running preprocessing for {dataset}...")
    try:
        subprocess.run([sys.executable, os.path.abspath(preprocessing_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)

def run_algorithm(dataset, algorithm):
    """Run the selected algorithm analysis and result script."""
    analysis_script = os.path.join(dataset, "Analysis", "plots_and_tables", f"{algorithm.lower()}_analysis.py")
    results_script = os.path.join(dataset, "Results", f"{algorithm.lower()}_results.py")

    for script in [results_script, analysis_script]:
        if os.path.isfile(script):
            print(f"Running {os.path.basename(script)}...")
            try:
                subprocess.run([sys.executable, os.path.abspath(script)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while running {script}: {e}")
                sys.exit(1)
        else:
            print(f"Error: Script {script} not found.")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Execute algorithms on datasets with optional preprocessing.")
    parser.add_argument("--algorithm", required=True,
                        help=f"Algorithm to run. Supported: {', '.join(VALID_ALGORITHMS)}")
    parser.add_argument("--dataset", required=True,
                        help=f"Dataset folder to use. Supported: {', '.join(VALID_DATASETS)}")
    parser.add_argument("--run-preprocessing", action="store_true",
                        help="Flag to run preprocessing step before analysis. Including PCA and UMAP for visualisation.")

    args = parser.parse_args()

    # Validate algorithm
    if not is_valid_algorithm(args.algorithm):
        print(f"Error: '{args.algorithm}' is not a valid algorithm. Supported: {', '.join(VALID_ALGORITHMS)}")
        sys.exit(1)

    # Validate dataset
    if not is_valid_dataset(args.dataset):
        print(f"Error: '{args.dataset}' is not a valid dataset. Supported: {', '.join(VALID_DATASETS)}")
        sys.exit(1)

    dataset_folder = os.path.abspath(args.dataset)

    # Run preprocessing if flagged
    if args.run_preprocessing:
        run_preprocessing(dataset_folder)

    # Run the chosen algorithm
    run_algorithm(dataset_folder, args.algorithm)

if __name__ == "__main__":
    main()
