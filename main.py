import argparse
import os
import sys
from pathlib import Path
import importlib.util

REQUIREMENTS = [
    'scipy', 'pandas', 'scikit-learn', 'numpy', 'joblib', 'matplotlib',
    'seaborn', 'scikit_posthocs', 'pyamg', 'umap-learn', 'plotly'
]


def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        'scipy': 'scipy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'numpy': 'numpy',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scikit_posthocs': 'scikit_posthocs',
        'pyamg': 'pyamg',
        'umap-learn': 'umap',
        'plotly': 'plotly'
    }

    missing_packages = []
    for pip_name, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print("Error: Missing required packages:", ', '.join(missing_packages))
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements using:")
        print("pip install -r requirements.txt")
        sys.exit(1)


def import_script(script_path):
    """Dynamically import a Python script from a given path."""
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def run_preprocessing(dataset_path):
    """Run preprocessing for the dataset."""
    print("\nRunning preprocessing...")
    preprocessing_script = Path(os.path.join(dataset_path, "Preprocessing", "preprocessing.py"))

    if preprocessing_script.exists():
        import_script(preprocessing_script)
    else:
        raise FileNotFoundError(f"Preprocessing script not found: {preprocessing_script}")


def run_algorithm(dataset_path, algorithm):
    """Run the selected algorithm's analysis and result scripts."""
    print(f"\nRunning {algorithm} algorithm analysis...")

    # Cross-platform paths using Path and os.path.join
    algorithm_script = Path(os.path.join(dataset_path, "Analysis", f"{algorithm.lower()}_analysis.py"))
    results_script = Path(os.path.join(dataset_path, "Results", f"{algorithm.lower()}_results.py"))

    # Run results script first
    if results_script.exists():
        import_script(results_script)
    else:
        raise FileNotFoundError(f"Results script not found: {results_script}")

    # Run algorithm analysis script
    if algorithm_script.exists():
        import_script(algorithm_script)
    else:
        raise FileNotFoundError(f"Analysis script not found: {algorithm_script}")


def main():
    check_dependencies()

    parser = argparse.ArgumentParser(description="Run clustering algorithms on datasets with preprocessing.")
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['kmeans', 'spectral_clustering', 'fuzzy', 'xmeans', 'global_kmeans', 'optics'],
                        help="Algorithm to run.")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['Hepatitis', 'Mushroom', 'Pen-based'],
                        help="Dataset to use.")
    parser.add_argument('--run-preprocessing', action='store_true',
                        help="Flag to run preprocessing before analysis.")

    args = parser.parse_args()

    # Resolve dataset path
    project_root = Path(__file__).parent
    dataset_path = project_root / args.dataset

    if not dataset_path.exists():
        print(f"Error: Dataset folder '{dataset_path}' does not exist.")
        sys.exit(1)

    # Run preprocessing if flagged
    if args.run_preprocessing:
        run_preprocessing(dataset_path)

    # Run the selected algorithm
    try:
        run_algorithm(dataset_path, args.algorithm)
        print("\nExecution completed successfully!")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
