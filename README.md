Here’s a complete `setup_and_run.md` file to guide users on creating the environment, installing dependencies, and running the `main.py` script.

---

# **Setup and Execution Guide**

This document provides step-by-step instructions to set up the environment, install the required dependencies, and execute the `main.py` script for running algorithms on datasets with optional preprocessing.

**Note**: You may need to use `python3` instead of  `python` depending on your python installation.

---

## **1. Prerequisites**

Ensure the following software is installed on your system:

- **Python** (Version 3.8 or higher)
- **pip** (Python package manager)

Check the versions:
```bash
python --version
pip --version
```

---

## **2. Environment Setup**

1. **Create a Virtual Environment**

Run the following command in your project root directory:

- **For Windows**:
   ```bash
   python -m venv env
   .\env\Scripts\activate
   ```

- **For macOS/Linux**:
   ```bash
   python -m venv env
   source env/bin/activate
   ```

> **Note**: You should see `(env)` at the start of your terminal prompt, indicating the environment is active.

---

## **3. Install Requirements**

All dependencies are listed in the `requirements.txt` file. Install them with:

```bash
pip install -r requirements.txt
```

---

## **4. Directory Structure**

```plaintext
IML3/
│
├── main.py
├── Classes/
│   ├── AnalysisUtils.py
│   ├── EvaluationUtils.py
│   ├── FuzzyClustering.py
│   ├── GlobalKMeans.py
│   ├── KMeans.py
│   ├── OpticsClustering.py
│   ├── Reader.py
│   ├── ResultUtils.py
│   ├── SpectralClustering.py
│   ├── ViolinPlotsUtils.py
│   └── XMeans.py
│
├── Datasets/
│   ├── hepatitis.arff
│   ├── mushroom.arff
│   └── pen-based.arff
│
├── Hepatitis/
│   ├── Preprocessing/
│   │   └── preprocessing.py
│   ├── Analysis/
│   │   └── plots_and_tables/
│   │       ├── kmeans_analysis.py
│   │       ├── spectral_clustering_analysis.py
│   │       ├── ...
│   │       
│   └── Results/
│       ├── kmeans_results.py
│       ├── spectral_clustering_results.py
│       ├── ...
│
├── Mushroom/     # Other datasets follow the same structure
├── Pen-Based/     # Other datasets follow the same structure
└── requirements.txt
```

---

## **5. Running the Script**

### **General Syntax**

```bash
python main.py --algorithm <ALGORITHM> --dataset <DATASET> [--run-preprocessing]
```

| Flag                 | Description                                              |
|-----------------------|----------------------------------------------------------|
| `--algorithm`         | Algorithm to run (e.g., KMeans, SpectralClustering).     |
| `--dataset`           | Dataset folder to use (e.g., Hepatitis, Mushroom).       |
| `--run-preprocessing` | Optional flag to preprocess the dataset (PCA, UMAP).     |

---

### **Examples**

1. **Run `KMeans` on the `Hepatitis` dataset with preprocessing**:
   ```bash
   python main.py --algorithm KMeans --dataset Hepatitis --run-preprocessing
   ```

2. **Run `SpectralClustering` on the `Mushroom` dataset without preprocessing**:
   ```bash
   python main.py --algorithm SpectralClustering --dataset Mushroom
   ```

3. **Handle invalid inputs**:
   If an unsupported algorithm or dataset is provided, the script will display an error:
   ```
   Error: 'InvalidAlgorithm' is not a valid algorithm.
   ```

---

## **6. Valid Algorithms and Datasets**

### **Supported Algorithms**
- `KMeans`
- `SpectralClustering`
- `Fuzzy`
- `XMeans`
- `GlobalKMeans`
- `Optics`

### **Supported Datasets**
- `Hepatitis`
- `Mushroom`
- `Pen-based`

---

## **7. Automating All Runs**

To execute all algorithms on all datasets, create and run the `run_all.py` script:

```python
import subprocess
import sys

datasets = ["Hepatitis", "Mushroom"]
algorithms = ["KMeans", "SpectralClustering", "Fuzzy", "XMeans", "GlobalKMeans", "Optics"]

for dataset in datasets:
    for algorithm in algorithms:
        print(f"Running {algorithm} on {dataset}...")
        subprocess.run([sys.executable, "main.py", 
                        "--algorithm", algorithm, 
                        "--dataset", dataset, 
                        "--run-preprocessing"], check=True)
```

Run the automation:
```bash
python run_all.py
```

---

## **8. Deactivate Environment**

When you're done, deactivate the virtual environment:

```bash
deactivate
```

---

## **9. Troubleshooting**

1. **Virtual Environment Issues**: Ensure you’ve activated the environment (`env`) before running any script.

2. **Missing Dependencies**: Reinstall them using:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

3. **Incorrect Directory Structure**: Ensure all scripts and folders match the expected structure.

4. **Permission Issues**: Grant executable permissions on Linux/macOS:
   ```bash
   chmod +x main.py
   ```

