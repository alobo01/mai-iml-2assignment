import os
import sys

from Classes.Reader import DataPreprocessor

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

binary_features = [
    "SEX",
    "STEROID",
    "ANTIVIRALS",
    "FATIGUE",
    "MALAISE",
    "ANOREXIA",
    "LIVER_BIG",
    "LIVER_FIRM",
    "SPLEEN_PALPABLE",
    "SPIDERS",
    "ASCITES",
    "VARICES",
    "HISTOLOGY"
]

arff_path = os.path.join(dataset_path, "../Datasets/hepatitis.arff")
complete_df = DataPreprocessor.load_arff(arff_path)

# Initialize and fit the preprocessor on the training data and transform
reader = DataPreprocessor(complete_df, class_column="Class")
preprocessed_df = reader.fit_transform(ordinal_features=binary_features)

preprocessed_path = os.path.join(dataset_path, "Preprocessing/hepatitis.csv")
preprocessed_df.to_csv(preprocessed_path)

pca = DataPreprocessor.convert_dataframe_to_principal_components(preprocessed_df)
pca.to_csv(os.path.join(dataset_path, "Preprocessing/hepatitis_pca.csv"))

umap = DataPreprocessor.convert_dataframe_to_UMAP(preprocessed_df)
umap.to_csv(os.path.join(dataset_path, "Preprocessing/hepatitis_umap.csv"))
