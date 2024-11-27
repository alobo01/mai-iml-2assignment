from Classes.Reader import DataPreprocessor
import os

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Hepatitis'

features = [
    "AGE",
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
    "BILIRUBIN",
    "ALK_PHOSPHATE",
    "SGOT",
    "ALBUMIN",
    "PROTIME",
    "HISTOLOGY"
]


arff_path = os.path.join(dataset_path, "../Datasets/hepatitis.arff")
complete_df = DataPreprocessor.load_arff(arff_path)
complete_df.to_csv('prueba.csv')

# Initialize and fit the preprocessor on the training data and transform
reader = DataPreprocessor(complete_df, class_column="Class")
preprocessed_df = reader.fit_transform(ordinal_features=features)

preprocessed_path = os.path.join(dataset_path, "Preprocessing/hepatitis.csv")
preprocessed_df_removed_columns = (preprocessed_df.to_csv(preprocessed_path))
