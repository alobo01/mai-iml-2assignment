from Classes.Reader import DataPreprocessor

if __name__ == "__main__":
    dataset_path = '..\\Hepatitis'
else:
    dataset_path = 'Hepatitis'

binary_features  = [
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

complete_df = DataPreprocessor.load_arff("../Datasets/hepatitis.arff")

# Initialize and fit the preprocessor on the training data and transform
reader = DataPreprocessor(complete_df, class_column="Class")
preprocessed_df = reader.fit_transform(ordinal_features=binary_features)
removed_features = DataPreprocessor.get_columns_with_missing_values_over_threshold(complete_df)
preprocessed_df_removed_columns = (preprocessed_df.drop(columns=removed_features)
                                   .to_csv("../Preprocessed_datasets/hepatitis.csv"))
