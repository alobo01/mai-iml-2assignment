from Classes.Reader import DataPreprocessor
import os

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Mushroom'

categorical_features = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises?",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

arff_path = os.path.join(dataset_path, "../Datasets/mushroom.arff")
complete_df = DataPreprocessor.load_arff(arff_path)

# Initialize and fit the preprocessor on the training data and transform
reader = DataPreprocessor(complete_df, class_column="class")
preprocessed_df = reader.fit_transform(ordinal_features=categorical_features)
# Renaming the last column
preprocessed_df.rename(columns={preprocessed_df.columns[-1]: 'Class'}, inplace=True)

preprocessed_path = os.path.join(dataset_path, "Preprocessing/mushroom.csv")
preprocessed_df.to_csv(preprocessed_path)

pca = DataPreprocessor.convert_dataframe_to_principal_components(preprocessed_df)
pca.to_csv(os.path.join(dataset_path, "Preprocessing/mushroom_pca.csv"))

umap = DataPreprocessor.convert_dataframe_to_UMAP(preprocessed_df)
umap.to_csv(os.path.join(dataset_path, "Preprocessing/mushroom_umap.csv"))