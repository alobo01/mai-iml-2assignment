from Classes.Reader import DataPreprocessor
import os
import pandas as pd

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Pen-based'

arff_path = os.path.join(dataset_path, "../Datasets/pen-based.arff")
complete_df = DataPreprocessor.load_arff(arff_path)

# Initialize and fit the preprocessor on the training data and transform
reader = DataPreprocessor(complete_df, class_column="a17")
preprocessed_df = reader.fit_transform()

# Renaming the last column
preprocessed_df.rename(columns={preprocessed_df.columns[-1]: 'Class'}, inplace=True)

preprocessed_path = os.path.join(dataset_path, "Preprocessing/pen-based.csv")
preprocessed_df.to_csv(preprocessed_path)

pca = DataPreprocessor.convert_dataframe_to_principal_components(preprocessed_df)
pca.to_csv(os.path.join(dataset_path, ("Preprocessing/pen-based_pca.csv")), index=False)