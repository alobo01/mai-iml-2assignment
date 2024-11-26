from Classes.Reader import DataPreprocessor
import os

if __name__ == "__main__":
    dataset_path = '..'
else:
    dataset_path = 'Pen-based'

arff_path = os.path.join(dataset_path, "../Datasets/pen-based.arff")
complete_df = DataPreprocessor.load_arff(arff_path)
complete_df["Class"] = complete_df["a17"]
complete_df.drop(["a17"],axis=1,inplace=True)
preprocessed_path = os.path.join(dataset_path, "Preprocessing/pen-based.csv")
complete_df.to_csv(preprocessed_path)