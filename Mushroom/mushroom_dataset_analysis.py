from classes.Reader import DataPreprocessor
import classes.analyzer as analyzer

complete_df = DataPreprocessor.get_whole_dataset_as_df(
    "../datasets/mushroom/mushroom.fold.000000.test.arff",
    "../datasets/mushroom/mushroom.fold.000000.train.arff"
)

analyzer.save_dataframe_description_analysis(complete_df)
analyzer.save_feature_distributions_by_class(complete_df)
analyzer.analyze_feature_importance(complete_df)
