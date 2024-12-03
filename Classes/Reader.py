import pandas as pd
from scipy.io.arff import loadarff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os

from sklearn.decomposition import PCA


class DataPreprocessor:
    """
    A class for preprocessing data with support for ARFF files, handling both numerical and categorical features.
    Applies OrdinalEncoder to specified ordinal features, TargetEncoder to other categorical features,
    and MinMaxScaler to numerical features.
    """

    def __init__(self, data=None, class_column='class'):
        """
        Initialize the DataPreprocessor instance.

        Parameters:
        -----------
        data : pandas.DataFrame or str, optional
            Data to be preprocessed. Can be a DataFrame or a path to an ARFF file.
        class_column : str, default='class'
            The name of the target class column if present in the dataset.
        """
        self.preprocessor = None
        self.data = None
        self.feature_names_ = []
        self.categorical_cols = []
        self.numeric_cols = []
        self.class_column = class_column
        self.class_encoder = LabelEncoder()
        self.has_class = False
        self.class_data = None

        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str):
            try:
                self.data = self.load_arff(data)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file {data} was not found.")
            except Exception as e:
                raise Exception(f"Error loading ARFF file: {str(e)}")

    def fit(self, data=None, ordinal_features=None):
        """
        Fit the preprocessor to the data.

        Parameters:
        -----------
        data : pandas.DataFrame, optional
            The input data to fit on. Uses the data provided during initialization if None.
        ordinal_features : list of str, optional
            List of categorical features to encode using OrdinalEncoder.
        """
        if ordinal_features is None:
            ordinal_features = []
        if data is None:
            if self.data is None:
                raise ValueError("No data provided. Either pass data to fit() or initialize with data.")
            data = self.data.copy()

        self.has_class = self.class_column in data.columns
        if self.has_class:
            self.class_data = data[self.class_column].copy()
            self.class_encoder.fit(self.class_data.dropna())
            data = data.drop(columns=[self.class_column])

        self.categorical_cols = list(data.select_dtypes(include=['object']).columns)
        self.numeric_cols = list(data.select_dtypes(include=['number']).columns)

        transformers = []
        non_ordinal_features = [col for col in self.categorical_cols if col not in ordinal_features]

        if self.categorical_cols:
            if ordinal_features:
                ordinal_steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder())
                ]
                transformers.append(('ordinal', Pipeline(ordinal_steps), ordinal_features))
                self.feature_names_.extend(ordinal_features)

            if non_ordinal_features:
                if not self.has_class:
                    raise ValueError("Target encoding requires a class column.")
                target_steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('target_encoder', TargetEncoder())
                ]
                transformers.append(('target', Pipeline(target_steps), non_ordinal_features))
                self.feature_names_.extend(non_ordinal_features)

        if self.numeric_cols:
            numeric_steps = [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ]
            transformers.append(('numeric', Pipeline(numeric_steps), self.numeric_cols))
            self.feature_names_.extend(self.numeric_cols)

        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(data, y=self.class_data if non_ordinal_features else None)

    def transform(self, data=None):
        """
        Transform the input data using the fitted preprocessor.

        Parameters:
        -----------
        data : Union[str, pandas.DataFrame, None], optional
            The input data to transform. Can be:
            - A pandas DataFrame
            - A file path to an ARFF file
            - None (uses the data provided during initialization)

        Returns:
        --------
        pandas.DataFrame
            The transformed data.

        Raises:
        -------
        ValueError
            If the preprocessor is not fitted or if the input format is invalid.
        FileNotFoundError
            If the provided file path does not exist.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # Handle different input types
        if data is None:
            if self.data is None:
                raise ValueError("No data provided. Either pass data to transform() or initialize with data.")
            data_to_transform = self.data.copy()
        elif isinstance(data, str):
            try:
                data_to_transform = self.load_arff(data)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file {data} was not found.")
            except Exception as e:
                raise Exception(f"Error loading ARFF file: {str(e)}")
        elif isinstance(data, pd.DataFrame):
            data_to_transform = data.copy()
        else:
            raise ValueError("data must be None, a pandas DataFrame, or a file path string")

        # Extract class data if present
        if self.class_column in data_to_transform.columns:
            class_data = data_to_transform[self.class_column].copy()
            data_to_transform = data_to_transform.drop(columns=[self.class_column])
        else:
            class_data = None

        # Verify all required columns are present
        missing_cols = set(self.feature_names_) - set(data_to_transform.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        # Ensure columns are in the correct order
        data_to_transform = data_to_transform[self.feature_names_]

        # Transform the data
        transformed_data = self.preprocessor.transform(data_to_transform)
        result_df = pd.DataFrame(transformed_data, columns=self.feature_names_)

        # Handle class column if present
        if class_data is not None:
            # Map unknown classes to the first known class
            class_data = class_data.map(
                lambda x: x if x in self.class_encoder.classes_ else self.class_encoder.classes_[0])
            result_df[self.class_column] = self.class_encoder.transform(class_data)

        return result_df

    def fit_transform(self, data=None, **kwargs):
        """Fit and transform data in one step."""
        self.fit(data, **kwargs)
        return self.transform(data)

    def save(self, filepath):
        """Save the fitted preprocessor to a file."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Nothing to save.")

        save_dict = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names_,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
            'class_column': self.class_column,
            'class_encoder': self.class_encoder,
            'has_class': self.has_class
        }
        joblib.dump(save_dict, filepath)

    @classmethod
    def load(cls, filepath):
        """Load a saved preprocessor from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} was not found.")

        preprocessor = cls()
        save_dict = joblib.load(filepath)

        preprocessor.preprocessor = save_dict['preprocessor']
        preprocessor.feature_names_ = save_dict['feature_names']
        preprocessor.categorical_cols = save_dict['categorical_cols']
        preprocessor.numeric_cols = save_dict['numeric_cols']
        preprocessor.class_column = save_dict['class_column']
        preprocessor.class_encoder = save_dict['class_encoder']
        preprocessor.has_class = save_dict['has_class']

        return preprocessor

    @staticmethod
    def load_arff(file_path):
        """Load an ARFF file and return it as a DataFrame."""
        data, _ = loadarff(file_path)
        df = pd.DataFrame(data)
        for column in df.select_dtypes([object]).columns:
            df[column] = df[column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        df = df.replace("?", np.nan)
        return df

    @staticmethod
    def get_whole_dataset_as_df(path1, path2):
        """Load and concatenate training and test datasets from ARFF files."""
        test_data = DataPreprocessor.load_arff(path1)
        train_data = DataPreprocessor.load_arff(path2)
        return pd.concat([train_data, test_data], ignore_index=True)

    @staticmethod
    def get_columns_with_missing_values_over_threshold(data, threshold=0.4):
        """Identify columns with missing values above a specified threshold."""
        missing_percentage = data.isnull().mean()
        return missing_percentage[missing_percentage > threshold].index

    import pandas as pd

    @staticmethod
    def convert_dataframe_to_principal_components(dataframe, n_components=None):
        # Separate indices and features
        indices = dataframe.iloc[:, 0]
        features = dataframe.iloc[:, 1:]

        # Perform PCA
        pca = PCA()
        principal_components = pca.fit_transform(features)

        # Create a new DataFrame with the indices and principal components
        principal_df = pd.DataFrame(
            principal_components,
            columns=[f"PC{i + 1}" for i in range(principal_components.shape[1])]
        )
        principal_df.insert(0, "Index", indices)

        return principal_df

