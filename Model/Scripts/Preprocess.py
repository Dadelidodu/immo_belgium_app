import os
from sklearn.model_selection import train_test_split

class Preprocessor:

    """
    Class defined as "Preprocessor".
    """

    def __init__(self, 
                columns_to_score: list = ['Subtype of Property','State of the Building','PEB'],
                scored_columns: list = ['Median Revenue per Commune', 'Median Price per Commune','Livable Space (m2)','Subtype of Property Score',
                                 'State of the Building Score','PEB Score','Primary Energy Consumption (kWh/m2)','Surface of the Land (m2)','Construction Year','Number of Rooms','Number of Facades']):
        
        """
        Function to initialize Class "Preprocessor".
        :default parameters: to be modified here if needed
        """

        self.columns_to_score = columns_to_score
        self.scored_columns = scored_columns

    def split_sets(self, df):

        """
        Function to:
        * split Train & Test sets ('Price' is the target)
        * stratify splitting avoiding lone classes
        """
        X = df
        y = df['Price']

        if y.nunique() > 1 and y.value_counts().min() > 1: 
            stratify_option = y
        else:
            stratify_option = None

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                            
                                                            stratify=stratify_option
                                                            )
        
        return X_train, X_test, y_train, y_test

    def score_columns(self, X_train, X_test):
        
        """
        Function to:
        * make a Score for categorical and unordered columns such as 'Zip Code'
        * Score is the result of a calculus of mean income per category
        """

        for col in self.columns_to_score:

            col_mean = X_train.groupby(col)['Median Price per Commune'].mean()
            
            X_train[f'{col} Score'] = X_train[col].map(col_mean).fillna(col_mean.mean())
            X_test[f'{col} Score'] = X_test[col].map(col_mean).fillna(col_mean.mean())
        
        return X_train, X_test

    def standardize_columns(self, X_train, X_test):
        
        """
        Function to:
        * Standardize values for Machine learning
        """

        for col in self.scored_columns:
            train_mean = X_train[col].mean()
            train_std = X_train[col].std()
            
            X_train[f'Standardized {col}'] = ((X_train[col] - train_mean) / train_std).round(3)
            X_test[f'Standardized {col}'] = ((X_test[col] - train_mean) / train_std).round(3)

        # Filter Columns for Machine Learning

        Standardized_columns = [f'Standardized {col}' for col in self.scored_columns]
        final_columns = Standardized_columns


        X_train = X_train[final_columns]
        X_test = X_test[final_columns]

        return X_train, X_test


def preprocess_dataset(df):   
    
    # Initialize the Preprocessor class

    data_preprocessor = Preprocessor()

    # Perform the preprocessing steps

    X_train, X_test, y_train, y_test = data_preprocessor.split_sets(df)
    X_train, X_test = data_preprocessor.score_columns(X_train, X_test)
    X_train, X_test = data_preprocessor.standardize_columns(X_train, X_test)

    print('\n', X_train.info())
    print(X_test.info())

    # Save to Standardized_dataset

    script_dir = os.path.dirname(os.path.abspath(__file__))

    X_train_Standardized_path = os.path.join(script_dir, '../data/X_train.csv')
    X_train.to_csv(X_train_Standardized_path, index=False)

    X_test_Standardized_path = os.path.join(script_dir, '../data/X_test.csv')
    X_test.to_csv(X_test_Standardized_path, index=False)

    y_train_path = os.path.join(script_dir, '../data/y_train.csv')
    y_train.to_csv(y_train_path, index=False)

    y_test_path = os.path.join(script_dir, '../data/y_test.csv')
    y_test.to_csv(y_test_path, index=False)

    return X_train, X_test, y_train, y_test