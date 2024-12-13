import os
class Cleaner:
    """
    Class defined as "Cleaner".
    """
    def __init__(self, 
                columns_to_drop: list = ['Type of Property','Type of Sale','Url','Any Fireplace ?'],
                columns_to_dropna: list = ['Price','Median Price per Commune','Median Revenue per Commune','Primary Energy Consumption (kWh/m2)','Construction Year','State of the Building', 'Number of Facades', 'Number of Rooms', 'PEB'],
                property_subtypes_to_drop: list = ['apartment-block','mixed-use-building','other-property','mansion','country-cottage',
                                                'bungalow','farmhouse','chalet','castle','manor-house'],
                columns_to_fillna: list = ['Garden Area (m2)','Surface of the Land (m2)','Terrace Area (m2)'],
                columns_to_map: list = ['State of the Building','PEB'],
                columns_with_duplicates: list = ['Url'],
                except_columns: list =['Subtype of Property', 'Locality'],
                
                max_thresholds: dict = {
                'Price/m2': 10000,
                'Price':3000000,
                'Livable Space (m2)': 1000,
                'Surface of the Land (m2)': 10000,
                'Primary Energy Consumption (kWh/m2)': 1000,
                'Garden Area (m2)': 1000},
                
                min_thresholds: dict = {
                'Price/m2': 0,
                'Price': 0,
                'Livable Space (m2)': 0,
                'Surface of the Land (m2)': 0,
                'Primary Energy Consumption (kWh/m2)': 0,
                'Garden Area (m2)': 0}
                ):
        
        """
        Function to initialize Class "Cleaner".
        :default parameters: to be modified here if needed
        """

        self.columns_to_drop = columns_to_drop
        self.columns_to_dropna = columns_to_dropna
        self.property_subtypes_to_drop = property_subtypes_to_drop
        self.columns_to_fillna = columns_to_fillna
        self.columns_to_map = columns_to_map
        self.columns_with_duplicates = columns_with_duplicates
        self.except_columns = except_columns
        self.max_thresholds = max_thresholds
        self.min_thresholds = min_thresholds

    def drop_columns(self, df):
        
        """
        Function to : 
        * drop useless columns (considered as such after analysis)
        * drop rows with null values in key column : 'Price'
        * drop rows with unclear subtypes of property
        """
        df = df.drop_duplicates(subset=self.columns_with_duplicates)
        df= df.drop(columns=self.columns_to_drop)
        df = df.dropna(subset=self.columns_to_dropna)
        df['Price/m2'] = df['Price'] / df['Livable Space (m2)']

        for subtype in self.property_subtypes_to_drop:
            df = df[df['Subtype of Property'] != subtype]

        return df
    
    def fill_columns(self, df):
        
        """
        Function to: 
        * fill in the columns needed in dataframe 
        """

        for column in self.columns_to_fillna:

            if column == 'Surface of the Land (m2)':
                
                df.loc[df['Type of Property'] == 'Apartment', 'Surface of the Land (m2)'] = 0
                df = df.dropna(subset=['Surface of the Land (m2)'])
            
            if column == 'Construction Year':
                df[column] = df[column].fillna(df[column].mean()) 

            else:
                df[column] = df[column].fillna(0)
        
        return df
    
    def map_columns(self, df):
        
        """
        Function to: 
        * map the categorical columns in dataframe with corresponding numerical value

        """
        for column in self.columns_to_map:

            if column == 'State of the Building':
                state_mapping = {
                    'To restore': 0,
                    'To renovate': 1,
                    'To be done up': 2,
                    'Good': 3,
                    'Just renovated': 4,
                    'As new': 5}
                
                df['State of the Building'] = df['State of the Building'].replace(state_mapping)
            
            elif column == 'PEB':
                PEB_mapping = {
                    'G': 0,
                    'F': 1,
                    'E': 2,
                    'D': 3,
                    'C': 4,
                    'B': 5,
                    'A': 6}

                df['PEB'] = df['PEB'].replace(PEB_mapping)
            
        return df
        
    def remove_outliers(self, df):
        
        """
        Function to: 
        * remove outliers and select a range of entries that fit the most to model needs
        """
        
        for column, threshold in self.max_thresholds.items():
            df = df[df[column] <= threshold]

        for column, threshold in self.min_thresholds.items():
            df = df[df[column] >= threshold]
        
        return df
    
    def values_as_integers(self, df):

        """
        Function to: 
        * set all numerical values as integers
        """
        
        df_droped = df.drop(columns=self.except_columns)
        columns_list = df_droped.columns.tolist()

        for column in columns_list:
            df[column] = df[column].astype(int)
        
        return df
    
def clean_dataset(df):

    # Initialize the Cleaner class

    data_cleaner = Cleaner()
    
    # Perform the cleaning steps

    
    df = data_cleaner.fill_columns(df)
    df = data_cleaner.map_columns(df)
    df = data_cleaner.drop_columns(df)
    df = data_cleaner.remove_outliers(df)
    df = data_cleaner.values_as_integers(df)

    # # Save the cleaned DataFrame

    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dataset_path = os.path.join(script_dir, '../data/cleaned_dataset.csv')
    df.to_csv(clean_dataset_path, index=False)
    
    return df