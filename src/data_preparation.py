# imports
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings

warnings.filterwarnings('ignore')

class DataCleaning:
    """
    A class to handle data cleaning operations.

    Attributes:
        data (pd.DataFrame): The input DataFrame for cleaning operations.

    Methods:
        remove_null_features(max_null, inplace=True): Remove columns with more than max_null null values.
        remove_null_rows(max_null, inplace=True): Remove rows with more than max_null null values.
        fill_null_obs(inplace=True): Forward-fill null values in the DataFrame.
    """

    def __init__(self, data):
        """
        Initialize the DataCleaning object.

        Args:
            data (pd.DataFrame): The input DataFrame for cleaning operations.
        """
        self.data = data

    def remove_null_features(self, max_null, inplace=True):
        """
        Remove columns with more than max_null null values.

        Args:
            max_null (int): Maximum number of allowed null values in a column.
            inplace (bool, optional): If True, modify the DataFrame in-place. Default is True.

        Returns:
            pd.DataFrame: Cleaned DataFrame with selected features.
        """
        # Calculate the number of null values in each column
        null_counts = self.data.isnull().sum()

        # Get the column names where null counts are less than or equal to the max_null
        selected_features = null_counts[null_counts <= max_null].index.tolist()

        # Filter the DataFrame to keep only the selected features
        cleaned_data = self.data[selected_features]

        if inplace:
            self.data = cleaned_data

        return cleaned_data

    def remove_null_rows(self, max_null, inplace=True):
        """
        Remove rows with more than max_null null values.

        Args:
            max_null (int): Maximum number of allowed null values in a row.
            inplace (bool, optional): If True, modify the DataFrame in-place. Default is True.

        Returns:
            pd.DataFrame: Cleaned DataFrame with selected rows.
        """
        # Keep the rows where null counts are less than or equal to the max_null
        cleaned_data = self.data[self.data.isnull().sum(axis=1) <= max_null]

        if inplace:
            self.data = cleaned_data

        return cleaned_data

    def fill_null_obs(self, inplace=True):
        """
        Forward-fill null values in the DataFrame.

        Args:
            inplace (bool, optional): If True, modify the DataFrame in-place. Default is True.

        Returns:
            pd.DataFrame: DataFrame with null values forward-filled.
        """
        # Forward fill null observations
        null_filled_data = self.data.fillna(method='ffill', inplace=False)

        if inplace:
            self.data = null_filled_data

        return null_filled_data
    
    
class FeatureEngineering:
    """
    A class for feature engineering operations.

    Attributes:
        data (pd.DataFrame): The input DataFrame for feature engineering.
        transformation_codes (dict): A dictionary to store transformation codes for each feature.

    Methods:
        transform_features(): Apply predefined transformations to features.
        add_lagged_features(lag_values): Add lagged versions of features as new columns.
    """

    def __init__(self, data):
        """
        Initialize the FeatureEngineering object.

        Args:
            data (pd.DataFrame): The input DataFrame for feature engineering.
        """
        self.data = data
        self.transformation_codes = None

    def __tranfrom_feat__(self, feat_col, trans_code):
        """
        Apply the specified transformation to a feature column.

        Args:
            feat_col (pd.Series): The feature column to be transformed.
            trans_code (int): The transformation code.

        Returns:
            pd.Series: The transformed feature column.
        """
        # no transformation
        if trans_code == 1:
            return feat_col
        
        # Δ(x_t) transformation
        elif trans_code == 2:
            return feat_col.diff()
        
        # Δ^2(x_t) transformation
        elif trans_code == 3:
            return feat_col.diff(periods=2)
        
        # log(x_t) transformation
        elif trans_code == 4:
            return feat_col.apply(np.log)
        
        # Δ(log(x_t)) transformation
        elif trans_code == 5:
            feat_col = feat_col.apply(np.log).diff(periods=2)
            return feat_col
        
        # Δ^2(log(x_t)) transformation
        elif trans_code == 6:
            feat_col = feat_col.apply(np.log).diff(periods=2)
            return feat_col
        
        # Δ((x_t)/(x_t-1)-1) transformation
        elif trans_code == 7:
            feat_col = feat_col.pct_change().diff()
            return feat_col

    def transform_features(self):
        """
        Apply predefined transformations to features.

        Updates:
            data (pd.DataFrame): DataFrame with transformed features.
            transformation_codes (dict): Updated transformation codes for each feature.
        """
        # Keep transformation codes for each feature in a dictionary 
        transformation_codes = {}
        
        # Create an empty DataFrame to store transformed features
        df_tmp = pd.DataFrame(columns=self.data.columns)
        
        # Extract feature (column) names and transformation code
        for col in self.data.columns:
            df_tmp[col] = self.data[col].iloc[1:] 
            transformation_codes[col] = self.data[col].iloc[0]
        
        # Update the class's data and transformation codes
        self.data = df_tmp
        self.transformation_codes = transformation_codes
        
        # Change 'Date' column to datetime datatype
        df_tmp['Date'] = pd.to_datetime(df_tmp['Date'])
        
        # Apply transformation to each feature
        data_transformed = pd.DataFrame(columns=self.data.columns)
        for col in self.data.columns:
            if col == 'Date':
                data_transformed[col] = self.data[col]
            else:
                data_transformed[col] = self.__tranfrom_feat__(self.data[col], transformation_codes[col])
        self.data = data_transformed

    def add_lagged_features(self, lag_values):
        """
        Add lagged versions of features as new columns.

        Args:
            lag_values (list): List of lag values.

        Returns:
            pd.DataFrame: DataFrame with added lagged features.
        """
        for col in self.data.drop(['Date'], axis=1):
            for n in lag_values:
                self.data['{} {}M Lag'.format(col, n)] = self.data[col].shift(n).ffill().values
        self.data.dropna(axis=0, inplace=True)
        return self.data


class TrendFiltering:
    """
    A class for performing trend filtering on market data.

    Attributes:
        mkt_data (pd.DataFrame): The market data DataFrame.

    Methods:
        l1_trend_filter(lambda_val=0.16): Apply the l1-trend-filtering algorithm.
    """

    def __init__(self, mkt_data):
        """
        Initialize the TrendFiltering object.

        Args:
            mkt_data (pd.DataFrame): The market data DataFrame.
        """
        self.mkt_data = mkt_data
    
    def __calc_return__(self, data_col='Close'):
        """
        Calculate the returns of the market data.

        Args:
            data_col (str): The column to calculate returns for.

        Updates:
            mkt_data (pd.DataFrame): Updated market data DataFrame with return column.
        """
        self.mkt_data['Return'] = self.mkt_data[data_col].pct_change()
        self.mkt_data.dropna(inplace=True)
        self.mkt_data = self.mkt_data[['Close', 'Return']]
        
    def l1_trend_filter(self, lambda_val=0.16):
        """
        Apply the l1-trend-filtering algorithm.

        Args:
            lambda_val (float): The lambda parameter for the algorithm.

        Returns:
            pd.DataFrame: DataFrame with market regime labels.
        """
        # Calculate return and keep required columns
        self.__calc_return__()
        ret_series = self.mkt_data['Return'].values
        
        # Size of D matrix
        n = np.size(ret_series)
        x_ret = ret_series.reshape(n)
        
        # Create D matrix
        Dfull = np.diag([1]*n) - np.diag([1]*(n-1), 1)
        D = Dfull[0:(n-1),]
        
        # Define and solve the optimization problem
        beta = cp.Variable(n)
        lambd = cp.Parameter(nonneg=True)
        
        def tf_obj(x, beta, lambd):
            return cp.norm(x - beta, 2)**2 + lambd * cp.norm(cp.matmul(D, beta), 1)
        
        problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))
        lambd.value = lambda_val
        problem.solve()
        
        # Add labels to data
        betas_df = pd.DataFrame({'TFBeta': beta.value}, index=self.mkt_data.index)
        betas_df['MktRegime'] = betas_df['TFBeta'].apply(lambda x: 0 if x > 0 else 1)
        self.mkt_data = pd.concat([self.mkt_data, betas_df], axis=1)  
        
        return self.mkt_data[['MktRegime']]