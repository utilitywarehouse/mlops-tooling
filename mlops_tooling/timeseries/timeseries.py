from typing import Optional, List
import pandas as pd
import numpy as np
from scipy import stats

class Timeseries:
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        group_cols: Optional[List[str]] = None,
        lags: Optional[List[int]] = [1],
        prediction_steps: Optional[List[int]] = [1],
        static_covariates: Optional[List[str]] = None,
        past_covariates: Optional[List[str]] = None,
        covariate_lags: Optional[List[int]] = [1],
        future_covariates: Optional[List[str]] = None,
        covariate_leads: Optional[List[int]] = [1],
        date_features: Optional[dict] = None,
        seasonal_periods: Optional[List[int]] = None,
        seasonal_order: Optional[int] = None,
        seasonal_trend: Optional[bool] = False,
        seasonal_trend_order: Optional[int] = None,
    ):
        """
        A class used to create a flattened timeseries dataframe, from a time based dataframe.

        Args:
            df : pd.DataFrame
                Pandas dataframe containing the time series data.
            date_col : str
                Name of the date column in the dataframe.
            target_col : str
                Name of the target column in the dataframe.
            group_cols : Optional[List[str]]
                Columns to group the data by, usually information inherent to a class. 
                If no groups are specified, the data is assumed to be from a single class.
            lags : Optional[List[int]]
                Lagged target variable values to add to the dataframe. Defaults to [1].
            prediction_steps : Optional[List[int]]
                Steps a future model should predict if building a single model.
                This will add a column 'step' which is linked to the output. Defaults to [1]. 
                Currently not implemented.
            static_covariates : Optional[List[str]]
                Name of static covariates in the model. Defaults to None.
            past_covariates : Optional[List[str]]
                Name of dynamic covariates in the model for which we want to include past data.
                Defaults to None.
            covariate_lags : Optional[List[int]]
                Lagged covariable values to add to the dataframe.
                Defaults to None.
            future_covariates : Optional[List[str]]
                Name of dynamic covariates in the model for which we want to include future data.
                Defaults to None.
            covariate_leads : Optional[List[int]]
                Leading covariable values to add to the dataframe.
                Defaults to None.
            date_features : Optional[dict]
                Dictionary of date features to add to the dataframe.
                Should be in the format {"column_name":"date_function"}.
                Available functions can be found here under attributes at the following link: 
                https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DatetimeIndex.html
                week_of_month, is_first_week_of_month, is_last_week_of_month are also available.
                Defaults to None.
        """
        self.df = df.copy(deep=True).sort_values(by = date_col)
        self.date_col = date_col
        self.target_col = target_col
        self.group_cols = group_cols
        self.lags = lags
        self.static_covariates = static_covariates
        self.past_covariates = past_covariates
        self.covariate_lags = covariate_lags
        self.future_covariates = future_covariates
        self.covariate_leads = covariate_leads
        self.date_features = date_features
        self.seasonal_periods = seasonal_periods
        self.seasonal_trend = seasonal_trend
        
        if seasonal_order:
            self.seasonal_order = seasonal_order
        elif seasonal_periods:
            self.seasonal_order = 4
            
        if seasonal_trend_order:
            self.seasonal_trend_order = seasonal_trend_order
        elif seasonal_trend:
            self.seasonal_trend_order = 1
        
        self._force_correct_typings()
        
    def prepare_dataset(self) -> pd.DataFrame:
        """
        Flattens a timeseries dateframe into flattened X, y, and time index dataframes.
        """
        self.add_date_info()
        self.add_target_lags() 
        
        if (self.past_covariates) and (self.future_covariates):
            self.add_covariate_lags()
            self.add_covariate_leads()
            columns_to_drop = list(set(self.future_covariates + self.past_covariates))
        
        elif self.past_covariates:
            self.add_covariate_lags()
            columns_to_drop = self.past_covariates
            
        elif self.future_covariates:
            self.add_covariate_leads()
            columns_to_drop = self.future_covariates 
        
        else:
            columns_to_drop = []
            
        if self.seasonal_periods:
            for seasonal_period in self.seasonal_periods:
                self.add_fourier_harmonics(self.seasonal_order, seasonal_period)
                
        if self.seasonal_trend:
            self.add_seasonal_trend(self.seasonal_trend_order)
            
        flattened_df = self.df.drop(columns = columns_to_drop).dropna().reset_index(drop=True)
        
        dt_idx = flattened_df[[self.date_col]]
        y = flattened_df[[self.target_col]]
        X = flattened_df.drop(columns = [self.target_col, self.date_col])
        
        return X, y, dt_idx
    
    def add_date_info(self) -> pd.DataFrame:
        """
        Converts the date column to numerous columns containing information about the date.
        See https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DatetimeIndex.html for available information.
        
        An example of the date_features dictionary: 
        date_features = {   
            "month": "month",
            "quarter": "quarter",
            "year": "year",
            "week_of_year": "weekofyear",
            "week_of_month": "week_of_month",
            "is_first_week_of_month":"is_first_week_of_month",
            "is_last_week_of_month":"is_last_week_of_month",
        }

        Parameters
        ----------
        date_features : dict
            A dictionary containing {column_name : date feature} for the date column. 
            
        Returns
        ----------
        df : pd.DataFrame
            The original dataframe with date information added.
        """
        if self.date_features:
            for date_feat_name, date_feat_func in self.date_features.items():
                self.df = self.get_date_feature(self.df, self.date_col, date_feat_name, date_feat_func)
                
    def add_target_lags(self) -> pd.DataFrame:
        """
        Adds the specified lags and rolling averages of the target variable as columns for each row. 
        These columns will be called taget_col_lag_X and target_col_roll_X.

        Parameters
        ----------
        target_col : dict
            The target column to add lags for.
        lags : list
            A list of lags to add to the target column, this must be positive.
            
        Returns
        ----------
        df : pd.DataFrame
            The original dataframe with lagged target columns added.
        """
        for lag in self.lags:
            if self.group_cols:
                self.df[f"{self.target_col}_lag_{lag}"] = self.df.groupby(self.group_cols)[self.target_col].shift(lag)
                self.df[f"{self.target_col}_roll_{lag}"] = self.df.groupby(self.group_cols)[self.target_col].apply(lambda v: v.shift(1).rolling(lag).mean())
            else:
                self.df[f"{self.target_col}_lag_{lag}"] = self.df[self.target_col].shift(lag)
                self.df[f"{self.target_col}_roll_{lag}"] = self.df[self.target_col].shift(1).rolling(lag).mean()
            
    def add_covariate_lags(self) -> pd.DataFrame:
        """
        Adds the specified lags for the specified covariates as columns.

        Parameters
        ----------
        past_covariates : list
            The covariates column to add lags for.
        covariate_lags : list
            A list of lags to add for the covariates, this must be positive.
            
        Returns
        ----------
        df : pd.DataFrame
            The original dataframe with lagged covariate columns added.
        """
        for covariate in self.past_covariates:
            for lag in self.covariate_lags:
                if self.group_cols:
                    self.df[f"{covariate}_lag_{lag}"] = self.df.groupby(self.group_cols)[covariate].shift(lag)
                else:
                    self.df[f"{covariate}_lag_{lag}"] = self.df[covariate].shift(lag)
                
    def add_covariate_leads(self) -> pd.DataFrame:
        """
        Adds the specified leading values (i.e. future) for the specified covariates as columns.

        Parameters
        ----------
        future_covariates : list
            The covariates column to add leads for.
        covariate_lads : list
            A list of leads to add for the covariates, this must be positive.
            
        Returns
        ----------
        df : pd.DataFrame
            The original dataframe with leading covariate columns added.
        """
        for covariate in self.future_covariates:
            for lead in self.covariate_leads:
                if self.group_cols:
                    self.df[f"{covariate}_lead_{lead}"] = self.df.groupby(self.group_cols)[covariate].shift(-lead)
                else:
                    self.df[f"{covariate}_lead_{lead}"] = self.df[covariate].shift(-lead)

    @staticmethod  
    def get_date_feature(df: pd.DataFrame, date_col: str, date_feature: str, date_feature_func: Optional[str] = None):
        """
        Calculates a date feature using the specified date_feature_func, and adds it as a column to the dataframe with 
        the its name as date_feature.

        Parameters
        ----------
        df : pd.DataFrame
            The original timeseries dataframe.
        date_col : str
            The date column to generate information for.
        date_feature : str
            The column name of the date feature to add.
        date_feature_func : str
            The pandas function to pull the date feature from the column. 
            For week_of_month, is_last_week_of_month, is_first_week_of_month this is a custom function.
            
        Returns
        ----------
        df : pd.DataFrame
            The original dataframe with the date feature added.
        """
        if date_feature == 'week_of_month':
            df['week_start_date'] = df[date_col].dt.to_period('W-SUN').dt.start_time
            
            try:
                week_of_month_temp_df = df[['year','month','week_start_date']]
            except ValueError:
                raise ValueError(f"Did not find year and month date features")
            
            week_of_month_temp_df = week_of_month_temp_df.drop_duplicates().reset_index(drop = True)
            
            week_of_month_temp_df['week_of_month'] = week_of_month_temp_df.groupby(['year','month']).rank(method="first", ascending=True).astype(int)
            
            df = df.merge(week_of_month_temp_df, left_on = ['year', 'month', 'week_start_date'], right_on = ['year', 'month', 'week_start_date'])
        
        elif date_feature in ['is_last_week_of_month', 'is_first_week_of_month']:
            try:
                'week_of_month' in df.columns
            except ValueError:
                df = Timeseries.get_date_feature(df, date_col, 'week_of_month')
        
            weeks_in_month = df.groupby(['year', 'month'])[['week_of_month']].max().reset_index().rename(columns = {'week_of_month':'no_weeks'})
            
            df = df.merge(
                weeks_in_month,
                left_on = ['year', 'month'],
                right_on = ['year', 'month']
            )
            
            equal_to = 1 if date_feature == 'is_first_week_of_month' else df['no_weeks']
            
            df[date_feature] = df['week_of_month'] == equal_to
            df[date_feature] = df[date_feature].astype(int)
            
            df = df.drop(columns = ['no_weeks'])
            
        else:
            df[date_feature] = getattr(df[date_col].dt, date_feature_func).astype(int)
            
        return df

    def _force_correct_typings(self):
        """
        Forces the date column to be a pandas datetime type, and ensures any group columns are category types.
        """
        try:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        except:
            pass
        
        try:
            for column in self.group_cols:
                self.df[column] = self.df[column].astype('category')
        except:
            pass
    
    def add_fourier_harmonics(self, seasonal_order, seasonal_period):
        n = self.df.shape[0]
        x = 2 * np.pi * np.arange(1, seasonal_order + 1) / seasonal_period
        t = np.arange(1, n + 1)
        x = x * t[:, None]
        
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        column_names = [f"fourier_period_{seasonal_period}_cos_{i+1}" for i in range(seasonal_order)] + [f"fourier_period_{seasonal_period}_sin_{i+1}" for i in range(seasonal_order)]
        
        fourier_df = pd.DataFrame(fourier_series, columns=column_names)
        
        self.df = pd.concat([self.df, fourier_df], axis=1)
    
    def add_seasonal_trend(self, seasonal_trend_order):
        y = self.df[self.df[self.target_col] > 0][self.target_col]
        n = len(y)
        x = np.arange(1, n + 1) ** seasonal_trend_order
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        n = self.df.shape[0]
        x = np.arange(1, n + 1) ** seasonal_trend_order
        
        trend = slope * x * r_value + intercept
        
        self.df['sign_ups'] = self.df['sign_ups'] / trend
        self.df['trend'] = trend
