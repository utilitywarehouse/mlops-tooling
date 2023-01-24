

from typing import Optional, List
import pandas as pd

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
    ):
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
        
        self._force_correct_typings()
        
    def prepare_dataset(self) -> pd.DataFrame:
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
            
        flattened_df = self.df.drop(columns = columns_to_drop).dropna().reset_index(drop=True)
        
        dt_idx = flattened_df[[self.date_col]]
        y = flattened_df[[self.target_col]]
        X = flattened_df.drop(columns = [self.target_col, self.date_col])
        
        return X, y, dt_idx
    
    def add_date_info(self) -> pd.DataFrame:
        if self.date_features:
            for date_feat_name, date_feat_func in self.date_features.items():
                self.df = self.get_date_feature(self.df, self.date_col, date_feat_name, date_feat_func)
                
    def add_target_lags(self) -> pd.DataFrame:
        for lag in self.lags:
            if self.group_cols:
                self.df[f"{self.target_col}_lag_{lag}"] = self.df.groupby(self.group_cols)[self.target_col].shift(lag)
                self.df[f"{self.target_col}_roll_{lag}"] = self.df.groupby(self.group_cols)[self.target_col].apply(lambda v: v.shift(1).rolling(lag).mean())
            else:
                self.df[f"{self.target_col}_lag_{lag}"] = self.df[self.target_col].shift(lag)
                self.df[f"{self.target_col}_roll_{lag}"] = self.df[self.target_col].shift(1).rolling(lag).mean()
            
    def add_covariate_lags(self) -> pd.DataFrame:
        for covariate in self.past_covariates:
            for lag in self.covariate_lags:
                if self.group_cols:
                    self.df[f"{covariate}_lag_{lag}"] = self.df.groupby(self.group_cols)[covariate].shift(lag)
                else:
                    self.df[f"{covariate}_lag_{lag}"] = self.df[covariate].shift(lag)
                
    def add_covariate_leads(self) -> pd.DataFrame:
        for covariate in self.future_covariates:
            for lead in self.covariate_leads:
                if self.group_cols:
                    self.df[f"{covariate}_lead_{lead}"] = self.df.groupby(self.group_cols)[covariate].shift(-lead)
                else:
                    self.df[f"{covariate}_lead_{lead}"] = self.df[covariate].shift(-lead)

    @staticmethod  
    def get_date_feature(df: pd.DataFrame, date_col: str, date_feature: str, date_feature_func: Optional[str] = None):
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
        try:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        except:
            pass
        
        try:
            for column in self.group_cols:
                self.df[column] = self.df[column].astype('category')
        except:
            pass