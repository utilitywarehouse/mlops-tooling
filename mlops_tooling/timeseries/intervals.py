import pandas as pd
import numpy as np
import scipy.stats

def bootstrap_prediction_interval(model, x_val: pd.DataFrame, y_val: pd.DataFrame, n: int = 100, alpha: float = 0.05):
    """
    Calculates residuals from a validation set, and bootstraps the values to create a prediction interval banding.
    Only use unseen values to generate the prediction interval banding.

    Parameters
    ----------
    model : sklearn style model
        Model used to generate predictions.
    x_val : pd.DataFrame
        Data to input to the model.
    y_val : pd.DataFrame
        Ground truth values for the predictions.
    n : int
        Number of bootstrap samples.
    alpha: float
        Significance level of the prediction interval banding.
        
    Returns
    ----------
    lower_band : float
        Banding to subtract from the prediction to generate a prediction interval.
    upper_band : float
        Banding to add to the prediction to generate a prediction interval.
    """
    residuals = y_val.squeeze() - model.predict(x_val)

    bootstrap = np.asarray([np.random.choice(residuals, size=residuals.shape) for i in range(n)])
    bootstrap_quantiles = np.quantile(bootstrap, q=[alpha/2, 1-alpha/2], axis=0)
    
    lower_band = bootstrap_quantiles[0].mean()
    upper_band = bootstrap_quantiles[1].mean()
    
    return lower_band, upper_band

def rmse_prediction_interval(model, x_val: pd.DataFrame, y_val: pd.DataFrame, alpha: float = 0.05):
    """
    Calculates residuals from a validation set, and uses RMSE to calculate a prediction interval banding.
    Only use unseen values to generate the prediction interval banding.
    
    Parameters
    ----------
    model : sklearn style model
        Model used to generate predictions.
    x_val : pd.DataFrame
        Data to input to the model.
    y_val : pd.DataFrame
        Ground truth values for the predictions.
    alpha: float
        Significance level of the prediction interval banding.
        
    Returns
    ----------
    lower_band : float
        Banding to subtract from the prediction to generate a prediction interval.
    upper_band : float
        Banding to add to the prediction to generate a prediction interval.
    """
    residuals = y_val.squeeze() - model.predict(x_val)
    rmse = np.sqrt(sum([res**2 for res in residuals]) / len(residuals))
    
    band_scale = scipy.stats.norm.ppf(alpha)
    
    lower_band = -band_scale*rmse
    upper_band = band_scale*rmse

    return lower_band, upper_band
