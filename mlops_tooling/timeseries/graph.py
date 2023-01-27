from matplotlib import pyplot as plt
import pandas as pd

def plot_forecast(
    df: pd.DataFrame,
    x: str,
    y: str = None,
    y_hat: str = None,
    y_upper: str = None,
    y_lower: str = None
):
    """
    Plots the forecast of a given variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.
    x : str
        Name of the column containing the data.
    y : str, optional
        Name of the column containing the target.
    y_hat : str, optional
        Name of the column containing the forecast.
    y_upper : str, optional
        Name of the column containing the upper bound for the forecast.
    y_lower : str, optional
        Name of the column containing the lower bound for the forecast.
    
    Returns
    -------
    Plotted figure of the actuals, the forecast, and the prediction interval.
    """
    if (not y)&(not y_hat):
        raise ValueError("You must specify at least one of y or y_hat")
    
    plt.figure(figsize=(10, 5))
    
    x = df[x]
    
    if y:
        y = df[y]
        plt.plot(x, y, linewidth=1, color="black", label=f"{y}")
    
    if y_hat:
        y_hat = df[y_hat]
        plt.plot(x, y_hat, linewidth=1, color="black", label=f"Predicted {y}", linestyle='dashed')
    
    if (y_upper is not None)&(y_lower is not None):
        lower_bound = df[y_upper]
        upper_bound = df[y_lower]

        plt.fill_between(
            x,
            lower_bound,
            upper_bound,
            color="blue",
            alpha=0.3,
            label="Prediction Interval",
        )
        
    plt.xlabel("Date")
    plt.ylabel(f"{y}")
    plt.legend()
    plt.show()
