import numpy as np


def wape(y_true, y_pred):
    masked_arr = ~((y_pred == 0) & (y_true == 0))
    y_pred, y_true = y_pred[masked_arr], y_true[masked_arr]
    numerator = np.sum(np.abs((y_true - y_pred)))
    denominator = np.sum(np.abs(y_true))
    metric = numerator / denominator

    return metric


def wape_eval(y_true, y_pred):
    wape_val = wape(y_true, y_pred)
    return "WAPE", wape_val, False


def lgbm_wape(preds, train_data):
    labels = train_data.get_label()
    wape_val = wape(labels, preds)
    return "WAPE", wape_val, False


def smape(y_true, y_pred):
    n = len(y_pred)
    masked_arr = ~((y_pred + y_true == 0))
    y_pred, y_true = y_pred[masked_arr], y_true[masked_arr]
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_pred) + np.abs(y_true)) / 2
    smape_val = (100 * np.sum(num / denom)) / n
    return smape_val


def smape_eval(y_true, y_pred):
    smape_val = smape(y_true, y_pred)
    return "SMAPE", smape_val, False


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(labels, preds)
    return "SMAPE", smape_val, False


def rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    residuals = y_true.squeeze() - y_pred
    rmse = np.sqrt(sum([res**2 for res in residuals]) / len(residuals))
    return rmse
