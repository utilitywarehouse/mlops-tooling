import pandas as pd
import scipy
import datetime as dt
from typing import Optional, List
import optuna
import lightgbm as lgb
from mlops_tooling.timeseries import Timeseries
from mlops_tooling.timeseries.metrics import wape_eval, rmse


class LGBMForecaster:
    def __init__(
        self,
        date_col: str,
        target_col: str,
        group_cols: Optional[List[str]] = None,
        lags: Optional[List[int]] = [1],
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
        self.lgbm_params = None

        if seasonal_order:
            self.seasonal_order = seasonal_order
        elif seasonal_periods:
            self.seasonal_order = 4

        if seasonal_trend_order:
            self.seasonal_trend_order = seasonal_trend_order
        elif seasonal_trend:
            self.seasonal_trend_order = 1

    def has_trend(self):
        pass

    def get_input(self, df, date=None):
        df_timeseries = Timeseries(
            df,
            self.date_col,
            self.target_col,
            self.group_cols,
            self.lags,
            self.prediction_steps,
            self.static_covariates,
            self.past_covariates,
            self.covariate_lags,
            self.future_covariates,
            self.covariate_leads,
            self.date_features,
            self.seasonal_periods,
            self.seasonal_order,
            self.seasonal_trend,
            self.seasonal_trend_order,
        )

        X, y, idx = df_timeseries.prepare_dataset()

        if date:
            X = X[idx == date]
            y = y[idx == date]

        return X, y, idx

    def fit(
        self,
        df: pd.DataFrame,
        train_start_date: str,
        val_start_date: str,
        test_start_date: str,
        lgbm_params: dict = None,
    ):
        self.train_start_date = train_start_date
        self.val_start_date = val_start_date
        self.test_start_date = test_start_date

        X, y, self.idx = self.get_input(df)

        train_X, train_y, val_X, val_y, test_X, test_y = self._split_data(
            X, y, train_start_date, val_start_date, test_start_date
        )

        if self.seasonal_trend:
            X_train, trend_train = self._detrend(X_train)
            X_val, trend_val = self._detrend(X_val)
            X_test, trend_test = self._detrend(X_test)

        if lgbm_params is None:
            self.lgbm_params = {
                "verbosity": -1,
                "boosting_type": "gbdt",
                "seed": 42,
                "linear_tree": False,
                "learning_rate": 0.1,
                "min_child_samples": 5,
                "num_leaves": 31,
            }
        else:
            self.lgbm_params = lgbm_params

        self.lgbm_model = lgb.LGBMRegressor(**self.lgbm_params)

        self.lgbm_model.fit(
            train_X,
            train_y,
            eval_metric=wape_eval,
            eval_set=[(val_X, val_y)],
            verbose=0,
            early_stopping_rounds=100,
        )

        y_test = test_y.squeeze()
        test_forecast = self.lgbm_model.predict(X_test)

        y_val = val_y.squeeze()
        val_forecast = self.lgbm_model.predict(X_val)

        if self.seasonal_trend:
            val_forecast = val_forecast * trend_val.squeeze()
            y_val = y_val * trend_val.squeeze()

            test_forecast = test_forecast * trend_test.squeeze()
            y_test = y_test * trend_test.squeeze()

        self.val_rmse = rmse(y_val, val_forecast)
        self.test_wape = wape_eval(y_test, test_forecast)

        print(f"Test WAPE: {self.test_wape}")

    def _split_data(self, X, y, train_start_date, val_start_date, test_start_date):
        train_X = X[
            (self.idx["week_start_date"] >= train_start_date)
            & (self.idx["week_start_date"] < val_start_date)
        ]
        train_y = y[
            (self.idx["week_start_date"] >= train_start_date)
            & (self.idx["week_start_date"] < val_start_date)
        ]

        val_X = X[
            (self.idx["week_start_date"] >= val_start_date)
            & (self.idx["week_start_date"] < test_start_date)
        ]
        val_y = y[
            (self.idx["week_start_date"] >= val_start_date)
            & (self.idx["week_start_date"] < test_start_date)
        ]

        test_X = X[self.idx["week_start_date"] >= test_start_date]
        test_y = y[self.idx["week_start_date"] >= test_start_date]

        return train_X, train_y, val_X, val_y, test_X, test_y

    @staticmethod
    def _detrend(X: pd.DataFrame):
        trend = X[["trend"]]
        X = X.drop(columns=["trend"])

        return X, trend

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        param = {
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["gbdt", "goss"]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, step=2),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000, step=10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 0.25),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 50),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 50),
        }

        lgbm = lgb.LGBMRegressor(verbose=-1, **param)

        lgbm.fit(
            X_train,
            y_train,
            eval_metric=wape_eval,
            eval_set=[(X_val, y_val)],
            verbose=-1,
            early_stopping_rounds=100,
            categorial_columns=self.group_cols,
        )

        forecast = lgbm.predict(X_val)
        wape_score = wape_eval(y_val, forecast)[1]

        return wape_score

    def _optimise(self, trial, n_trials, X_train, y_train, X_val, y_val):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=600,
        )

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return trial.params

    def fit_optimize(
        self,
        df: pd.DataFrame,
        train_start_date: str,
        val_start_date: str,
        test_start_date: str,
        n_trials: int = 1000,
    ):
        self.train_start_date = train_start_date
        self.val_start_date = val_start_date
        self.test_start_date = test_start_date

        X, y, self.idx = self.get_input(df)

        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(
            X, y, train_start_date, val_start_date, test_start_date
        )

        if self.seasonal_trend:
            X_train, trend_train = self._detrend(X_train)
            X_val, trend_val = self._detrend(X_val)
            X_test, trend_test = self._detrend(X_test)

        self.lgbm_params = self._optimise(
            self._objective, n_trials, X_train, y_train, X_val, y_val
        )

        self.fit(
            df,
            self.train_start_date,
            self.val_start_date,
            self.test_start_date,
            self.lgbm_params,
        )

    def predict(
        self,
        df: pd.DataFrame,
        forecast_start_date: str,
        period: int,
        horizon: int,
        quantiles: List[float],
    ):
        results = []
        col_names = [self.date_col, self.target_col + "_forecast"]

        df = df.copy(deep=True)

        for i in range(horizon):
            pred_date = forecast_start_date + dt.timedelta(days=i * period)
            str_date = pred_date.strftime("%Y-%m-%d 00:00:00")

            X, _, _ = self.get_input(df)

            X = X[self.idx[self.date_col] == str_date]

            y_hat = self.lgbm_model.predict(X)

            df.loc[df[self.date_col] == str_date, "sign_ups"] = y_hat

            result = [str_date, y_hat]

            if quantiles:
                for alpha in quantiles:
                    band_scale = scipy.stats.norm.ppf(alpha)
                    lower_band = -band_scale * self.val_rmse
                    upper_band = band_scale * self.val_rmse

                    lower_name = f"{self.target_col}_{round(1 - alpha, 2)}th_pc"
                    upper_name = f"{self.target_col}_{round(alpha, 2)}th_pc"

                    result.extend([lower_band, upper_band])
                    col_names.extend([lower_name, upper_name])

            results.append(result)

        return pd.DataFrame(results, columns=col_names)
