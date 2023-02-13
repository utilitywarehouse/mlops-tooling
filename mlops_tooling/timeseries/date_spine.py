import pandas as pd


def month_converter(month_of_year: int):
    months = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]
    return months[month_of_year - 1]


def create_date_table(
    fy_month_start: int = 4,
    bank_hol_df: pd.DataFrame = None,
    start: str = "2010-01-01",
    end: str = "2030-12-31",
):
    df = pd.DataFrame({"date": pd.date_range(start, end)})
    df["week_start_date"] = df.date.dt.to_period("W-SUN").dt.start_time

    df["weekday"] = df.date.dt.day_name()
    df["day_of_week"] = df.date.dt.dayofweek
    df["day_of_year"] = df.date.dt.dayofyear
    df["week_of_year"] = df.date.dt.week
    df["month_of_year"] = df.date.dt.month
    df["month_name"] = df.date.dt.month_name()
    df["quarter_of_year"] = df.date.dt.quarter
    df["half_of_year"] = df.date.dt.month.map(lambda mth: 1 if mth < 7 else 2)
    df["year"] = df.date.dt.year

    fy_month_str = month_converter(fy_month_start - 1)

    df["financial_month_of_year"] = (
        (df.date.dt.month + (12 - fy_month_start)) % 12
    ) + 1
    df["financial_quarter_of_year"] = df.date.dt.to_period(
        f"Q-{fy_month_str}"
    ).dt.quarter
    df["financial_half_of_year"] = df.financial_month_of_year.map(
        lambda mth: 1 if mth < 7 else 2
    )
    df["financial_year"] = df.date.dt.to_period(f"Q-{fy_month_str}").dt.qyear

    fy_date_df = df[["financial_year", "date"]].sort_values(
        ["financial_year", "date"], ascending=True
    )

    fy_date_df["financial_day_of_year"] = (
        fy_date_df.groupby(["financial_year"]).cumcount() + 1
    )

    fy_week_df = (
        df[["financial_year", "week_start_date"]]
        .drop_duplicates()
        .groupby(["week_start_date"])["financial_year"]
        .min()
        .reset_index()
        .sort_values(["financial_year", "week_start_date"], ascending=True)
    )

    fy_week_df["financial_week_of_year"] = (
        fy_week_df.groupby(["financial_year"]).cumcount() + 1
    )

    df = df.merge(fy_date_df).merge(fy_week_df)[
        [
            "date",
            "week_start_date",
            "weekday",
            "day_of_week",
            "day_of_year",
            "week_of_year",
            "month_of_year",
            "month_name",
            "quarter_of_year",
            "half_of_year",
            "year",
            "financial_day_of_year",
            "financial_week_of_year",
            "financial_month_of_year",
            "financial_quarter_of_year",
            "financial_half_of_year",
            "financial_year",
        ]
    ]

    df = df[
        (df["financial_year"] > df["financial_year"].min())
        & (df["financial_year"] < df["financial_year"].max())
    ]

    if bank_hol_df is not None:
        df = df.merge(bank_hol_df, left_on="date", right_on="date", how="left")
        df["is_bank_holiday"] = df["is_bank_holiday"].fillna(0).astype(int)

    return df
