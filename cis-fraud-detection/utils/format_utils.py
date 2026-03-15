import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def rename_columns(df):
    for col in df.columns:
        if str(col).startswith('id'):
            col_name = col.replace('id-', 'id_')
            df = df.rename(columns={col: col_name})
    return df


def check_duplicates(df):
    print("Duplicate Rows :", df[df.duplicated()]  )
    print("Duplicate Columns :", df.columns[df.columns.duplicated()])


def missing_values(df):
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100
    }).reset_index(drop=True)
    return missing_df[missing_df['missing_pct'] > 0]


def group_by_feature(feature: str, df) -> None:

    """
    Groups the data by a feature and calculates the total count and fraud count
    """

    result = (
        df
        .groupby(feature)
        .agg(
            total_count=(feature, "count"),
            fraud_count=("isFraud", "sum")
        )
        .reset_index()
    )

    result["fraud_pct"] = result["fraud_count"] / result["total_count"]
    result["fraud_propensity"] = result["fraud_count"] * result["fraud_pct"] 
    result.sort_values(by=["fraud_propensity"], ascending=False, inplace=True)

    return result

def unique_categories(df, data_summary):
    """
    Adds a 'unique_categories' column to data_summary, counting unique values for categorical features.
    """
    unique_cats = []
    for _, row in data_summary.iterrows():
        feature = row['feature']
        if row['type'] == 'categorical' and feature in df.columns:
            unique_cats.append(df[feature].nunique())
        else:
            unique_cats.append(None)
    data_summary = data_summary.copy()
    data_summary['unique_categories'] = pd.Series(unique_cats, dtype='Int64')
    return data_summary