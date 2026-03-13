import pandas as pd


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