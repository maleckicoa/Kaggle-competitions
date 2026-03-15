
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def plot_binary_distribution(
    df,
    target="isFraud",
    labels=("Not Fraud", "Fraud"),
    colors=("gray", "orange"),
    figsize=(5, 2),
    bar_width=0.45,
    label_fontsize=8,
    ylim_factor=1.3
):
    """
    Plots the binary distribution of the target variable
    """

    counts = df[target].value_counts().sort_index()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(labels, counts, color=colors, width=bar_width)

    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=label_fontsize
        )

    # increase vertical headroom so labels don't overflow
    ax.set_ylim(0, counts.max() * ylim_factor)

    ax.set_title(f"Distribution of {target}")

    ax.margins(x=0.2)
    plt.tight_layout()
    plt.show()



def bar_plot(
    feature: str,
    df,
    ax=None,
    rotate_xticks=False,
    show_annotations=True,
    label_fontsize=7,
    title=None,
    n_categories=None,
    figsize=(6,3),
    bar_width=0.45
):
    """
    Plots a bar chart of the feature distribution by the target variable
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    counts = (
        df
        .groupby([feature, "isFraud"])
        .size()
        .unstack(fill_value=0)
    )

    counts["total_cat"] = counts.sum(axis=1)

    counts = counts.sort_values("total_cat", ascending=False)

    if n_categories is not None:
        counts = counts.head(n_categories)

    x = np.arange(len(counts))
    total = len(df)

    bars_nf = ax.bar(
        x - bar_width/2,
        counts[0],
        bar_width,
        color="gray",
        label="Not Fraud"
    )

    bars_f = ax.bar(
        x + bar_width/2,
        counts[1],
        bar_width,
        color="orange",
        label="Fraud"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        counts.index.astype(str),
        rotation=90 if rotate_xticks else 0
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Count")

    if title is None:
        title = f"Fraud vs Not Fraud by {feature}"

    ax.set_title(title)
    ax.legend()

    if show_annotations:
        for i, (b_nf, b_f) in enumerate(zip(bars_nf, bars_f)):

            count_nf = int(counts.iloc[i][0])
            count_f = int(counts.iloc[i][1])
            total_cat = counts.iloc[i]["total_cat"]

            if count_nf > 0:
                ax.text(
                    b_nf.get_x() + b_nf.get_width()/2,
                    count_nf,
                    f"{count_nf}\n{count_nf/total:.1%} total\n{count_nf/total_cat:.1%} cat",
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize
                )

            if count_f > 0:
                ax.text(
                    b_f.get_x() + b_f.get_width()/2,
                    count_f,
                    f"{count_f}\n{count_f/total:.1%} total\n{count_f/total_cat:.1%} cat",
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize
                )

    ax.margins(y=0.2)

    


def scatter_plot(
    result: pd.DataFrame,
    feature: str,
    ax=None,
    n_labels: int = 5,
    label_mode: str = "random",
    figsize=(5,4)
) -> None:
    
    """
    Plots a scatter plot of the feature distribution by the target variable
    Designed for high cardinality features
    y-axis: fraud percentage
    x-axis: fraud count
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        data=result,
        x="fraud_count",
        y="fraud_pct",
        ax=ax
    )

    if label_mode == "x":
        selected = result.nlargest(n_labels, "fraud_count")

    elif label_mode == "y":
        selected = result.nlargest(n_labels, "fraud_pct")

    elif label_mode == "random":
        selected = result.sample(min(n_labels, len(result)))

    else:
        raise ValueError("label_mode must be 'x', 'y', or 'random'")

    for _, row in selected.iterrows():
        ax.text(
            row["fraud_count"],
            row["fraud_pct"],
            str(row[feature]),
            fontsize=8,
            ha="left",
            va="bottom"
        )

    ax.set_xlabel("Fraud Count")
    ax.set_ylabel("Fraud Percentage")
    ax.set_title(feature)



def fraud_rate_percentile_bar(feature, df, n_bins=8, ax=None):
    """
    Plots a bar chart of the feature PERCENTILE BINSdistribution by the target variable
    Designed for numerical features
    y-axis: fraud rate
    x-axis: percentilebin
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))

    x = df[feature]

    # qcut on non-NaN values
    bins = pd.qcut(x.dropna(), q=n_bins, duplicates="drop")

    # create full bin column
    bin_col = pd.Series(index=x.index, dtype="object")
    bin_col.loc[x.notna()] = bins.astype(str)
    bin_col.loc[x.isna()] = "NaN"

    summary = (
        df.assign(bin=bin_col)
        .groupby("bin", observed=False)
        .agg(
            fraud_rate=("isFraud","mean"),
            count=("isFraud","size")
        )
        .reset_index()
    )

    bars = ax.bar(range(len(summary)), summary["fraud_rate"])

    ax.set_title(feature)
    ax.set_xlabel("Bin")
    ax.set_ylabel("Fraud rate")

    # annotate bars
    for i, bar in enumerate(bars):

        bin_label = summary["bin"].iloc[i]
        count = summary["count"].iloc[i]
        rate = summary["fraud_rate"].iloc[i]

        label = f"{bin_label}\nn={count}\n{rate:.3f}"

        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height()/2,
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
            color="black"
        )

    return ax

def plot_fraud_rate_by_amount_bucket(df, 
                                     feature ='TransactionAmt', 
                                     amount_limit = 7000, 
                                     bucket_size=100, 
                                     max_amt=None):
    
    """
    Plots a fraud rate against the amount bucket
    The amount bucket size can be changed by the user
    """

    data = df[df[feature]<=amount_limit][[feature, "isFraud"]].dropna()

    if max_amt is not None:
        data = data[data[feature] <= max_amt]

    # FAST bucket calculation
    data = data.assign(bucket=(data[feature] // bucket_size).astype(int))

    grouped = (
        data
        .groupby("bucket", observed=True)
        .agg(
            total=("isFraud", "count"),
            fraud=("isFraud", "sum")
        )
    )

    grouped["fraud_rate"] = grouped["fraud"] / grouped["total"]

    x = grouped.index.values

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(x, grouped["fraud_rate"], marker="o")

    # labels
    for i, row in grouped.iterrows():
        ax.text(
            i,
            row["fraud_rate"],
            str(int(row["total"])),
            fontsize=8,
            ha="center",
            va="bottom"
        )

    # nicer x labels
    labels = [f"{int(i*bucket_size)}-{int((i+1)*bucket_size)}" for i in x]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)

    ax.set_xlabel(f"{feature} Bucket ($)")
    ax.set_ylabel("Fraud Rate")
    ax.set_title(f"Fraud Rate by {feature} Bucket")

    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()







def plot_correlation_matrix(df, features, figsize=(10,8)):
    """
    Plots a correlation heatmap for numerical features
    """
    corr = df[features].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,     # show correlation numbers
        fmt=".2f",
        square=True,
        annot_kws={"size": 8} 
    )

    plt.title("Correlation Matrix")
    plt.show()



def categorical_heatmap(df, feature1, feature2):

    """
    Plots a cross-tabulation heatmap for two numerical features which are converted into categorical fetures.
    This is a special case when numerical features have a high share of NaN values.
    NaN values are then treated as a separate category and numerical values are binned into n percentile bins
    It can only hanle two features at a time
    """

    train_df_temp = df.copy()

    train_df_temp["feature1_bin"] = pd.qcut(train_df_temp[feature1], q=10)
    train_df_temp["feature2_bin"] = pd.qcut(train_df_temp[feature2], q=10, duplicates="drop")

    train_df_temp["feature1_bin"] = (
        train_df_temp["feature1_bin"]
        .astype("object")
        .fillna("NaN")
    )

    train_df_temp["feature2_bin"] = (
        train_df_temp["feature2_bin"]
        .astype("object")
        .fillna("NaN")
    )


    table = pd.crosstab(train_df_temp["feature1_bin"], train_df_temp["feature2_bin"], normalize="index")

    plt.figure(figsize=(6,4))
    sns.heatmap(table, annot=True, cmap="viridis", fmt=".2f")
    plt.title(f"{feature1} vs {feature2}")
    plt.ylabel(feature1)
    plt.xlabel(feature2)
    plt.show()

    del train_df_temp
