import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




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



def bar_plot_one_feature(feature, df, rotate_xticks=False, label_fontsize=6):

    stats = (
        df
        .groupby([feature, "isFraud"])
        .size()
        .unstack(fill_value=0)
    )

    stats["total_cat"] = stats.sum(axis=1)

    x = range(len(stats))
    width = 0.4
    total = len(df)

    fig, ax = plt.subplots(figsize=(7,3))

    bars_nf = ax.bar(
        [i - width/2 for i in x],
        stats[0],
        width,
        color="gray",
        label="Not Fraud"
    )

    bars_f = ax.bar(
        [i + width/2 for i in x],
        stats[1],
        width,
        color="orange",
        label="Fraud"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(stats.index.astype(str), rotation=90 if rotate_xticks else 0)

    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.set_title(f"Fraud vs Not Fraud by {feature}")
    ax.legend()

    for i, (b_nf, b_f) in enumerate(zip(bars_nf, bars_f)):

        count_nf = int(stats.iloc[i][0])
        count_f = int(stats.iloc[i][1])
        total_cat = stats.iloc[i]["total_cat"]

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
    plt.tight_layout()




def bar_plot_many_features(
    feature: str,
    df,
    title: str,
    ax=None,
    rotate_xticks=False,
    show_annotations=True
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,3))

    counts = (
        df
        .groupby([feature, "isFraud"])
        .size()
        .unstack(fill_value=0)
    )

    x = range(len(counts))
    width = 0.4

    bars0 = ax.bar([i - width/2 for i in x], counts[0], width, color="gray", label="Not Fraud")
    bars1 = ax.bar([i + width/2 for i in x], counts[1], width, color="orange", label="Fraud")

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=90 if rotate_xticks else 0)

    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if show_annotations:
        total = len(df)
        for bars in [bars0, bars1]:
            for b in bars:
                count = int(b.get_height())
                if count == 0:
                    continue
                pct = 100 * count / total
                ax.text(
                    b.get_x() + b.get_width()/2,
                    count,
                    f"{count}\n({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=7
                )

    ax.margins(y=0.15)
    plt.tight_layout()


def group_by_feature(feature: str, df) -> None:
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


def scatter_plot(
    result: pd.DataFrame,
    feature: str,
    ax,
    n_labels: int = 5,
    label_mode: str = "random"
) -> None:

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
