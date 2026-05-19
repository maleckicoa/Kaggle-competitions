import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency



def plot_binary_distribution(
    df,
    target="Credit risk",
    labels=("0", "1"),
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



def plot_bars_abs(
    df,
    feature: str,
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
        .groupby([feature, "Credit risk"])
        .size()
        .unstack(fill_value=0)
    )

    counts["total_cat"] = counts.sum(axis=1)

    counts = counts.sort_values("total_cat", ascending=False)

    if n_categories is not None:
        counts = counts.head(n_categories)

    x = np.arange(len(counts))
    total = len(df)

    total_good = counts[0].sum()
    total_bad = counts[1].sum()

    bars_nf = ax.bar(
        x - bar_width/2,
        counts[0],
        bar_width,
        color="gray",
        label="Good Loans"
    )

    bars_f = ax.bar(
        x + bar_width/2,
        counts[1],
        bar_width,
        color="orange",
        label="Bad Loans"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        counts.index.astype(str),
        rotation=90 if rotate_xticks else 0
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Count")

    if title is None:
        title = f"Good vs. Bad Loans by {feature}"

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
                    f"{count_nf}\n"
                    f"{count_nf/total:.1%} total\n"
                    f"{count_nf/total_good:.1%} good\n"
                    f"{count_nf/total_cat:.1%} categ.",
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize
                )

            if count_f > 0:
                ax.text(
                    b_f.get_x() + b_f.get_width()/2,
                    count_f,
                    f"{count_f}\n"
                    f"{count_f/total:.1%} total\n"
                    f"{count_f/total_bad:.1%} bad\n"
                    f"{count_f/total_cat:.1%} categ.",
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize
                )

    ax.margins(y=0.2)


def plot_bars_rel(
    df,
    feature='Credit amount',
    feature_type="numeric",
    amount_limit=10000,
    bucket_size=10,
    figsize=(12,5),
    variable_bar_width=True,
    hline=None,
    short_x_labels=False,
    max_categories=15
):
    """
    Bar plot of bad rate:
    - numeric → bucketed
    - categorical → per category
    """

    data = df[[feature, "Credit risk"]].dropna()

    # --- numeric ---
    if feature_type == "numeric":
        if amount_limit is not None:
            data = data[data[feature] <= amount_limit]

        data = data.assign(bucket=(data[feature] // bucket_size).astype(int))

    # --- categorical ---
    elif feature_type == "categorical":
        data = data.assign(bucket=data[feature].astype(str))

    else:
        raise ValueError("feature_type must be 'numeric' or 'categorical'")

    # --- aggregation ---
    grouped = (
        data
        .groupby("bucket", observed=True)
        .agg(
            total=("Credit risk", "count"),
            bad=("Credit risk", "sum")
        )
        .reset_index()
    )

    # --- limit categories ---
    if feature_type == "categorical":
        grouped = grouped.sort_values("total", ascending=False).head(max_categories)

    grouped["bad_rate"] = grouped["bad"] / grouped["total"]

    total_obs = grouped["total"].sum()
    total_bad = grouped["bad"].sum()

    x = np.arange(len(grouped))

    fig, ax = plt.subplots(figsize=figsize)

    # --- widths ---
    if variable_bar_width:
        widths = grouped["total"] / grouped["total"].max()
        widths = widths * 0.8
    else:
        widths = np.full(len(grouped), 0.6)

    bars = ax.bar(x, grouped["bad_rate"], width=widths, color="#f2e274")

    max_rate = grouped["bad_rate"].max()
    ax.set_ylim(0, max_rate * 1.25)

    # --- annotations (FIXED) ---
    for bar, row in zip(bars, grouped.itertuples(index=False)):

        total = row.total
        bad = row.bad
        bad_rate = row.bad_rate

        label = (
            f"n={int(total)}\n"
            f"{total/total_obs:.1%} of all\n"
            f"{(bad/total_bad if total_bad > 0 else 0):.1%} of all bad\n"
            f"{bad_rate:.1%} Bad rate"
        )

        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max_rate * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=6
        )

    # --- horizontal line ---
    if hline is not None:
        ax.axhline(y=hline, color="red", linestyle="--", linewidth=1, alpha=0.7)

    # --- x labels ---
    if feature_type == "numeric":
        if short_x_labels:
            labels = [f"{int(i*bucket_size)}" for i in grouped["bucket"]]
        else:
            labels = [
                f"{int(i*bucket_size)}-{int((i+1)*bucket_size)}"
                for i in grouped["bucket"]
            ]
    else:
        labels = grouped["bucket"].astype(str).tolist()

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)

    ax.set_title(f"Bad Loans Rate by {feature}", fontsize=10)
    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel("Bad Loans Rate", fontsize=9)

    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()



def plot_bad_loans_rate_heatmap(
    df,
    feature_x,
    feature_y,
    target="Credit risk",
    figsize=(10, 7),
    cmap="RdYlGn_r",
    min_count=None,
    ax=None,
    fontsize=8,
    min_bad_count_filter=None,
    min_bad_rate_filter=None,
    return_signal=False  # if True → just check, don't plot
):
    created_fig = False
    if ax is None and not return_signal:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    data = df[[feature_x, feature_y, target]].dropna().copy()

    grouped = (
        data
        .groupby([feature_y, feature_x], observed=True)
        .agg(
            bad_rate=(target, lambda x: (x == 1).mean()),
            bad_count=(target, lambda x: (x == 1).sum()),
            total_count=(target, "size")
        )
        .sort_index()
    )

    # FILTER SIGNAL LOGIC
    if min_bad_count_filter is not None and min_bad_rate_filter is not None:
        signal = (
            (grouped["bad_count"] >= min_bad_count_filter) &
            (grouped["bad_rate"] >= min_bad_rate_filter)
        ).any()
    else:
        signal = True  # no filtering → always plot

    if return_signal:
        return signal

    if not signal:
        return False

    # --- heatmap values ---
    rate_pivot = grouped["bad_rate"].unstack(feature_x)

    annot = pd.DataFrame("", index=rate_pivot.index, columns=rate_pivot.columns)

    for y_val in rate_pivot.index:
        for x_val in rate_pivot.columns:
            if (y_val, x_val) not in grouped.index:
                continue

            row = grouped.loc[(y_val, x_val)]
            rate = row["bad_rate"]
            bad = row["bad_count"]
            total = row["total_count"]

            if pd.isna(rate):
                annot.loc[y_val, x_val] = ""
            elif min_count is not None and total < min_count:
                annot.loc[y_val, x_val] = f"n<{min_count}"
            else:
                annot.loc[y_val, x_val] = f"{rate:.2%}\n{int(bad)} / {int(total)}"

    sns.heatmap(
        rate_pivot,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Bad Rate"} if created_fig else None,
        annot_kws={"size": fontsize},
        ax=ax
    )

    ax.set_title(f"{feature_y} vs {feature_x}", fontsize=fontsize)
    ax.set_xlabel(feature_x, fontsize=fontsize)
    ax.set_ylabel(feature_y, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    if created_fig:
        plt.tight_layout()
        plt.show()

    return True



def plot_cramers_v_heatmap(
    df,
    columns=None,
    figsize=(10, 8),
    cmap="coolwarm",
    annot=True,
    fontsize=7
):
    """
    Cramér's V heatmap for categorical features
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    if columns is None:
        columns = df.select_dtypes(include="object").columns

    mat = pd.DataFrame(index=columns, columns=columns, dtype=float)

    for c1 in columns:
        for c2 in columns:
            ct = pd.crosstab(df[c1], df[c2])

            if ct.shape[0] < 2 or ct.shape[1] < 2:
                mat.loc[c1, c2] = np.nan
                continue

            chi2 = chi2_contingency(ct)[0]
            n = ct.values.sum()
            r, k = ct.shape

            mat.loc[c1, c2] = np.sqrt(chi2 / (n * min(r - 1, k - 1)))
    
    mat = mat.round(2)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        mat,
        annot=annot,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot_kws={"size": fontsize}
    )

    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    plt.title("Cramér's V Heatmap", fontsize=fontsize + 2)
    plt.show()



def print_cv_results(cv_results):
    print("\nROC-AUC per fold:", np.round(cv_results["test_roc_auc"], 3))
    print(f"ROC-AUC: {cv_results['test_roc_auc'].mean():.3f} (+/- {cv_results['test_roc_auc'].std():.3f})")

    print("\nPR-AUC per fold:", np.round(cv_results["test_pr_auc"], 3))
    print(f"PR-AUC:  {cv_results['test_pr_auc'].mean():.3f} (+/- {cv_results['test_pr_auc'].std():.3f})")

    print("\nRecall per fold:", np.round(cv_results["test_recall"], 3))
    print(f"Recall:  {cv_results['test_recall'].mean():.3f} (+/- {cv_results['test_recall'].std():.3f})")

    print("\nPrecision per fold:", np.round(cv_results["test_precision"], 3))
    print(f"Precision:  {cv_results['test_precision'].mean():.3f} (+/- {cv_results['test_precision'].std():.3f})")

    