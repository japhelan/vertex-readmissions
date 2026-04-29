import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def set_theme():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="flare",
        rc={
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.dpi": 120,
        },
    )


def plot_formatting_helper(
    column_name,
    ax,
    title_suffix,
    y_label,
    rotate_x_threshold=None,
    n_items=None,
):
    """
    helper for formatting count-style plots (barplots, histograms with count stat, etc)
    with consistent titles, axis labels, grid, and optional x-axis rotation for long category names
    """
    pretty_name = column_name.replace("_", " ").title()
    ax.set_title(f"{pretty_name} {title_suffix}", pad=12, weight="bold")
    ax.set_xlabel(pretty_name)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)

    if (
        rotate_x_threshold is not None
        and n_items is not None
        and n_items > rotate_x_threshold
    ):
        ax.tick_params(axis="x", rotation=35)


def plot_histogram(
    df,
    column_name,
    bins="auto",
    stat="count",
    kde=False,
    show_bin_counts=False,
    kde_color="#0f4c81",
    ax=None,
):
    """
    make a histogram of a selected column (distribution)
    options for bins, stat type (count, density, etc), kde overlay, and showing bin counts
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column_name,
        bins=bins,
        stat=stat,
        kde=kde,
        line_kws={"linewidth": 2},
        color=sns.color_palette("flare")[2],
        edgecolor="white",
        linewidth=1,
        alpha=0.9,
        ax=ax,
    )

    # Force KDE line styling after plot creation for consistent behavior.
    if kde and ax.lines:
        ax.lines[-1].set_color(kde_color)
        ax.lines[-1].set_linewidth(2.2)

    y_label = "Count" if stat == "count" else stat.title()
    plot_formatting_helper(
        column_name=column_name,
        ax=ax,
        title_suffix="Distribution",
        y_label=y_label,
    )
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    if show_bin_counts:
        for patch in ax.patches:
            height = patch.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:,.0f}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    textcoords="offset points",
                    xytext=(0, 4),
                    fontsize=9,
                    color="#2f2f2f",
                )

    if created_fig is not None:
        created_fig.tight_layout()
        return created_fig, ax

    return ax.figure, ax


def plot_barplot(
    df,
    column_name,
    show_bin_counts=True,
    top_n=None,
    order="desc",
    palette="flare",
    single_color=None,
    ax=None,
):
    """
    make barplot of a selected column (value counts)
    can optionally show bin counts, order by count, and choose color scheme
    """
    plot_df = df.copy()

    # confirm column exists
    if column_name not in plot_df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe")

    # get value counts of column for graphing
    val_counts = (
        plot_df[column_name]
        .astype("string")
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis(column_name)
        .reset_index(name="count")
    )

    # optional order of value counts
    if order == "asc":
        val_counts = val_counts.sort_values("count", ascending=True)
    else:
        val_counts = val_counts.sort_values("count", ascending=False)

    if top_n is not None:
        val_counts = val_counts.head(top_n)

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=val_counts,
        x=column_name,
        y="count",
        color=single_color if single_color is not None else "#9f86c0",
        edgecolor="white",
        linewidth=1,
        ax=ax,
    )

    # Apply one color per bar from the chosen palette.
    if single_color is None:
        bar_colors = sns.color_palette(palette, n_colors=len(ax.patches))
        for patch, color in zip(ax.patches, bar_colors):
            patch.set_facecolor(color)

    # call helper to make the graph look nicer
    plot_formatting_helper(
        column_name=column_name,
        ax=ax,
        title_suffix="Counts",
        y_label="Count",
        rotate_x_threshold=8,
        n_items=len(val_counts),
    )

    # add bin counts if declared
    if show_bin_counts:
        for patch in ax.patches:
            height = patch.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:,.0f}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    textcoords="offset points",
                    xytext=(0, 4),
                    fontsize=9,
                    color="#2f2f2f",
                )

    if created_fig is not None:
        created_fig.tight_layout()
        return created_fig, ax

    return ax.figure, ax


def plot_grid(
    df,
    column_names,
    plot_func,
    n_plot_cols=2,
    sharex=False,
    sharey=False,
    figsize_scale=(8, 5),
    **plot_kwargs,
):
    """Generic grid renderer for any plotting function that accepts df, column_name, and ax."""
    n_plots = len(column_names)
    n_rows = (n_plots - 1) // n_plot_cols + 1

    fig, axes = plt.subplots(
        n_rows,
        n_plot_cols,
        figsize=(figsize_scale[0] * n_plot_cols, figsize_scale[1] * n_rows),
        sharex=sharex,
        sharey=sharey,
    )

    axes = np.array(axes).reshape(-1)

    for i, column_name in enumerate(column_names):
        plot_func(
            df=df,
            column_name=column_name,
            ax=axes[i],
            **plot_kwargs,
        )

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig, axes[:n_plots]


def plot_barplot_grid(
    df,
    column_names,
    show_bin_counts=True,
    top_n=None,
    order="desc",
    palette="flare",
    single_color=None,
    n_plot_cols=2,
    sharey=False,
):
    """Backwards-compatible wrapper around the generic plot_grid for barplots."""
    return plot_grid(
        df=df,
        column_names=column_names,
        plot_func=plot_barplot,
        n_plot_cols=n_plot_cols,
        sharey=sharey,
        show_bin_counts=show_bin_counts,
        top_n=top_n,
        order=order,
        palette=palette,
        single_color=single_color,
    )


def plot_numeric_x_numeric_grid(
    df,
    base_numeric_col,
    numeric_cols,
    plot_func="hexbin",
    n_plot_cols=2,
    figsize_scale=(8, 5),
    **plot_kwargs,
):
    """
    Generic grid renderer for plotting a base numeric column against multiple other numeric columns.

    Args:
        df: DataFrame
        base_numeric_col: The numeric column to compare against (e.g., "bmi")
        numeric_cols: List of numeric columns to plot against the base (e.g., ["age", "weight_kg", "length_of_stay"])
        plot_func: 'hexbin' (default), 'scatter', or custom plotting function
        n_plot_cols: Number of columns in the grid
        figsize_scale: Tuple to scale figure size (width per plot, height per plot)
        **plot_kwargs: Additional keyword arguments to pass to the plotting function
                      For hexbin: gridsize, cmap, etc.
                      For scatter: alpha, color, size, etc.
    """
    n_plots = len(numeric_cols)
    n_rows = (n_plots - 1) // n_plot_cols + 1

    fig, axes = plt.subplots(
        n_rows,
        n_plot_cols,
        figsize=(figsize_scale[0] * n_plot_cols, figsize_scale[1] * n_rows),
    )

    axes = np.array(axes).reshape(-1)

    for i, x_col in enumerate(numeric_cols):
        if plot_func == "hexbin":
            axes[i].hexbin(df[x_col], df[base_numeric_col], **plot_kwargs)
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(base_numeric_col)
        elif plot_func == "scatter":
            sns.scatterplot(
                data=df, x=x_col, y=base_numeric_col, ax=axes[i], **plot_kwargs
            )
        else:
            # Custom function - assume it accepts data, x, y, ax params
            plot_func(data=df, x=x_col, y=base_numeric_col, ax=axes[i], **plot_kwargs)

        axes[i].set_title(f"{x_col} vs {base_numeric_col}")

    # Remove empty subplots
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig, axes[:n_plots]


def plot_numeric_x_across_categories_grid(
    df,
    numeric_col,
    category_cols,
    plot_func=sns.boxplot,
    n_plot_cols=2,
    sharex=False,
    sharey=False,
    figsize_scale=(8, 5),
    **plot_kwargs,
):
    """
    Generic grid renderer for plotting a single numeric column across multiple categorical columns.

    Args:
        df: DataFrame
        numeric_col: The numeric column to plot (e.g., "bmi")
        category_cols: List of categorical columns to plot against
        plot_func: The plotting function to use (e.g., sns.boxplot, sns.violinplot)
        n_plot_cols: Number of columns in the grid
        sharex, sharey: Whether to share axes
        figsize_scale: Tuple to scale figure size (width per plot, height per plot)
        **plot_kwargs: Additional keyword arguments to pass to the plotting function
    """
    n_plots = len(category_cols)
    n_rows = (n_plots - 1) // n_plot_cols + 1

    fig, axes = plt.subplots(
        n_rows,
        n_plot_cols,
        figsize=(figsize_scale[0] * n_plot_cols, figsize_scale[1] * n_rows),
        sharex=sharex,
        sharey=sharey,
    )

    axes = np.array(axes).reshape(-1)

    for i, cat_col in enumerate(category_cols):
        plot_func(
            data=df,
            x=cat_col,
            y=numeric_col,
            ax=axes[i],
            **plot_kwargs,
        )
        axes[i].set_title(f"{numeric_col} by {cat_col}")

    # Remove empty subplots
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig, axes[:n_plots]


def plot_all_numeric_by_base_category_grid(
    df,
    base_cat_col="target",
    numeric_cols=None,
    plot_func=sns.boxplot,
    n_plot_cols=3,
    figsize_scale=(6, 4),
    exclude_id_like=True,
    **plot_kwargs,
):
    """
    Plot all numeric columns split by one base categorical column in a subplot grid.

    Args:
        df: DataFrame
        base_cat_col: Categorical column used on x-axis for every subplot
        numeric_cols: Optional list of numeric columns; if None, auto-detect numeric columns
        plot_func: Plot function (default sns.boxplot)
        n_plot_cols: Number of subplot columns
        figsize_scale: Width/height scale per subplot
        exclude_id_like: Exclude columns ending in '_id' during auto-detection
        **plot_kwargs: Extra args passed into plot_func
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c != base_cat_col]

    if exclude_id_like:
        numeric_cols = [c for c in numeric_cols if not c.lower().endswith("_id")]

    if not numeric_cols:
        raise ValueError("No numeric columns available to plot.")

    n_plots = len(numeric_cols)
    n_rows = (n_plots - 1) // n_plot_cols + 1

    fig, axes = plt.subplots(
        n_rows,
        n_plot_cols,
        figsize=(figsize_scale[0] * n_plot_cols, figsize_scale[1] * n_rows),
    )
    axes = np.array(axes).reshape(-1)

    for i, num_col in enumerate(numeric_cols):
        plot_func(
            data=df,
            x=base_cat_col,
            y=num_col,
            ax=axes[i],
            **plot_kwargs,
        )
        axes[i].set_title(f"{num_col} by {base_cat_col}")

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig, axes[:n_plots]


def plot_categorical_x_categorical_grid(
    df,
    base_cat_col,
    category_cols,
    plot_func=sns.countplot,
    n_plot_cols=2,
    sharex=False,
    sharey=False,
    figsize_scale=(8, 5),
    **plot_kwargs,
):
    """
    Generic grid renderer for plotting relationships between multiple categorical columns and a base column.

    Args:
        df: DataFrame
        base_cat_col: The base categorical column to use as hue for all plots (e.g., "target")
        category_cols: List of categorical columns to plot on x-axis (e.g., ["smoker", "has_diabetes", "exercise_frequency"])
        plot_func: The plotting function to use (e.g., sns.countplot)
        n_plot_cols: Number of columns in the grid
        sharex, sharey: Whether to share axes
        figsize_scale: Tuple to scale figure size (width per plot, height per plot)
        **plot_kwargs: Additional keyword arguments to pass to the plotting function
    """
    n_plots = len(category_cols)
    n_rows = (n_plots - 1) // n_plot_cols + 1

    fig, axes = plt.subplots(
        n_rows,
        n_plot_cols,
        figsize=(figsize_scale[0] * n_plot_cols, figsize_scale[1] * n_rows),
        sharex=sharex,
        sharey=sharey,
    )

    axes = np.array(axes).reshape(-1)

    for i, cat_col_x in enumerate(category_cols):
        plot_func(
            data=df,
            x=cat_col_x,
            hue=base_cat_col,
            ax=axes[i],
            **plot_kwargs,
        )
        axes[i].set_title(f"{cat_col_x} by {base_cat_col}")

    # Remove empty subplots
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig, axes[:n_plots]
