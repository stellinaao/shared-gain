import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

""" ACROSS SESSIONS """


# plot data distro as a raincloud
def plot_raincloud(data, label="", ax=None, log=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(2.5, 2.5), tight_layout=True)

    cloud_color = "#666666"
    rain_color = "#246193"

    if log:
        data = np.log10(data)
        label = f"log₁₀({label})"

    # half violin (KDE)
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 300)
    y = kde(x)
    y_norm = y / y.max() * 0.3  # scale height

    offset = 0.5
    ax.fill_between(
        x, offset, offset + y_norm, alpha=0.5, color=cloud_color
    )  # 0.6 is to offset to the top of figure
    ax.plot(x, offset + y_norm, color=cloud_color, linewidth=0.5)

    # jittered strip plot
    jitter = np.random.uniform(-0.125, 0.125, size=len(data))
    ax.scatter(data, 0.3 + jitter, s=2, alpha=0.4, color=rain_color)

    # boxplot
    ax.boxplot(
        data,
        vert=False,
        positions=[0.1],
        widths=[0.08],
        patch_artist=True,
        boxprops=dict(facecolor=rain_color, alpha=0.85),
        medianprops=dict(color="k", linewidth=1.5),
        whiskerprops=dict(color="k"),
        capprops=dict(color="k"),
        flierprops=dict(marker=".", color=rain_color, markersize=2, alpha=0.4),
    )

    ax.axvline(x=0, color="#666666", linestyle="--", linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    return ax


# plot data distro as boxplots across sessions
def plot_boxplots_sess(data_sess, sess_ids, data_label):
    fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)

    ax.boxplot(
        data_sess,
        widths=0.5,
        patch_artist=True,
        flierprops=dict(
            marker="o",
            markersize=1.5,
            alpha=0.4,
            markeredgewidth=0,
            markerfacecolor="steelblue",
        ),
        medianprops=dict(color="k", linewidth=1),
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    ax.axhline(y=0, color="k", linewidth=0.5, linestyle="--", zorder=0)

    ax.set_xlabel("Session")
    ax.set_ylabel(data_label)
    ax.set_xticks(np.arange(len(sess_ids)) + 1)
    ax.set_xticklabels(sess_ids, rotation=45, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


# plot data distro as ridges across sessions (from sns galler)
def plot_ridges_sess(data_sess, key, sess_ids, data_label):
    saved_params = plt.rcParams.copy()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # initialize the facetgrid object
    palette = sns.cubehelix_palette(len(sess_ids), rot=-0.25, light=0.7)
    grid = sns.FacetGrid(
        data_sess, row="key", hue="key", aspect=15, height=0.5, palette=palette
    )

    # draw the densities
    grid.map(
        sns.kdeplot,
        key,
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    grid.map(sns.kdeplot, key, clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    grid.refline(y=0, linewidth=1.2, linestyle="-", color=None, clip_on=False)

    # add a line at 0
    grid.map(plt.axvline, x=0, color="#ffffff", linewidth=2, linestyle="--")

    # define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    grid.map(label, key)

    # set the subplots to overlap
    grid.figure.subplots_adjust(hspace=-0.25)

    # remove axes details that don't play well with overlap
    grid.set_titles("")
    grid.set(yticks=[], ylabel="")
    grid.despine(bottom=True, left=True)

    # update axis label
    grid.set_axis_labels(x_var=data_label, y_var="")

    plt.rcParams.update(saved_params)
