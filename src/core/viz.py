import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

""" ACROSS SESSIONS """


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
