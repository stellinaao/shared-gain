import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from utils.viz_utils import center_title

""" DISTRO RELATIONSHIP (2) """


# plot two data distros against each other as a scatter
def plot_scatter(
    x,
    y,
    xlabel="",
    ylabel="",
    title="",
    mn=None,
    mx=None,
    color=None,
    cmap="viridis",
    label=None,
    add_unity=False,
    add_lr=False,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2), tight_layout=True)

    r = pearsonr(x, y).statistic
    gr = np.mean(x < y)

    mn = 1.05 * min([np.min(x), np.min(y)]) if mn is None else mn
    mx = 1.05 * max([np.max(x), np.max(y)]) if mx is None else mx

    ax.scatter(x, y, s=0.5, cmap=cmap, c=color, alpha=0.5, label=label)

    if add_unity:
        ax.plot([mn, mx], [mn, mx], linewidth=0.5, linestyle="--", color="#666666")

    if add_lr:
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        m = lr.coef_[0]
        b = lr.intercept_

        ax.plot(
            [mn, mx],
            [m * mn + b, m * mx + b],
            linewidth=0.5,
            color="#BA3737",
            label=f"{m:.3f}x+{b:.3f}",
        )
        ax.legend(loc="upper right")

    ax.axhline(y=0, linewidth=0.5, color="k")
    ax.axvline(x=0, linewidth=0.5, color="k")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([mn, mx])
    ax.set_ylim([mn, mx])

    if title == "":
        title = f"r={r:.3f}, gr={gr:.3f}"
    else:
        title = f"{title} (r={r:.3f}), gr={gr:.3f}"
    ax.set_title(title)

    return ax


# plot two data distros against each other as a 2d histogram
def plot_hist2d(
    x,
    y,
    mn=None,
    mx=None,
    xlabel="",
    ylabel="",
    title="",
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.25, 2), tight_layout=True)
    else:
        fig = ax.figure

    r = pearsonr(x, y).statistic

    mn = 1.05 * min([np.min(x), np.min(y)]) if mn is None else mn
    mx = 1.05 * max([np.max(x), np.max(y)]) if mx is None else mx

    _, _, _, im = ax.hist2d(
        x, y, range=[[mn, mx], [mn, mx]], bins=50, cmap="Blues", density=True
    )
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title == "":
        title = f"r={r:.3f}"
    else:
        title = f"{title} (r={r:.3f})"
    ax.set_title(title)

    return ax


def plot_contour(
    x,
    y,
    mn=None,
    mx=None,
    fill=True,
    levels=20,
    thresh=0,
    bw_adjust=1.0,
    cmap="Blues",
    cbar=False,
    xlabel="",
    ylabel="",
    title="",
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(2.25, 2), tight_layout=True)

    r = pearsonr(x, y).statistic

    mn = 1.05 * min([np.min(x), np.min(y)]) if mn is None else mn
    mx = 1.05 * max([np.max(x), np.max(y)]) if mx is None else mx

    sns.kdeplot(
        x=x,
        y=y,
        fill=fill,
        levels=levels,
        thresh=thresh,
        bw_adjust=bw_adjust,
        cmap=cmap,
        clip=((mn, mx), (mn, mx)),
        cbar=cbar,
        cbar_kws=dict(fraction=0.05, pad=0.05) if cbar else None,
        ax=ax,
    )

    ax.set_xlim([mn, mx])
    ax.set_ylim([mn, mx])
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title == "":
        title = f"r={r:.3f}"
    else:
        title = f"{title} (r={r:.3f})"
    ax.set_title(title)

    return ax


# plot multiple pairs of distros against each other
def plot_2d_row(
    fn,
    data: dict,  # panel_label -> (x, y) tuple
    color: dict = None,
    xlabel="",
    ylabel="",
    sharey=True,
    sharex=True,
    panel_width=2.5,
    panel_height=3,
    title="",  # overall row title
    **kwargs,
):
    fig, axes = plt.subplots(
        ncols=len(data),
        nrows=1,
        figsize=(panel_width * len(data), panel_height),
        sharey=sharey,
        sharex=sharex,
        tight_layout=True,
    )
    if len(data) == 1:
        axes = [axes]

    for i, (panel_label, (x, y)) in enumerate(data.items()):
        if color is None:
            fn(
                x=x,
                y=y,
                xlabel=xlabel,
                ylabel=ylabel if i == 0 else "",
                title=panel_label,
                ax=axes[i],
                **kwargs,
            )
        else:
            fn(
                x=x,
                y=y,
                color=color[panel_label],
                xlabel=xlabel,
                ylabel=ylabel if i == 0 else "",
                title=panel_label,
                ax=axes[i],
                **kwargs,
            )

    if title:
        center_title(fig, axes, title)

    return fig, axes


# plot groups of two distros against each other on the same scatter plot
def plot_scatter_groups(
    data: dict,  # group_label -> (x, y) tuple
    xlabel="",
    ylabel="",
    title="",
    colors: dict = {},
    add_unity=False,
    add_lr=False,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2), tight_layout=True)

    x_all = np.concatenate([x for x, y in data.values()])
    y_all = np.concatenate([y for x, y in data.values()])

    r = pearsonr(x_all, y_all).statistic

    mn = 1.05 * min(x_all.min(), y_all.min())
    mx = 1.05 * max(x_all.max(), y_all.max())

    for group_label, (x, y) in data.items():
        ax.scatter(x, y, s=0.5, c=colors.get(group_label), alpha=0.5, label=group_label)

    if add_unity:
        ax.plot([mn, mx], [mn, mx], linewidth=0.5, linestyle="--", color="#666666")

    if add_lr:
        lr = LinearRegression().fit(x_all.reshape(-1, 1), y_all)
        m, b = lr.coef_[0], lr.intercept_
        ax.plot(
            [mn, mx],
            [m * mn + b, m * mx + b],
            linewidth=0.5,
            color="#BA3737",
            label=f"{m:.3f}x+{b:.3f}",
        )

    ax.axhline(y=0, linewidth=0.5, color="k")
    ax.axvline(x=0, linewidth=0.5, color="k")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    title = f"r={r:.3f}" if title == "" else f"{title} (r={r:.3f})"
    ax.set_title(title)

    ax.legend(fontsize=5, loc="upper left")

    return ax


""" TRAJECTORIES """


# plot a 2d trajectory with line color encoding time (LineCollection)
def plot_trajectory(
    x,
    y,
    mn=None,
    mx=None,
    xlabel="",
    ylabel="",
    title="",
    cmap="plasma",
    s=2,
    start_marker="*",
    end_marker="s",
    mid_marker=".",
    start_size_mult=10,
    end_size_mult=2,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2), tight_layout=True)

    t = np.arange(len(x))
    vmin, vmax = t.min(), t.max()

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, alpha=0.5, cmap=cmap, array=t[:-1], linewidth=0.5)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)

    # middle points
    if len(x) > 2:
        ax.scatter(
            x[1:-1],
            y[1:-1],
            c=t[1:-1],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=s,
            marker=mid_marker,
            zorder=3,
        )

    # start and end, drawn on top with distinct markers
    ax.scatter(
        x[0],
        y[0],
        c="#ffffff",
        s=s * start_size_mult,
        marker=start_marker,
        zorder=4,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.scatter(
        x[-1],
        y[-1],
        c="#ffffff",
        s=s * end_size_mult,
        marker=end_marker,
        zorder=4,
        edgecolors="k",
        linewidths=0.5,
    )

    # axes
    ax.axvline(x=0, color="k", linewidth=0.5, zorder=-2)
    ax.axhline(y=0, color="k", linewidth=0.5, zorder=-2)

    # lim
    if (mn is not None) and (mx is not None):
        ax.set_xlim([mn, mx])
        ax.set_ylim([mn, mx])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax


""" MULTIPLE DISTROS """


# plot multiple distros as kdes
def plot_kdes(
    data: dict,
    xlim=None,
    label="",
    ylabel=True,
    legend=True,
    cmap="tab10",
    add_means=True,
    line_kwargs: dict = {},
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(2.5, 2.5), tight_layout=True)

    from scipy.stats import gaussian_kde

    if xlim is not None:
        (mn, mx) = xlim
    else:
        mn = np.min([np.min(v) for v in data.values()])
        mx = np.max([np.max(v) for v in data.values()])
    x = np.linspace(mn, mx, 300)

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(data)))

    for (data_label, data_vals), default_color in zip(data.items(), colors):
        kde = gaussian_kde(data_vals)
        y = kde(x)

        style = {"color": default_color, "linewidth": 0.5}
        style.update(line_kwargs.get(data_label, {}))

        ax.plot(x, y, label=data_label, **style)

        if add_means:
            avg = np.mean(data_vals)
            ax.plot(
                avg,
                kde(avg),
                marker="v",
                color=style["color"],
                markersize=1,
                linestyle="none",
            )
            ax.axvline(x=avg, **style)
    ax.set_xlabel(label)
    if ylabel:
        ax.set_ylabel("density")
    if legend:
        ax.legend()

    return ax


# plot nested distros as a row of kdes
def plot_kde_row(
    data: dict,  # dict of dicts
    line_kwargs: dict = {},
    title: str = None,
    sharey=True,
    panel_width=1.5,
    panel_height=1.5,
    legend_panel=-1,
    legend_kwargs: dict = {},
    **kwargs,
):
    fig, axes = plt.subplots(
        ncols=len(data),
        nrows=1,
        figsize=(panel_width * len(data), panel_height),
        sharey=sharey,
        tight_layout=True,
    )
    if len(data) == 1:
        axes = [axes]  # plt.subplots returns a bare Axes, not an array, when ncols=1

    for i, (k, data_) in enumerate(data.items()):
        ax = plot_kdes(
            data_,
            line_kwargs=line_kwargs,
            ax=axes[i],
            ylabel=(i == 0),
            legend=False,
            **kwargs,
        )
        ax.set_title(k)
    # title
    if title is not None:
        center_title(fig, axes, title)

    # legend
    if legend_panel is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        small_legend = dict(
            fontsize=5,
            handlelength=1.0,
            labelspacing=0.3,
            borderaxespad=0.3,
            frameon=False,
        )
        small_legend.update(legend_kwargs)
        axes[legend_panel].legend(handles, labels, **small_legend)

    return fig, axes


""" DISTROS """


# plot data distro as a raincloud
def plot_raincloud(data, label="", xlim=None, ax=None, log=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(2.5, 2.5), tight_layout=True)

    cloud_color = "#666666"
    rain_color = "#246193"

    if log:
        data = np.log10(data)
        label = f"log₁₀({label})"

    # half violin (KDE)
    from scipy.stats import gaussian_kde

    if xlim is not None:
        x = np.linspace(xlim[0], xlim[1], 300)
    else:
        x = np.linspace(data.min(), data.max(), 300)

    kde = gaussian_kde(data)
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
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    return ax


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
