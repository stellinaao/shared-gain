import matplotlib.pyplot as plt


# center title
def center_title(fig, axes: list, title, fontsize=7):
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    x_center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.suptitle(title, x=x_center, fontsize=fontsize)


# save fig
def save_fig(fig, fpath, fname):
    fpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath / fname, dpi=300, bbox_inches="tight")

    plt.close(fig)
