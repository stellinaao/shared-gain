# center title
def center_title(fig, axes: list, title):
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    x_center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.suptitle(title, x=x_center, fontsize=7)
