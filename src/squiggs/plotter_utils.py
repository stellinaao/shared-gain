def separate_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    yti = ax.get_yticks()
    yti = yti[
        (yti >= ax.get_ylim()[0]) & (yti <= ax.get_ylim()[1] + 10**-3)
    ]  # Add a small value to cover for some very tiny added values
    ax.spines["left"].set_bounds([yti[0], yti[-1]])
    xti = ax.get_xticks()
    xti = xti[(xti >= ax.get_xlim()[0]) & (xti <= ax.get_xlim()[1] + 10**-3)]
    ax.spines["bottom"].set_bounds([xti[0], xti[-1]])
    return
