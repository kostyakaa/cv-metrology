"""Inspect a saved metric height grid in Matplotlib."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path, help="surfaces.npz or area.npz")
    parser.add_argument("--out", type=Path, help="Save a PNG instead of opening a window")
    args = parser.parse_args()

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.widgets import Button, CheckButtons

    with np.load(args.map, allow_pickle=False) as data:
        height = data["height_mm"]
        pitch = float(data["cell_mm"])
        origin = data["xy_origin"]
        confirmed = data["confirmed"] & np.isfinite(height)
        filled = data.get("filled", np.zeros_like(confirmed))
        model = data.get("model_mm", height)
    if height.ndim != 2 or min(height.shape) == 0 or not np.isfinite(pitch) or pitch <= 0:
        parser.error("Expected a non-empty XY height grid with positive cell spacing")
    x = origin[0] + pitch * np.arange(height.shape[1])
    y = origin[1] + pitch * np.arange(height.shape[0])
    xx, yy = np.meshgrid(x, y)
    uncertain = np.isfinite(height) & ~confirmed
    norm = Normalize(-15, 15)
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([.24, .17, .73, .76], projection="3d")
    observations = ax.scatter(xx[confirmed], yy[confirmed], height[confirmed],
                              c=height[confirmed], cmap="RdYlBu_r", norm=norm,
                              s=8, linewidths=0, depthshade=False)
    guesses = ax.scatter(xx[uncertain], yy[uncertain], height[uncertain], c="#8092a4",
                         s=4, alpha=.35, linewidths=0, depthshade=False, visible=False)
    completion = ax.scatter(xx[filled], yy[filled], model[filled], c="#733f87",
                            s=10, linewidths=0, depthshade=False)
    layers = {"Согласованные": observations, "Спорные / одиночные": guesses, "Достройка": completion}
    check = CheckButtons(fig.add_axes([.025,.53,.20,.18]), list(layers), [True,False,True])

    def toggle(label):
        artist = layers[label]
        artist.set_visible(not artist.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(toggle)
    buttons = []
    for i, (label, angles) in enumerate([("Объём", (37,67)), ("Сверху", (90,-90)), ("Сбоку", (0,-90))]):
        button = Button(fig.add_axes([.04,.43-i*.062,.16,.045]), label)
        button.on_clicked(lambda _, a=angles: (ax.view_init(*a), fig.canvas.draw_idle()))
        buttons.append(button)

    ax.set(xlabel="X, мм", ylabel="Y, мм", zlabel="Z, мм", title="Карта высот")
    ax.set_box_aspect((max(np.ptp(x),pitch),max(np.ptp(y),pitch),max(np.ptp(ax.get_zlim()),pitch)))
    ax.view_init(37,67)
    fig.colorbar(observations, cax=fig.add_axes([.34,.105,.49,.025]), orientation="horizontal",
                 extend="both", label="Z относительно стола, мм")
    if args.out:
        fig.savefig(args.out, dpi=160)
    else:
        plt.show()


if __name__ == "__main__":
    main()
