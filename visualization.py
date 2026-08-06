import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_heatmap(C, x=None, y=None, title="2D Concentration Heatmap"):
    """
    Plot a 2D concentration field as a heatmap.

    Parameters
    ----------
    C : ndarray
        2D concentration field C[y, x].

    x : ndarray or None
        1D x-coordinate array.

    y : ndarray or None
        1D y-coordinate array.

    title : str
        Plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    if x is not None and y is not None:
        extent = [x.min(), x.max(), y.min(), y.max()]
    else:
        extent = None

    img = ax.imshow(
        C,
        origin="lower",
        extent=extent,
        interpolation="none"
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Concentration")

    ax.set_aspect("equal")

    plt.show()

    return fig, ax


def plot_centerline(x, C, title="Centerline Concentration Profile"):
    """
    Plot the horizontal centerline concentration profile.

    Parameters
    ----------
    x : ndarray
        x-coordinate array.

    C : ndarray
        2D concentration field.

    title : str
        Plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    center_y = C.shape[0] // 2
    profile = C[center_y, :]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x, profile)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Concentration")
    ax.grid(True)

    plt.show()

    return fig, ax


def show_heatmap_slider(frames, times=None, x=None, y=None, title="2D Diffusion"):
    """
    Display diffusion frames using a slider.

    Parameters
    ----------
    frames : ndarray
        Shape should be (num_frames, height, width).

    times : ndarray or None
        Time values corresponding to frames.

    x : ndarray or None
        x-coordinate array.

    y : ndarray or None
        y-coordinate array.

    title : str
        Base title.

    Returns
    -------
    fig, ax, slider
        Matplotlib figure, axis, and slider.
    """

    frames = np.asarray(frames)

    if frames.ndim != 3:
        raise ValueError("frames must have shape (num_frames, height, width).")

    num_frames = frames.shape[0]

    if times is None:
        times = np.arange(num_frames)
    else:
        times = np.asarray(times)

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplots_adjust(bottom=0.25)

    if x is not None and y is not None:
        extent = [x.min(), x.max(), y.min(), y.max()]
    else:
        extent = None

    img = ax.imshow(
        frames[0],
        origin="lower",
        extent=extent,
        interpolation="none",
        vmin=0,
        vmax=frames.max()
    )

    ax.set_title(f"{title} | t = {times[0]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Concentration")

    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.04])

    slider = Slider(
        ax=slider_ax,
        label="Frame",
        valmin=0,
        valmax=num_frames - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        frame_index = int(slider.val)
        img.set_data(frames[frame_index])
        ax.set_title(f"{title} | t = {times[frame_index]:.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    return fig, ax, slider