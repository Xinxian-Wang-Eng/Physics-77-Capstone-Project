import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# def plot_heatmap(C, x=None, y=None, title="2D Concentration Heatmap"):
#     """
#     Plot a 2D concentration field as a heatmap.

#     Parameters
#     ----------
#     C : ndarray
#         2D concentration field C[y, x].

#     x : ndarray or None
#         1D x-coordinate array.

#     y : ndarray or None
#         1D y-coordinate array.

#     title : str
#         Plot title.

#     Returns
#     -------
#     fig, ax
#         Matplotlib figure and axis.
#     """

#     fig, ax = plt.subplots(figsize=(6, 5))

#     if x is not None and y is not None:
#         extent = [x.min(), x.max(), y.min(), y.max()]
#     else:
#         extent = None

#     img = ax.imshow(
#         C,
#         origin="lower",
#         extent=extent,
#         interpolation="none"
#     )

#     ax.set_title(title)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")

#     cbar = plt.colorbar(img, ax=ax)
#     cbar.set_label("Concentration")

#     ax.set_aspect("equal")

#     plt.show()

#     return fig, ax


# def plot_centerline(x, C, title="Centerline Concentration Profile"):
#     """
#     Plot the horizontal centerline concentration profile.

#     Parameters
#     ----------
#     x : ndarray
#         x-coordinate array.

#     C : ndarray
#         2D concentration field.

#     title : str
#         Plot title.

#     Returns
#     -------
#     fig, ax
#         Matplotlib figure and axis.
#     """

#     center_y = C.shape[0] // 2
#     profile = C[center_y, :]

#     fig, ax = plt.subplots(figsize=(7, 4))

#     ax.plot(x, profile)

#     ax.set_title(title)
#     ax.set_xlabel("x")
#     ax.set_ylabel("Concentration")
#     ax.grid(True)

#     plt.show()

#     return fig, ax

def get_concentration_colormap():
    """
    Create the standard concentration colormap.

    Masked concentration values are transparent.
    Higher concentration appears darker red.

    Returns
    cmap: Matplotlib colormap.
    """

    cmap = plt.cm.Reds.copy()
    #I learned about this code from AI overview in Google search

    # Masked values become completely transparent.
    cmap.set_bad(color=(1, 1, 1, 0))
    #This is to achieve the effect of having low to no conentration area be transparent by"masking" them, this color pick is also recommended by Google AI overview
    #(1, 1, 1, 0) with the last index alpha controling opacity, 0 being transparent
    return cmap

def mask_low_concentration(
    concentration,
    threshold=1e-12
):
    """
    Make zero or extremely small concentration values transparent.

    Parameters
    ----------
    concentration : numpy.ndarray
        2D concentration field.

    threshold : float
        Concentrations less than or equal to this value
        will be masked.

    Returns
    masked_concentration
        NumPy masked array.
    """

    concentration = np.asarray(concentration)
    masked_concentration = np.ma.masked_where(concentration <= threshold, concentration)
    #This section is leearned from Google Search AI overview
    #The key here is using np.ma, it masks specific elements that fit the condition listed in () from the array
    return masked_concentration

def plot_heatmap(
    concentration,
    l=1.0,
    title="2D Concentration",
    threshold=1e-12,
    vmax=None,
    show_colorbar=True
):
    """
    Plot one 2D concentration field.

    Parameters
    ----------
    concentration : numpy.ndarray
        2D concentration array with shape [y, x].

    dx : float
        Physical grid spacing.

    title : str
        Title of the plot.

    threshold : float
        Concentration values below this threshold are transparent.

    vmax : float or None
        Maximum concentration represented by the color scale.
        If None, the maximum of the current frame is used.

    show_colorbar : bool
        Whether to show the concentration colorbar.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    concentration = np.asarray(concentration)

    height, width = concentration.shape
    masked_concentration = mask_low_concentration(concentration, threshold=threshold)

    cmap = get_concentration_colormap()

    # Physical dimensions of the simulation grid
    x_max = (width - 1) * l
    y_max = (height - 1) * l

    if vmax is None:
        vmax = np.max(concentration)

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        masked_concentration,
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        extent=[0, x_max, 0, y_max]
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    if show_colorbar:

        colorbar = plt.colorbar(image, ax=ax)
        colorbar.set_label("Concentration")

    plt.show()

    return fig, ax

def plot_result_frame(
    result,
    frame_index=20,
    threshold=1e-12,
    fixed_color_scale=True
):
    """
    Plot one concentration frame from a simulation result dictionary.

    Parameters
    result : dict
        Standard simulation result dictionary.

    frame_index : int
        Index of the frame to display.

    threshold : float
        Concentrations below this value are transparent.

    fixed_color_scale : bool
        If True, use the maximum concentration from all frames.
        If False, use the maximum from the selected frame.

    Returns
    fig, ax
        Matplotlib figure and axes.
    """

    frame = result["frames"]
    times = result["times"]

    if frame_index < 0 or frame_index >= len(frame):
        raise ValueError("frame_index is outside the available frames.")
    
    concentration = frame[frame_index]
    time = times[frame_index]
    model = result["model"]
    l = result["l"]

    if fixed_color_scale:
        vmax = np.max(frame)
    else:
        vmax = np.max(concentration)

    title = (model.capitalize() + " 2D Diffusion" + " | t = " + str(round(time, 4)))

    return plot_heatmap(
        concentration=concentration,
        l=l,
        title=title,
        threshold=threshold,
        vmax=vmax
    )


def show_heatmap_slider(
    result,
    threshold=1e-12,
    fixed_color_scale=True
):
    """
    Display concentration frames with an interactive time slider.

    Parameters
    result : dict
        Standard simulation result dictionary.

    threshold : float
        Concentrations below this value are transparent.

    fixed_color_scale : bool
        If True, all frames use the same color scale.

    Returns
    fig, ax, slider
        Matplotlib objects.
    """

    frame = np.asarray(result["frames"])
    times = np.asarray(result["times"])
    model = result["model"]
    l = result["l"]

    if len(frame) != len(times):
        raise ValueError("Number of frames must match number of times.")

    num_frames = len(frame)

    height = frame.shape[1]
    width = frame.shape[2]

    x_max = (width - 1) * l
    y_max = (height - 1) * l

    cmap = get_concentration_colormap()

    # Color scale
    if fixed_color_scale:
        global_vmax = np.max(frame)

    else:
        global_vmax = np.max(frame[0])

    # First frame
    first_frame = mask_low_concentration(frame[0], threshold)
    fig, ax = plt.subplots(figsize=(7, 6))
    # Leave space for slider
    plt.subplots_adjust(bottom=0.3)

    image = ax.imshow(
        first_frame,
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=global_vmax,
        extent=[0, x_max, 0, y_max]
        #Extent here is an important setup used to correctly represent the physical positions on the canvas
        #Usually, on the graph, the interval will be like 0, 20, 40, 60, 80, 100, with the range set to 100 automatically
        #But we need to make it work with customimzable grid cell length like 0.1, 0.5, as well as customizable width/height
        #Thus we need to mark them out here
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(model.capitalize() + " 2D Diffusion" + " | t = " + str(round(times[0], 4)))

    colorbar = plt.colorbar(image, ax=ax)

    colorbar.set_label("Concentration")

    # Slider
    slider_axis = plt.axes([0.20, 0.08, 0.60, 0.04])
    slider = Slider(
        ax=slider_axis,
        label="Frame",
        valmin=0,
        valmax=num_frames - 1,
        valinit=0,
        valstep=1
    )

    # Update function

    def update(frame_number):
    #This is key to make the slider work, this function repeatedly "update" the graph as the slider being dragged
    #Each slider value would be paired to certain frame. 
    #As the slider being dragged, this function will be called and find the corresponding frame and pull it up to the graph
    #
        frame_index = int(slider.val)
        current_frame = frame[frame_index]
        #This step finds the corresponding frame from the frames collection

        masked_frame = mask_low_concentration(current_frame, threshold)
        #This redo the masking, for asthetic purpose...

        image.set_data(masked_frame)
        if not fixed_color_scale:
            image.set_clim(0, np.max(current_frame))
        #I learned this from Google Search overview as well, it regenrates the ax and basically update the image

        current_time = times[frame_index]

        ax.set_title(model.capitalize() + " 2D Diffusion" + " | t = " + str(round(current_time, 4)))

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    return fig, ax, slider


def plot_msd(result, show_theory = True):
    """
    Plot mean squared displacement versus physical time.

    Parameters
    ----------
    result : dict
        Standard simulation result dictionary.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    if "msd" not in result:
        raise ValueError("result does not contain MSD data.")

    times = result["times"]
    msd = result["msd"]

    model = result["model"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(times, msd, label=model.capitalize())
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Squared Displacement")
    ax.set_title("MSD vs Time")
    ax.grid(True)
    ax.legend()

    if show_theory:
        D = result["diffusion_coefficient"]
        theoretical_msd = (4 * D * times)

        ax.plot(times, theoretical_msd, "--", label="Theory: 4Dt")

    plt.show()

    return fig, ax


def plot_mass_history(result):
    """
    Plot estimated total mass versus physical time.

    This is mainly used to check mass conservation.
    """

    if "mass" not in result:
        raise ValueError("result does not contain mass data.")

    times = result["times"]
    mass = result["mass"]
    model = result["model"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(times, mass)
    ax.set_xlabel("Time")
    ax.set_ylabel("Estimated Total Mass")
    ax.set_title(model.capitalize() + " Mass Conservation")
    ax.grid(True)

    plt.show()

    return fig, ax


def plot_convergence(
    results
):
    """
    Plot continuum RMSE versus grid spacing.
    """

    dx_values = [
        result["dx"]
        for result in results
    ]

    rmse_values = [
        result["rmse"]
        for result in results
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        dx_values,
        rmse_values,
        marker="o"
    )

    ax.set_xlabel(
        "Grid spacing dx"
    )

    ax.set_ylabel(
        "RMSE"
    )

    ax.set_title(
        "Continuum Grid Convergence"
    )

    ax.grid(True)

    plt.show()

    return fig, ax