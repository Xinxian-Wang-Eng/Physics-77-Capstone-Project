"""
src/Analytical_2d.py

Analytical solution for free 2D diffusion from a point source.

This module is used as the reference solution for validating the Brownian and continuum numerical models.

Assumptions:
1. Infinite, homogeneous 2D medium
2. Constant diffusion coefficient
3. No porous obstacles
4. No external concentration gradient, force, or flow
5. Single Point-source initial condition

Array convention: concentration[y, x]

The analytical solution is:

    C(x, y, t) = M / (4*pi*D*t) * exp(-((x - x0)^2 + (y - y0)^2) / (4*D*t))
"""

import numpy as np
import Config

# 1. GRID CREATION
def Grid_Creator(
    width=101,
    height=101,
    dx=1.0,
    source_x=None,
    source_y=None
):
    """
    Create coordinate arrays measured relative to the source position.

    Parameters
    width: int, Number of grid cells in the x direction.
    height: int, Number of grid cells in the y direction.
    l: float, Grid cell length.
    source_x: int or None, x-coordinate of the source, None places it at the horizontal center.
    source_y: int or None, y-coordinate of the source, None places it at the vertical center.

    Returns: 
    X: 2D array, x-coordinate array with shape (height, width).
    Y: 2D array, y-coordinate array with shape (height, width).
    x: 1D array, x-coordinate array.
    y: 1D array, y-coordinate array.

    source_x: int, Final source x-index.

    source_y: int, Final source y-index.
    """

    if width < 1 or height < 1:
        raise ValueError("Width and height must be positive.")
    if dx <= 0:
        raise ValueError("Grid length must be positive.")
    if source_x is None:
        source_x = width // 2
    if source_y is None:
        source_y = height // 2
    if source_x < 0 or abs(source_x) >= width:
        raise ValueError("source_x is outside the grid.")
    if source_y < 0 or abs(source_y) >= height:
        raise ValueError("source_y is outside the grid.")

    # Coordinates are measured relative to the source, ensuring that the grid centers at the source
    x = (np.arange(width) - source_x) * dx
    y = (np.arange(height) - source_y) * dx

    X, Y = np.meshgrid(x, y)

    return X, Y, x, y, source_x, source_y


# 2. ANALYTICAL POINT-SOURCE SOLUTION

def analytical_point_source(
    t,
    width=101,
    height=101,
    dx=1.0,
    diffusion_coefficient=1.0,
    total_mass=1.0,
    source_x=None,
    source_y=None
):
    """
    Calculate the analytical 2D diffusion field at one time.

    Parameters
    t: float, Physical time.
    width: int, Number of grid cells in the x direction.
    height: int, Number of grid cells in the y direction.
    l: float, Grid cell length.
    diffusion_coefficient: float, Diffusion coefficient (D).
    total_mass: float, Total Number of Particles (M).
    source_x: int or None, x-coordinate of the source, None places it at the horizontal center.
    source_y: int or None, y-coordinate of the source, None places it at the vertical center.

    Returns
    concentration: numpy.ndarray, Analytical concentration field with shape (height, width) (Cxy for all coordinates on the grid).
    x: 1D array, x-coordinate array.
    y: 1D array, y-coordinate array.
    """

    if t < 0:
        raise ValueError("The analytical point-source formula requires t > 0. ")
    # if t == 0:
    #     raise ValueError("At t = 0, the ideal point source is a delta function.")
    if diffusion_coefficient < 0:
        raise ValueError("Diffusion_coefficient must be positive.")
    if total_mass < 0:
        raise ValueError("Total Particle Number cannot be negative.")

    X, Y, x, y, source_x, source_y = Grid_Creator(
        width=width,
        height=height,
        dx=dx,
        source_x=source_x,
        source_y=source_y
    )

    distance_squared = X**2 + Y**2

    denominator = 4 * np.pi * diffusion_coefficient * t
    exponent = (-distance_squared / (4 * diffusion_coefficient * t))

    concentration = (total_mass / denominator * np.exp(exponent))

    return concentration, x, y


# 3. ORIGIN CONCENTRATION

def origin_concentration(
    t,
    diffusion_coefficient=1.0,
    total_mass=1.0
):
    """
    Calculate the concentration at the source location.

    For a point source:

        C(x0, y0, t) = M / (4*pi*D*t)

    Parameters
    t : float, Physical time.

    diffusion_coefficient : float
        Diffusion coefficient D.

    total_mass : float
        Total mass M.

    Returns
    concentration_at_origin : float
        Concentration at the source position.
    """

    if t <= 0:
        raise ValueError("t must be positive.")
    if diffusion_coefficient <= 0:
        raise ValueError("Diffusion Coefficient must be positive.")
    if total_mass < 0:
        raise ValueError("Total Particle Number cannot be negative.")
    #Concentration at origin is calculated here (formula simplified since  x-x0 and y-y0 both equal 0) to use as C0 to calculate concentration at next t
    concentration_at_origin = (total_mass / (4 * np.pi * diffusion_coefficient * t))

    return concentration_at_origin


# 4. MULTIPLE ANALYTICAL FRAMES

def analytical_diffusion_frames(
    run_time,
    time_interval,
    width=101,
    height=101,
    dx=1.0,
    diffusion_coefficient=1.0,
    total_mass=1.0,
    source_x=None,
    source_y=None
):
    """
    Generate analytical concentration fields at each t iteration.
    run_time: int, Physical time for the diffusion process
    time_interval: float, time interval


    Returns
    frames: numpy array, A collection of "screenshots" at each t in times, has shape [t, y, x]

    times : numpy array, A collection of t
    """

    if run_time <= 0:
        raise ValueError("Diffusion time must be positive")

    times = np.arange(0, run_time, time_interval)
    frames = []

    for t in times:

        concentration, x, y = analytical_point_source(
            t=t,
            width=width,
            height=height,
            dx=dx,
            diffusion_coefficient=diffusion_coefficient,
            total_mass=total_mass,
            source_x=source_x,
            source_y=source_y
        )
        frames.append(concentration)

    frames = np.array(frames)

    return frames, times




# 5. CENTERLINE PROFILE

def centerline_profile(
    concentration,
    direction="horizontal",
    source_x=None,
    source_y=None
):
    """
    Extract a concentration profile through the source position.

    Parameters
    ----------
    concentration : numpy.ndarray
        A 2D concentration field.

    direction : str
        "horizontal" or "vertical".

    source_x : int or None
        Source x-index. None uses the center.

    source_y : int or None
        Source y-index. None uses the center.

    Returns
    -------
    profile : numpy.ndarray
        One-dimensional concentration profile.
    """

    if concentration.ndim != 2:
        raise ValueError("concentration must be a 2D array.")

    height, width = concentration.shape

    if source_x is None:
        source_x = width // 2

    if source_y is None:
        source_y = height // 2

    if direction == "horizontal":
        profile = concentration[source_y, :]

    elif direction == "vertical":
        profile = concentration[:, source_x]

    else:
        raise ValueError(
            "direction must be 'horizontal' or 'vertical'."
        )

    return profile


# 6. TOTAL MASS

def calculate_total_mass(concentration,dx=1.0):
    """
    Estimate total mass in the 2D concentration field.

    The continuous integral is approximated by: M approximately equals sum(C) * l^2

    Parameters
    concentration : 2D Array, concentration field.

    l : float, Grid spacing.

    Returns
    mass : float
        Estimated total mass inside the finite grid.
    """

    if concentration.ndim != 2:
        raise ValueError("concentration must be a 2D array.")
    if dx <= 0:
        raise ValueError("l must be greater than zero.")

    mass = np.sum(concentration) * dx**2
    #This approximamtion idea based on M approximately equals to Cxy * dx * dy, which should be done through double rieman sum or surface integral
    #We migh do that ...
    return mass


# 7. ANALYTICAL MSD

def analytical_msd(run_time, time_interval, diffusion_coefficient=1.0):
    """
    Calculate theoretical mean squared displacement in 2D. MSD(t) = 4*D*t

    Parameters
    times : list or numpy.ndarray, Physical time values.

    diffusion_coefficient : float
        Diffusion coefficient D.

    Returns
    -------
    msd : numpy.ndarray
        Theoretical MSD values.
    """
    if run_time < 0:
        raise ValueError("Run time must be positive.")
    times = np.arange(0, run_time, time_interval)

    if diffusion_coefficient <= 0:
        raise ValueError(
            "diffusion_coefficient must be greater than zero."
        )

    msd = 4 * diffusion_coefficient * times

    return msd

def analytical_gaussian(
    t,
    width=101,
    height=101,
    dx=1.0,
    diffusion_coefficient=1.0,
    total_mass=1.0,
    sigma=3.0,
    source_x=None,
    source_y=None
):
    """
    Analytical evolution of an initially Gaussian
    concentration distribution.
    """

    if source_x is None:
        source_x = width // 2

    if source_y is None:
        source_y = height // 2

    x = (np.arange(width)- source_x) * dx
    y = (np.arange(height)- source_y) * dx

    X, Y = np.meshgrid(x,y)

    variance = (sigma**2+ 2* diffusion_coefficient* t)
    concentration = (total_mass/ (2* np.pi* variance)* np.exp(-(X**2+ Y**2)/ (2* variance)))

    return concentration

# 8. CONFIGURATION WRAPPER

def run_from_config():
    """
    Generate analytical frames using settings from config.py.

    This wrapper is convenient for main.py. The other functions remain
    independently reusable from notebooks.
    """


    Config.validate_config()

    source_x, source_y = Config.get_source_position()

    # Total physical simulation time
    total_time = Config.NUM_STEPS * Config.DT
    # Time interval between saved frames
    time_interval = Config.DT * Config.SAVE_EVERY


    frames, times = analytical_diffusion_frames(
        run_time=total_time,
        time_interval=time_interval,
        width=Config.GRID_WIDTH,
        height=Config.GRID_HEIGHT,
        dx=Config.DX,
        diffusion_coefficient=Config.DIFFUSION_COEFFICIENT,
        total_mass=Config.TOTAL_MASS,
        source_x=source_x,
        source_y=source_y
    )

    mass_history = []

    for frame in frames:
        mass = calculate_total_mass(concentration=frame,dx=Config.DX)
        mass_history.append(mass)
    mass_history = np.array(mass_history)

    msd = analytical_msd(
        run_time=total_time,
        time_interval=time_interval,
        diffusion_coefficient=Config.DIFFUSION_COEFFICIENT
    )

    result = {
        "model": "analytical",
        "times": times,
        "frames": frames,
        "msd": msd,
        "mass": mass_history,
        "source_position": (source_x, source_y),
        "diffusion_coefficient": Config.DIFFUSION_COEFFICIENT,
        "total_mass": Config.TOTAL_MASS,
        "dt": Config.DT,
        "dx": Config.DX
    }

    return result


# 9. MODULE TEST

if __name__ == "__main__":

    result = run_from_config()

    print("Analytical 2D diffusion module test")
    print("-----------------------------------")
    print("Model:", result["model"])
    print("Number of frames:", len(result["frames"]))
    print("Frame shape:", result["frames"].shape)
    print("Source position:", result["source_position"])
    print("First time:", result["times"][0])
    print("Final time:", result["times"][-1])
    print("First estimated mass:", result["mass"][0])
    print("Final estimated mass:", result["mass"][-1])
    print("Final theoretical MSD:", result["msd"][-1])
