"""
continuum_2d.py

2D continuum diffusion using an explicit finite-difference method.

Array convention:
    concentration[y, x]
"""
import Config
import numpy as np
from src.Diffusion_rates import (
    calculate_hopping_rate,
    calculate_diffusion_ratio,
    validate_2d_diffusion_ratio,
    calculate_stay_probability_2d
)



def continuum_step(concentration, diffusion_coefficient, dt, dx):
    """
    Advance a 2D concentration field by one timestep.
    Uses the explicit finite-difference approximation to:
        dC/dt = D * (d2C/dx2 + d2C/dy2)

    Reflecting / no-flux outer boundaries are used.
    Parameters
    concentration : numpy.ndarray, Current 2D concentration field.
    diffusion_coefficient : float, Diffusion coefficient D.
    
    dt : float, Time step.
    dx : float, Grid spacing.

    Returns
    new_concentration : numpy.ndarray, Concentration field after one timestep.
    """

    concentration = np.asarray(concentration, dtype=float)

    if concentration.ndim != 2:
        raise ValueError("concentration must be a 2D array.")

    if diffusion_coefficient <= 0:
        raise ValueError("diffusion_coefficient must be positive.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    if dx <= 0:
        raise ValueError("dx must be positive.")

    r = calculate_diffusion_ratio(diffusion_coefficient, dt, dx)
    validate_2d_diffusion_ratio(r)

    # Reflecting / no-flux boundaries
    #This helps maintain dC/dn = 0 when crossing the boundary by creating a ghost grid cell outside of the bound
    #This is necessary as making boundary cells or outsider cells having C = 0 may cause "gradient", disrupting the algorithm and make all particles sucked there
    padded = np.pad(concentration, pad_width=1, mode="edge")

    #However, padding changes the size of the grid by adding a round of 0s on all sides, we need to re-phrase the original array
    #The center is just our original array!
    center = padded[1:-1, 1:-1]

    right = padded[1:-1, 2:]
    left = padded[1:-1, :-2]

    up = padded[2:, 1:-1]
    down = padded[:-2, 1:-1]

    # Explicit finite-difference update
    new_concentration = (center + r * (right+ left+ up+ down- 4 * center))

    return new_concentration

def initialize_gaussian(width, height, dx, total_mass,sigma,source_x, source_y):
    """
    Create a normalized 2D Gaussian concentration field.
    """

    x = (np.arange(width) - source_x) * dx
    y = (np.arange(height) - source_y) * dx

    X, Y = np.meshgrid(x, y)

    concentration = (total_mass / (2 * np.pi * sigma**2) * np.exp(-(X**2+ Y**2) / (2 * sigma**2)))

    return concentration

def initialize_point(width,height,source_x,source_y,total_mass,dx):
    """
    Initialize all mass at one grid cell.
    """

    concentration = np.zeros((height, width),dtype=float)
    concentration[source_y,source_x] = total_mass / dx**2

    return concentration

def calculate_continuum_msd(concentration,source_x,source_y,dx):
    """
    Calculate mean-square displacement from
    a 2D concentration field.
    """

    concentration = np.asarray(concentration,dtype=float)

    height, width = concentration.shape

    x = (np.arange(width)- source_x) * dx
    y = (np.arange(height)- source_y) * dx
    X, Y = np.meshgrid(x,y)

    squared_distance = (X**2 + Y**2)

    total_mass = (np.sum(concentration)* dx**2)

    if total_mass <= 0:
        raise ValueError("Total concentration mass must be positive.")
    msd = (np.sum(concentration* squared_distance)* dx**2/ total_mass)

    return msd

def simulate_continuum_2d(
    n_steps,
    width,
    height,
    dx,
    dt,
    diffusion_coefficient,
    total_mass,
    sigma,
    source_x,
    source_y,
    source_type,
    save_every=1
):
    """
    Simulate free 2D diffusion using an explicit
    finite-difference method.

    The simulation begins from a Gaussian concentration profile.

    Parameters
    n_steps : int, Number of numerical timesteps.
    width, height : intNumber of grid cells.
    dx : floatSpatial grid spacing.
    dt : floatNumerical timestep.
    diffusion_coefficient : floatDiffusion coefficient D.
    total_mass : floatTotal initial mass.
    sigma : floatWidth of the initial Gaussian distribution.
    source_x, source_y : intGrid indices of the Gaussian center.
    save_every : intSave one frame every this many timesteps.

    Returns
    frames : numpy.ndarray, Saved concentration fields, with shape [time_index, y, x].
    times : numpy.ndarray, Physical times corresponding to saved frames.
    mass_history : numpy.ndarray, Estimated total mass at each saved time.
    minimum_history : numpy.ndarray, Minimum concentration at each saved time.
    """
    # Initial concentration field
    if source_type == "point":
        concentration = initialize_point(width,height,source_x,source_y,total_mass,dx)
    elif source_type == "gaussian":
        concentration = initialize_gaussian(width,height,source_x,source_y,sigma,total_mass,dx)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    if save_every <= 0:
        raise ValueError("save_every must be positive.")


    # Storage
    frames = []
    times = []
    mass_history = []
    minimum_history = []
    msd_history = []

    # Save initial condition at t = 0
    frames.append(concentration.copy())
    times.append(0.0)
    mass_history.append(np.sum(concentration)* dx**2)
    minimum_history.append(np.min(concentration))
    initial_msd = calculate_continuum_msd(concentration,source_x,source_y,dx)
    msd_history.append(initial_msd)

    # Time evolution
    for step in range(1, n_steps + 1):
        concentration = continuum_step(
            concentration=concentration,
            diffusion_coefficient=diffusion_coefficient,
            dt=dt,
            dx=dx
        )

        # Save only selected steps
        if step % save_every == 0:
            current_time = (step * dt)
            current_mass = (np.sum(concentration)* dx**2)
            current_minimum = (np.min(concentration))
            frames.append(concentration.copy())
            times.append(current_time)
            mass_history.append(current_mass)
            minimum_history.append(current_minimum)
            current_msd = calculate_continuum_msd(concentration,source_x,source_y,dx)
            msd_history.append(current_msd)

    return (
        np.array(frames),
        np.array(times),
        np.array(msd_history),
        np.array(mass_history),
        np.array(minimum_history)
    )



def run_from_config():
    """
    Run the continuum diffusion simulation using config.py.
    """
    Config.validate_config()
    source_x, source_y = (Config.get_source_position())

    (frames,times,msd_history,mass_history,minimum_history) = simulate_continuum_2d(
        n_steps=Config.NUM_STEPS,
        width=Config.GRID_WIDTH,
        height=Config.GRID_HEIGHT,
        dx=Config.DX,
        dt=Config.DT,
        diffusion_coefficient=Config.DIFFUSION_COEFFICIENT,
        total_mass=Config.TOTAL_MASS,
        sigma=Config.GAUSSIAN_SIGMA,
        source_x=source_x,
        source_y=source_y,
        source_type=Config.SOURCE_TYPE,
        save_every=Config.SAVE_EVERY
    )

    hopping_rate = calculate_hopping_rate(Config.DIFFUSION_COEFFICIENT, Config.DX)

    r = calculate_diffusion_ratio(Config.DIFFUSION_COEFFICIENT, Config.DT, Config.DX)

    stay_probability = (calculate_stay_probability_2d(r))

    from src.Analysis import (fit_diffusion_coefficient_from_msd)
    measured_D, msd_slope, msd_intercept = (fit_diffusion_coefficient_from_msd(
        times,
        msd_history,
        start_fraction=Config.FIT_START_FRACTION,
        end_fraction=Config.FIT_END_FRACTION))


    from src.Analysis import (fit_diffusion_coefficient_from_msd,calculate_percent_error)
    percent_error = calculate_percent_error(measured_D,Config.DIFFUSION_COEFFICIENT)

    result = {
        "model": "continuum",
        "times": times,
        "frames": frames,
        "mass": mass_history,
        "msd": msd_history,
        "minimum_concentration":minimum_history,
        "source_position": (source_x,source_y),
        "diffusion_coefficient":Config.DIFFUSION_COEFFICIENT,
        "total_mass":Config.TOTAL_MASS,
        "sigma":Config.GAUSSIAN_SIGMA,
        "dx":Config.DX,
        "dt":Config.DT,
        "stability_ratio":r,
        "hopping_rate":hopping_rate,
        "diffusion_ratio":r,
        "stay_probability":stay_probability,
        "diffusion_coefficient":Config.DIFFUSION_COEFFICIENT,
        "measured_diffusion_coefficient":measured_D,
        "diffusion_percent_error":percent_error,
    }

    return result


