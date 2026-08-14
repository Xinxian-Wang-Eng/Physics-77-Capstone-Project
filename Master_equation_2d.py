"""
master_equation_2d.py

Deterministic 2D lattice diffusion using the Master Equation.

Instead of simulating individual random walkers, this model
directly evolves the probability P[y, x] of finding a particle
at each lattice site.

For unbiased diffusion:

    p_right = r
    p_left  = r
    p_up    = r
    p_down  = r

where:
    r = D * dt / dx^2
    from Diffusion_rates

The probability of staying at the current site is:
    p_stay = 1 - 4r

Therefore the Master Equation update is:

    P_new = (1 - 4r) * P+ r * (P_right + P_left + P_up + P_down)

This is mathematically equivalent to the explicit finite
difference form of the diffusion equation.
Computation methods and part of codes refer to:
https://www.rpgroup.caltech.edu/mbl_pboc/code/diffusion_master_equation.html
"""

import Config
import numpy as np
from src.Diffusion_rates import (
    calculate_hopping_rate,
    calculate_diffusion_ratio,
    validate_2d_diffusion_ratio,
    calculate_stay_probability_2d
)
from src.Analysis import (
    fit_diffusion_coefficient_from_msd
)


# INITIAL CONDITION

def initialize_probability(width,height,source_x,source_y):
    """
    Initialize a point-source probability distribution.

    Initially P[source_y, source_x] = 1 and all other sites have probability zero.

    The total probability is therefore equal to 1
    """

    probability = np.zeros((height, width), dtype=float)
    probability[source_y, source_x] = 1.0

    return probability


# ONE MASTER EQUATION STEP

def master_equation_step(probability, r):
    """
    Advance the probability field by one timestep.

    For an unbiased 2D four-neighbor lattice:
        P_new[i,j]=(1 - 4r) P[i,j] + r (P_right + P_left + P_up + P_down)

    Reflecting boundaries are represented using edge padding.
    """

    validate_2d_diffusion_ratio(r)

    stay_probability = (calculate_stay_probability_2d(r))

    # Add one layer around the grid. With mode="edge", it copies boundary values outward, giving the same reflecting / zero-flux treatment
    # Similar mechanism as in Continuum_walking_2d np.pad part
    # Currently used by the continuum model.
    
    padded = np.pad(probability, pad_width=1, mode="edge")

    # Neighboring probability fields
    right = padded[1:-1,2:]
    left = padded[1:-1,:-2]
    up = padded[:-2,1:-1]
    down = padded[2:,1:-1]

    # Master Equation
    new_probability = (stay_probability * probability + r*(right + left + up + down))

    return new_probability


# MSD
def calculate_probability_msd(probability,source_x,source_y,dx):
    """
    Calculate mean squared displacement from a
    probability distribution.

        MSD = sum(P * r_distance^2) / sum(P)

    For a normalized probability distribution:

        sum(P) = 1

    but division by total_probability is kept for
    numerical safety.
    """

    height, width = probability.shape
    x = (np.arange(width) - source_x) * dx

    y = (np.arange(height) - source_y) * dx

    X, Y = np.meshgrid(x,y)

    squared_distance = (X**2 + Y**2)

    total_probability = np.sum(probability)
    
    if total_probability <= 0:
        raise ValueError("Total probability must be positive.")

    msd = (np.sum(probability * squared_distance) / total_probability)

    return msd


# FULL SIMULATION

def simulate_master_equation_2d(
    width,
    height,
    dx,
    dt,
    diffusion_coefficient,
    total_mass,
    n_steps,
    source_x,
    source_y,
    save_every=1
):
    """
    Run the 2D Master Equation simulation.

    Returns
    pprobability_frames : numpy array, Saved probability fields.
    concentration_frames: numpy array, Saved concentration fields.
    times : numpy arrayPhysical time corresponding to each frame.

    msd_history : numpy array, Mean squared displacement at each saved time.
    probability_history : numpy array, Total probability at each saved time.
    """

    # Shared lattice diffusion parameter
    r = calculate_diffusion_ratio(diffusion_coefficient, dt, dx)
    validate_2d_diffusion_ratio(r)

    # Initial probability field
    probability = initialize_probability(width, height, source_x, source_y)

    probability_frames = []
    concentration_frames = []
    times = []
    msd_history = []
    probability_history = []
    mass_history = []

    # Save at t = 0
    probability_frames.append(probability.copy())
    concentration = (probability * total_mass / dx**2)
    concentration_frames.append(concentration.copy())

    times.append(0.0)
    msd_history.append(calculate_probability_msd(probability,source_x,source_y,dx))

    total_probability = np.sum(probability)
    probability_history.append(total_probability)

    current_mass = (np.sum(concentration)* dx**2)
    mass_history.append(current_mass)


    # Time evolution

    for step in range(1,n_steps + 1):

        probability = master_equation_step(probability,r)
        if step % save_every == 0:
            concentration = (probability * total_mass / dx**2)
            current_time = (step * dt)
            current_msd = (calculate_probability_msd(probability,source_x,source_y,dx))
            total_probability = np.sum(probability)
            current_mass = (np.sum(concentration) * dx**2)

            probability_frames.append(probability.copy())
            concentration_frames.append(concentration.copy())
            times.append(current_time)
            msd_history.append(current_msd)
            probability_history.append(total_probability)
            mass_history.append(current_mass)

    return (
        np.array(concentration_frames),
        np.array(probability_frames),
        np.array(times),
        np.array(msd_history),
        np.array(mass_history),
        np.array(probability_history)
    )

# RUN FROM CONFIG

def run_from_config():
    """
    Run the Master Equation model using parameters
    defined in config.py.

    Returns a result dictionary compatible with the
    rest of the project.
    """

    source_x, source_y = (Config.get_source_position())

    (concentration_frames, probability_frames, times, msd_history, mass_history, probability_history) = simulate_master_equation_2d(
    width=Config.GRID_WIDTH,
    height=Config.GRID_HEIGHT,
    dx=Config.DX,
    dt=Config.DT,
    diffusion_coefficient=Config.DIFFUSION_COEFFICIENT,
    total_mass=Config.TOTAL_MASS,
    n_steps=Config.NUM_STEPS,
    source_x=source_x,
    source_y=source_y,
    save_every=Config.SAVE_EVERY)

    # Measure D from MSD slope

    (measured_D, msd_slope, msd_intercept) = fit_diffusion_coefficient_from_msd(
        times,
        msd_history,
        start_fraction=(Config.FIT_START_FRACTION),
        end_fraction=(Config.FIT_END_FRACTION))

    diffusion_percent_error = (abs(measured_D - Config.DIFFUSION_COEFFICIENT) / Config.DIFFUSION_COEFFICIENT * 100)

    # Shared rate quantities

    hopping_rate = calculate_hopping_rate(Config.DIFFUSION_COEFFICIENT,Config.DX)

    r = calculate_diffusion_ratio(Config.DIFFUSION_COEFFICIENT,Config.DT,Config.DX)

    stay_probability = (calculate_stay_probability_2d(r))

    # Result dictionary

    result = {
        "model":"master",
        "times":times,
        "frames": concentration_frames,
        "msd": msd_history,
        
        # Here it represents total probability rather than physical mass.
        "mass":mass_history,

         # Probability normalization history.
        "probability": probability_history,
        "probability_frames":probability_frames,
        "total_mass": Config.TOTAL_MASS,
        "source_position": (source_x, source_y),
        "diffusion_coefficient": Config.DIFFUSION_COEFFICIENT,
        "measured_diffusion_coefficient": measured_D,
        "diffusion_percent_error": diffusion_percent_error,
        "msd_slope": msd_slope,
        "msd_intercept": msd_intercept,
        "hopping_rate": hopping_rate,
        "diffusion_ratio": r,
        "stay_probability": stay_probability,
        "dx":Config.DX,
        "dt":Config.DT
    }

    return result
