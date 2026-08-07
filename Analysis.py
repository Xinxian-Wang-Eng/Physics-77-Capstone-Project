# Calculates
# MSD
# effective diffusivity
# mass conservation
# radial concentration profile
# Brownian-versus-continuum error
# analytical-versus-numerical error

import numpy as np


def fit_diffusion_coefficient_from_msd(
    times,
    msd,
    start_fraction=0.10,
    end_fraction=0.70
):
    """
    Estimate the diffusion coefficient from the
    linear portion of MSD versus time.

    For 2D diffusion:

        MSD = 4 D t

    therefore:

        D = slope / 4
    """

    times = np.asarray(
        times,
        dtype=float
    )

    msd = np.asarray(
        msd,
        dtype=float
    )

    if len(times) != len(msd):
        raise ValueError(
            "times and msd must have the same length."
        )

    start_index = int(
        len(times)
        * start_fraction
    )

    end_index = int(
        len(times)
        * end_fraction
    )

    fit_times = times[
        start_index:end_index
    ]

    fit_msd = msd[
        start_index:end_index
    ]

    slope, intercept = np.polyfit(
        fit_times,
        fit_msd,
        1
    )

    diffusion_coefficient = (
        slope / 4
    )

    return (
        diffusion_coefficient,
        slope,
        intercept
    )




def calculate_rmse(
    numerical,
    reference
):
    """
    Calculate root-mean-square error between two arrays.
    """

    numerical = np.asarray(numerical,dtype=float)
    reference = np.asarray(reference,dtype=float)

    if numerical.shape != reference.shape:
        raise ValueError(
            "numerical and reference arrays "
            "must have the same shape."
        )

    error = (numerical- reference)

    rmse = np.sqrt(np.mean(error**2))

    return rmse


def calculate_grid_size(physical_length,dx):
    """
    Calculate the number of grid points needed to keep
    approximately the same physical domain length.

    The +1 includes both endpoints.
    """

    grid_size = int(round(physical_length / dx)) + 1

    return grid_size


def continuum_convergence_test(
    dx_values,
    physical_length,
    total_time,
    diffusion_coefficient,
    total_mass,
    sigma,
    stability_ratio=0.1
):
    """
    Test continuum-solver convergence at several grid resolutions.

    For each dx:
        dt = stability_ratio * dx^2 / D

    The numerical solution is compared against the analytical
    Gaussian solution at the final time.

    Returns
    -------
    results : list of dict
        One dictionary for each tested resolution.
    """

    from src.Continuum_Walking_2d import (simulate_continuum_2d)
    from src.Analytical_2d import (analytical_gaussian)

    results = []

    for dx in dx_values:

        # Choose dt so stability ratio stays constant

        dt = (stability_ratio* dx**2/ diffusion_coefficient)

        # Keep physical domain size fixed

        width = calculate_grid_size(physical_length,dx)
        height = width

        source_x = width // 2
        source_y = height // 2

        # Number of steps needed

        n_steps = int(round(total_time / dt))
        actual_total_time = (n_steps * dt)

        # Run numerical continuum model

        (frames,times,mass_history,minimum_history) = simulate_continuum_2d(
            n_steps=n_steps,
            width=width,
            height=height,
            dx=dx,
            dt=dt,
            diffusion_coefficient=diffusion_coefficient,
            total_mass=total_mass,
            sigma=sigma,
            source_x=source_x,
            source_y=source_y,
            save_every=n_steps
        )

        numerical_final = (frames[-1])

        # Analytical solution at same final time

        analytical_final = (
            analytical_gaussian(
                t=actual_total_time,
                width=width,
                height=height,
                dx=dx,
                diffusion_coefficient=
                    diffusion_coefficient,
                total_mass=total_mass,
                sigma=sigma,
                source_x=source_x,
                source_y=source_y
            )
        )

        # Error
        rmse = calculate_rmse(numerical_final,analytical_final)

        result = {
            "dx": dx,
            "dt": dt,
            "width": width,
            "height": height,
            "n_steps": n_steps,
            "total_time":actual_total_time,
            "stability_ratio":stability_ratio,
            "rmse":rmse,
            "initial_mass":mass_history[0],
            "final_mass":mass_history[-1],
            "minimum_concentration":np.min(minimum_history)
        }

        results.append(result)

    return results