# Calculates:
#   MSD
#   effective diffusivity
#   radial concentration profile
#   Brownian-versus-continuum error
#   analytical-versus-numerical error
# Checks mass conservation

import numpy as np

def fit_diffusion_coefficient_from_msd(times,msd,start_fraction=0.10,end_fraction=0.70):
    """
    Estimate the diffusion coefficient from the linear portion of MSD versus time.
    For 2D diffusion:
        MSD = 4 D t
    therefore:
        D = slope / 4
    """

    times = np.asarray(times,dtype=float)
    msd = np.asarray(msd,dtype=float)

    if len(times) != len(msd):
        raise ValueError("times and msd must have the same length.")

    start_index = int(len(times)* start_fraction)
    end_index = int(len(times)* end_fraction)

    fit_times = times[start_index:end_index]
    fit_msd = msd[start_index:end_index]
    slope, intercept = np.polyfit(fit_times,fit_msd,1)
    diffusion_coefficient = (slope / 4)

    return (diffusion_coefficient,slope,intercept)




def calculate_rmse(numerical, reference):
    """
    Calculate root-mean-square error between two arrays.
    """

    numerical = np.asarray(numerical,dtype=float)
    reference = np.asarray(reference,dtype=float)

    if numerical.shape != reference.shape:
        raise ValueError("numerical and reference arrays must have the same shape.")

    error = (numerical- reference)
    rmse = np.sqrt(np.mean(error**2))

    return rmse


def calculate_grid_size(physical_length,dx):
    """
    Calculate the number of grid points needed to keep approximately the same physical domain length.
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
    The numerical solution is compared against the analytical Gaussian solution at the final time.

    Returns
    results : list of dict, One dictionary for each tested resolution.
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
            save_every=n_steps)

        numerical_final = (frames[-1])

        # Analytical solution at same final time

        analytical_final = (analytical_gaussian(
                t=actual_total_time,
                width=width,
                height=height,
                dx=dx,
                diffusion_coefficient=diffusion_coefficient,
                total_mass=total_mass,
                sigma=sigma,
                source_x=source_x,
                source_y=source_y)
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



def calculate_percent_error(measured,expected):
    return (abs(measured - expected)/ abs(expected)* 100)

# MODEL COMPARISON

def calculate_rmse(array_1, array_2):
    """
    Calculate root mean squared error between two arrays.
        RMSE = sqrt(mean((array_1 - array_2)^2))
    """

    array_1 = np.asarray(array_1)
    array_2 = np.asarray(array_2)

    if array_1.shape != array_2.shape:
        raise ValueError("Arrays must have the same shape for RMSE calculation.")

    rmse = np.sqrt(np.mean((array_1 - array_2)**2))

    return rmse


def calculate_frame_rmse(frames_1,frames_2):
    """
    Calculate concentration-field RMSE at every saved timestep.

    frames shape:
        [time, y, x]
    Returns:
        rmse_history[time]
    """

    frames_1 = np.asarray(frames_1)
    frames_2 = np.asarray(frames_2)

    if frames_1.shape != frames_2.shape:
        raise ValueError("Frame arrays must have the same shape.")

    rmse_history = []

    for frame_index in range(len(frames_1)):

        rmse = calculate_rmse(frames_1[frame_index],frames_2[frame_index])
        rmse_history.append(rmse)

    return np.array(rmse_history)


def validate_comparison_results(brownian_result,master_result,continuum_result):
    """
    Check that the three models can be fairly compared.

    They must have matching:
        - saved times
        - frame shapes
        - dx
        - dt
        - input diffusion coefficient
        - total mass
    """

    results = [brownian_result,master_result,continuum_result]
    names = ["Brownian","Master","Continuum"]
    #Creates the lists containing the results that we want to compare in a specifc order
    reference = results[0]

    for result, name in zip(results[1:],names[1:]):
    #zip() helps pairt the two lists properly into tuples, giving
    # (analytical_result, "Analytical")
    # (continuum_result, "Continuum")    

    # Since we need to compare each pair independently, and cannot guarantee actual equality as most of the elements are floats on the scale of e-10, e-12, etc
    # To make the codes keep moving, we use isclose and allclose insitead of array_equal
    # Idea suggested by GPT
        if not np.allclose(reference["times"],result["times"]):
            raise ValueError(f"{name} times do not match Brownian times.")
        if (reference["frames"].shape!= result["frames"].shape):
            raise ValueError(f"{name} frame shape does not match Brownian.")
        # Using allclose here as the time frames should be shared with given times and dt from Config

        if not np.isclose(reference["dx"],result["dx"]):
            raise ValueError(f"{name} dx does not match Brownian.")
        if not np.isclose(reference["dt"],result["dt"]):
            raise ValueError(f"{name} dt does not match Brownian.")
        if not np.isclose(reference["diffusion_coefficient"],result["diffusion_coefficient"]):
            raise ValueError(f"{name} diffusion coefficient does not match Brownian.")
        if not np.isclose(reference["total_mass"],result["total_mass"]):
            raise ValueError(f"{name} total mass does not match Brownian.")


def compare_model_results(brownian_result,master_result,continuum_result):
    """
    Compare Brownian, Master Equation, and continuum diffusion results.

    Returns a dictionary containing:
        concentration-field RMSE histories
        MSD RMSE values
        measured diffusion coefficients
        diffusion coefficient errors
        mass conservation errors

    Brownian is expected to contain statistical noise.

    Master Equation and continuum should agree very closely when they use the same initial condition, grid, timestep, and boundary treatment.
    """

    validate_comparison_results(brownian_result,master_result,continuum_result)

    # Concentration field comparison

    brownian_master_rmse = (calculate_frame_rmse(brownian_result["frames"],master_result["frames"]))
    brownian_continuum_rmse = (calculate_frame_rmse(brownian_result["frames"],continuum_result["frames"]))
    master_continuum_rmse = (calculate_frame_rmse(master_result["frames"],continuum_result["frames"]))

    # MSD comparison

    brownian_master_msd_rmse = (calculate_rmse(brownian_result["msd"],master_result["msd"]))
    brownian_continuum_msd_rmse = (calculate_rmse(brownian_result["msd"],continuum_result["msd"]))
    master_continuum_msd_rmse = (calculate_rmse(master_result["msd"],continuum_result["msd"]))

    # Mass conservation check

    total_mass = (brownian_result["total_mass"])

    brownian_mass_error = (np.max(np.abs(brownian_result["mass"] - total_mass)))
    master_mass_error = (np.max(np.abs(master_result["mass"] - total_mass)))
    continuum_mass_error = (np.max(np.abs(continuum_result["mass"] - total_mass)))

    # Result dictionary

    comparison = {

        "times": brownian_result["times"],

        # Concentration RMSE versus time
        "brownian_master_rmse": brownian_master_rmse,
        "brownian_continuum_rmse": brownian_continuum_rmse,
        "master_continuum_rmse": master_continuum_rmse,

        # MSD comparison
        "brownian_master_msd_rmse": brownian_master_msd_rmse,
        "brownian_continuum_msd_rmse": brownian_continuum_msd_rmse,
        "master_continuum_msd_rmse": master_continuum_msd_rmse,

        # Measured diffusion coefficients
        "brownian_measured_D": brownian_result["measured_diffusion_coefficient"],
        "master_measured_D": master_result["measured_diffusion_coefficient"],
        "continuum_measured_D": continuum_result["measured_diffusion_coefficient"],

        # Percent errors
        "brownian_D_error": brownian_result["diffusion_percent_error"],
        "master_D_error": master_result["diffusion_percent_error"],
        "continuum_D_error": continuum_result["diffusion_percent_error"],

        # Mass conservation
        "brownian_max_mass_error": brownian_mass_error,
        "master_max_mass_error": master_mass_error,
        "continuum_max_mass_error": continuum_mass_error,

        #Plotting
        "brownian_result": brownian_result,
        "master_result": master_result,
        "continuum_result": continuum_result
    }

    return comparison


def analyze_particle_convergence(particle_counts,brownian_results,master_result):
    """
    Compare Brownian simulations with different particle counts against one fixed Master Equation result.

    The comparison is made at the final saved physical time.

    For simple statistical sampling, the error is expected to decrease approximately as:
        error ~ 1 / sqrt(N)
    """

    if len(particle_counts) != len(brownian_results):
        raise ValueError("particle_counts and brownian_results must have the same length.")


    # Master reference
    reference_times = (master_result["times"])
    reference_frame = (master_result["frames"][-1])
    final_time = (reference_times[-1])


    convergence_results = []


    # Compare each Brownian simulation to Master
    for num_particles, brownian_result in zip(particle_counts,brownian_results):

        # Make sure Brownian and Master are being
        # compared at the same saved times.
        if not np.allclose(brownian_result["times"], reference_times):
            raise ValueError("Brownian and Master time arrays do not match.")

        # Concentration-field RMSE at the final time.
        final_rmse = calculate_rmse(brownian_result["frames"][-1],reference_frame)


        # Brownian effective diffusion coefficient.
        measured_D = (brownian_result["measured_diffusion_coefficient"])


        # Percent error relative to the input D.
        diffusion_error = (brownian_result["diffusion_percent_error"])

        # If RMSE really scales approximately as 1/sqrt(N), then RMSE * sqrt(N) should remain roughly constant.
        scaled_rmse = (final_rmse* np.sqrt(num_particles))


        convergence_results.append(
            {
                "num_particles":num_particles,
                "final_rmse":final_rmse,
                "scaled_rmse":scaled_rmse,
                "measured_D":measured_D,
                "diffusion_percent_error":diffusion_error
            }
        )


    return {
        "final_time":final_time,
        "master_result":master_result,
        "results":convergence_results
    }

def print_particle_convergence(convergence):
    """
    Print the particle-number convergence results.
    """
    print("BROWNIAN PARTICLE-NUMBER CONVERGENCE")
    print("===============================================================================================================")
    print(
        f"Comparison time: "
        f"{convergence['final_time']:.4f}"
    )
    print(
        f"{'Particles':>12}"
        f"{'RMSE':>16}"
        f"{'RMSE sqrt(N)':>18}"
        f"{'D_eff':>14}"
        f"{'D error (%)':>16}"
    )

    for result in convergence["results"]:
        print(
            f"{result['num_particles']:>12d}"
            f"{result['final_rmse']:>16.6e}"
            f"{result['scaled_rmse']:>18.6e}"
            f"{result['measured_D']:>14.6f}"
            f"{result['diffusion_percent_error']:>16.4f}"
        )

    print("===============================================================================================================")
