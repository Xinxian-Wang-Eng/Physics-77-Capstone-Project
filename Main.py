"""
main.py

Main entry point for the Physics 77 diffusion project.

Simulation parameters are controlled through config.py.

Available models:
    analytical
    master
    brownian
    continuum
"""

import Config

from src.visualization import (
    plot_result_frame,
    show_heatmap_slider,
    plot_msd,
    plot_mass_history,
    plot_comparison_msd,
    plot_comparison_rmse,)

from src.Analysis import (
    compare_model_results,
    analyze_particle_convergence,
    print_particle_convergence)

# RUN SELECTED MODEL

def run_selected_model():

    if Config.MODEL == "analytical":
        from src.Analytical_2d import (run_from_config)
    elif Config.MODEL == "master":
        from src.Master_equation_2d import (run_from_config)
    elif Config.MODEL == "brownian":
        from src.Brownian_2d import (run_from_config)
    elif Config.MODEL == "continuum":
        from src.Continuum_Walking_2d import (run_from_config)
    else:
        raise ValueError(f"Unknown model: {Config.MODEL}")
    return run_from_config()

def run_three_model_comparison():
    """
    Run Brownian, Master Equation, and continuum
    models using the same parameters from Config.py.

    The three result dictionaries are then sent to
    Analysis.py for numerical comparison.
    """

    # Current Master Equation model only supports a point-source initial condition.
    # Therefore a fair three-model comparison also requires Brownian and continuum to use point sources.

    if Config.SOURCE_TYPE != "point":
        raise ValueError("Three-model comparison currently requires SOURCE_TYPE = 'point'.")


    from src.Brownian_2d import (run_from_config as run_brownian)
    from src.Master_equation_2d import (run_from_config as run_master)
    from src.Continuum_Walking_2d import (run_from_config as run_continuum)

    print()
    print("=" * 55)
    print("RUNNING THREE-MODEL COMPARISON")
    print("=" * 55)

    # Brownian
    print("Running Brownian model...")

    brownian_result = (run_brownian())

    # Master Equation
    print("Running Master Equation model...")
    master_result = (run_master())

    # Continuum
    print("Running continuum model...")

    continuum_result = (run_continuum())

    # Analysis
    comparison = compare_model_results(brownian_result, master_result, continuum_result)


    return comparison

def get_selected_result_from_comparison(comparison):
    """
    Return one of the already-computed model results according to Config.MODEL.
    This allows the existing visualization system to continue using Config.MODEL without rerunning the simulation.
    """

    if Config.MODEL == "brownian":
        return comparison["brownian_result"]
    elif Config.MODEL == "master":
        return comparison["master_result"]
    elif Config.MODEL == "continuum":
        return comparison["continuum_result"]
    else:
        # Analytical is not part of the three-model comparison.
        return None

def print_result_summary(result):
    """
    Print important information about one simulation.
    """
    print("Physics 77 Diffusion Simulation")
    print("===============================")
    print("Model:",result["model"])
    print("Number of frames:",len(result["frames"]))
    print("Time range:",result["times"][0],"to",result["times"][-1])
    print("Grid spacing:",result["dx"])
    print("Diffusion coefficient:",result["diffusion_coefficient"])
    print("Total mass:", result["total_mass"])


    # Mass
    if "mass" in result:
        print("Initial estimated mass:",result["mass"][0])
        print("Final estimated mass:",result["mass"][-1])
    #Master_Equation:
    if result["model"] == "master":
        print("Master Equation Parameters")
        print("--------------------------")
        print("Hopping rate k:",f"{result['hopping_rate']:.4f}")
        print("Diffusion ratio r:",f"{result['diffusion_ratio']:.4f}")
        print("Stay probability:",f"{result['stay_probability']:.4f}")
        print("Initial total probability:",f"{result['probability'][0]:.6f}")
        print("Final total probability:",f"{result['probability'][-1]:.6f}")

    # Brownian-specific results
    if result["model"] == "brownian":
        print("Number of particles:",result["num_particles"])
        print("Random seed:",result["seed"])
        print("Move probability:",result["move_probability"])
        print("Measured diffusion coefficient:",result["measured_diffusion_coefficient"])


    # Continuum-specific results
    if result["model"] == "continuum":
        print("Stability ratio:",result["stability_ratio"])
        print("Minimum concentration:",result["minimum_concentration"].min())
    if "measured_diffusion_coefficient" in result:
        print("Diffusion Validation")
        print("--------------------")
        print("Input D:",result["diffusion_coefficient"])
        print("Measured D:",result["measured_diffusion_coefficient"])
        print("Percent error:",result["diffusion_percent_error"],"%"
    )

def show_visualizations(result):
    """
    Show visualizations selected in config.py.
    """
    # Concentration field
    if Config.SHOW_HEATMAP:

        if (Config.VISUALIZATION_TYPE== "slider"):
            show_heatmap_slider(result,threshold=Config.TRANSPARENT_THRESHOLD,fixed_color_scale=Config.FIXED_COLOR_SCALE)
        elif (Config.VISUALIZATION_TYPE== "static"):
            plot_result_frame(result,frame_index=(len(result["frames"]) - 1),threshold=Config.TRANSPARENT_THRESHOLD,fixed_color_scale=Config.FIXED_COLOR_SCALE)


    # MSD
    if (Config.SHOW_MSD_PLOT and "msd" in result):
        plot_msd(result)
    # Mass conservation
    if (Config.SHOW_MASS_PLOT and "mass" in result):
        plot_mass_history(result)


def run_particle_convergence():
    """
    Test Brownian convergence toward the Master Equation as the number of particles increases.
    """

    if Config.SOURCE_TYPE != "point":
        raise ValueError("Particle convergence currently requires SOURCE_TYPE = 'point'.")


    from src.Brownian_2d import (run_from_config as run_brownian)
    from src.Master_equation_2d import (run_from_config as run_master)

    print("====================================")
    print("RUNNING PARTICLE-NUMBER CONVERGENCE")
    print("====================================")


    # Run Master Equation once for reference 
    print("Running Master Equation reference...")
    master_result = run_master()
    print("Master Equation completed.")


    # Run Brownian for each particle count
    brownian_results = []


    # Store the original value so that we can restore Config.NUM_PARTICLES after the experiment.
    original_num_particles = Config.NUM_PARTICLES


    for num_particles in Config.PARTICLE_COUNTS:
        print(f"Running Brownian with "
              f"N = {num_particles}...")

        # Temporarily change the Brownian particle number.
        Config.NUM_PARTICLES = num_particles

        # Run Brownian using the modified Config value.
        brownian_result = run_brownian()
        brownian_results.append(brownian_result)
        print(f"N = {num_particles} completed.")


    # Restore the normal Config value after all
    # convergence simulations have finished.
    Config.NUM_PARTICLES = original_num_particles


    # Analyze convergence
    convergence = analyze_particle_convergence(Config.PARTICLE_COUNTS,brownian_results,master_result)
    print_particle_convergence(convergence)


    return convergence

def main():
    """
    Run one complete diffusion experiment.
    """

    # Validate settings first
    Config.validate_config()

    if Config.RUN_PARTICLE_CONVERGENCE:
        run_particle_convergence()
        return


    # Run simulation
    if Config.COMPARE_MODELS:
        # Run Brownian, Master Equation, and Continuum and calculate their comparison results.
        comparison = run_three_model_comparison()

        # After comparison, use the model selected by Config.MODEL as the normal "result".
        if Config.MODEL == "brownian":result = comparison["brownian_result"]
        elif Config.MODEL == "master":result = comparison["master_result"]
        elif Config.MODEL == "continuum":result = comparison["continuum_result"]
        else:
            raise ValueError("When COMPARE_MODELS is True, MODEL must be 'brownian', 'master', or 'continuum'.")

    else:
        # Normal mode: only run the model selected by Config.MODEL.
        result = run_selected_model()

    # Print numerical summary
    print_result_summary(result)
    # Display figures
    show_visualizations(result)
    if Config.COMPARE_MODELS:
        plot_comparison_msd(comparison)
        plot_comparison_rmse(comparison)
if __name__ == "__main__":
    main()
