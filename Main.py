"""
main.py

Main entry point for the Physics 77 diffusion project.

Simulation parameters are controlled through config.py.

Available models:
    analytical
    brownian
    continuum
"""

import Config

from src.visualization import (
    plot_result_frame,
    show_heatmap_slider,
    plot_msd,
    plot_mass_history
)


# ============================================================
# RUN SELECTED MODEL
# ============================================================

def run_selected_model():
    """
    Run the model selected in config.MODEL.

    Returns
    -------
    result : dict
        Standard simulation result dictionary.
    """

    if Config.MODEL == "analytical":
        from src.Analytical_2d import (run_from_config)


    elif Config.MODEL == "brownian":
        from src.Brownian_2d import (run_from_config)


    elif Config.MODEL == "continuum walking":
        from src.Continuum_Walking_2d import (run_from_config)

    else:
        raise ValueError("Unknown model: "+ str(Config.MODEL))

    result = run_from_config()

    return result

def print_result_summary(result):
    """
    Print important information about one simulation.
    """
    print("Physics 77 Diffusion Simulation")
    print("===============================")
    print("Model:",result["model"])
    print("Number of frames:",len(result["frames"]))
    print("Time range:",result["times"][0],"to",result["times"][-1])
    print("Grid spacing:",result["l"])
    print("Diffusion coefficient:",result["diffusion_coefficient"])
    print("Total mass:",result["total_mass"])


    # Mass
    if "mass" in result:
        print("Initial estimated mass:",result["mass"][0])
        print("Final estimated mass:",result["mass"][-1])


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

def main():
    """
    Run one complete diffusion experiment.
    """

    # Validate settings first
    Config.validate_config()
    # Run simulation
    result = run_selected_model()
    # Print numerical summary
    print_result_summary(result)
    # Display figures
    show_visualizations(result)


if __name__ == "__main__":
    main()