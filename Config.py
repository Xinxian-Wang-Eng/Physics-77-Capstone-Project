"""
Config.py

Central settings for the 2D diffusion project.

Change values in this file to adjust the simulation.
Other modules should import these settings instead of redefining them.
"""

# 1. MODEL SELECTION

# Options:
# "analytical"
# "brownian"
# "continuum walking"
MODEL = "continuum walking"


# 2. CANVAS SETTINGS

GRID_WIDTH = 101
GRID_HEIGHT = 101

# Physical distance between neighboring grid points
l = 1.0


# 3. TIME SETTINGS

NUM_STEPS = 200
DT = 0.1

# Save one frame every SAVE_EVERY simulation steps
SAVE_EVERY = 1


# 4. DIFFUSION CEFFICIENT

DIFFUSION_COEFFICIENT = 1.0

TOTAL_MASS = 1.0
# 5. PORE GEOMETRY AND POROSITY

# Options:
# "free"
# "random"
# "channel"
# "layered"

GEOMETRY_TYPE = "free"

# Fraction of the grid that is open space
# 1.0 means no obstacles
POROSITY = 1.0

CHANNEL_WIDTH = 21
LAYER_SPACING = 15


# 6. INITIAL CONDITION

# Options:
# "point"
# "gaussian"
# "uniform"
# "gradient"

SOURCE_TYPE = "point"

# For the analytical model:
# total released mass
#
# For the continuum model:
# initial total concentration or source amount
#
# For the Brownian model:
# usually related to number of particles



# Use None to place the source at the center
SOURCE_X = None
SOURCE_Y = None

# Width of the Gaussian source
GAUSSIAN_SIGMA = 3.0



# ============================================================
# 7. BROWNIAN MODEL SETTINGS
# ============================================================

NUM_PARTICLES = 10000
RANDOM_SEED = 1
# Lattice movement distance per time step
# PARTICLE_STEP_SIZE = 1


# ============================================================
# 8. BOUNDARY CONDITIONS
# ============================================================

# Options:
# "reflecting"
# "absorbing"
# "periodic"
# "fixed"
OUTER_BOUNDARY = "reflecting"

# Solid obstacles will use no-flux behavior
OBSTACLE_BOUNDARY = "no_flux"


# ============================================================
# 10. ANALYSIS SETTINGS
# ============================================================

CALCULATE_MSD = True
CALCULATE_MASS_HISTORY = True
CALCULATE_EFFECTIVE_DIFFUSIVITY = True

# Fractions of the MSD data used for fitting D_eff
FIT_START_FRACTION = 0.10
FIT_END_FRACTION = 0.70


# ============================================================
# 11. VISUALIZATION SETTINGS
# ============================================================

# Options:
# "static"
# "slider"
# "animation"
VISUALIZATION_TYPE = "slider"


SHOW_HEATMAP = True
SHOW_MSD_PLOT = True
SHOW_MASS_PLOT = True

# Values below this are treated as transparent
TRANSPARENT_THRESHOLD = 1e-12

# Keep one color scale for all frames
FIXED_COLOR_SCALE = True


# ============================================================
# 12. OUTPUT SETTINGS
# ============================================================

SAVE_RESULTS = False
SAVE_FIGURES = False

RESULTS_DIRECTORY = "results"
FIGURES_DIRECTORY = "figures"


# 13. HELPER FUNCTIONS

def get_source_position():
    """
    Return the source position as (x, y).

    If SOURCE_X or SOURCE_Y is None, the center is used.
    """

    if SOURCE_X is None:
        x = GRID_WIDTH // 2
    else:
        x = SOURCE_X

    if SOURCE_Y is None:
        y = GRID_HEIGHT // 2
    else:
        y = SOURCE_Y

    return x, y


def get_total_time():
    """
    Return the total simulated time.
    """

    return NUM_STEPS * DT


def get_stability_ratio():
    """
    Return the stability ratio for the explicit 2D continuum solver.

    r = D * DT / DX^2
    """

    return DIFFUSION_COEFFICIENT * DT / l**2


def get_number_of_saved_frames():
    """
    Return the approximate number of saved frames.
    """

    return NUM_STEPS // SAVE_EVERY + 1


def validate_config():
    """
    Check whether the current settings are valid.

    This function raises a ValueError when it finds a problem.
    """

    allowed_models = [
        "analytical",
        "brownian",
        "continuum walking"
    ]

    allowed_geometries = [
        "free",
        "random",
        "channel",
        "layered"
    ]

    allowed_sources = [
        "point",
        "gaussian",
        "uniform",
        "gradient"
    ]

    allowed_boundaries = [
        "reflecting",
        "absorbing",
        "periodic",
        "fixed"
    ]

    allowed_visualizations = [
        "static",
        "slider",
        "animation"
    ]

    if MODEL not in allowed_models:
        raise ValueError(
            "MODEL must be analytical, brownian, or continuum."
        )

    if GEOMETRY_TYPE not in allowed_geometries:
        raise ValueError(
            "GEOMETRY_TYPE must be free, random, channel, or layered."
        )

    if SOURCE_TYPE not in allowed_sources:
        raise ValueError(
            "SOURCE_TYPE must be point, gaussian, uniform, or gradient."
        )

    if OUTER_BOUNDARY not in allowed_boundaries:
        raise ValueError(
            "OUTER_BOUNDARY is not recognized."
        )

    # if VISUALIZATION_TYPE not in allowed_visualizations:
    #     raise ValueError(
    #         "VISUALIZATION_TYPE must be static, slider, or animation."
    #     )

    if GRID_WIDTH < 3 or GRID_HEIGHT < 3:
        raise ValueError(
            "GRID_WIDTH and GRID_HEIGHT must be at least 3."
        )

    if l <= 0:
        raise ValueError("DX must be greater than zero.")

    if DT <= 0:
        raise ValueError("DT must be greater than zero.")

    if NUM_STEPS < 1:
        raise ValueError("NUM_STEPS must be at least 1.")

    if SAVE_EVERY < 1:
        raise ValueError("SAVE_EVERY must be at least 1.")

    if DIFFUSION_COEFFICIENT <= 0:
        raise ValueError(
            "DIFFUSION_COEFFICIENT must be greater than zero."
        )

    if POROSITY <= 0 or POROSITY > 1:
        raise ValueError(
            "POROSITY must be greater than 0 and no greater than 1."
        )

    if TOTAL_MASS < 0:
        raise ValueError(
            "SOURCE_STRENGTH cannot be negative."
        )

    if GAUSSIAN_SIGMA <= 0:
        raise ValueError(
            "GAUSSIAN_SIGMA must be greater than zero."
        )

    if NUM_PARTICLES < 1:
        raise ValueError(
            "NUM_PARTICLES must be at least 1."
        )



    # if PARTICLE_STEP_SIZE < 1:
    #     raise ValueError(
    #         "PARTICLE_STEP_SIZE must be at least 1."
    #     )

    source_x, source_y = get_source_position()

    if source_x < 0 or source_x >= GRID_WIDTH:
        raise ValueError(
            "SOURCE_X is outside the grid."
        )

    if source_y < 0 or source_y >= GRID_HEIGHT:
        raise ValueError(
            "SOURCE_Y is outside the grid."
        )

    if FIT_START_FRACTION < 0:
        raise ValueError(
            "FIT_START_FRACTION cannot be negative."
        )

    if FIT_END_FRACTION > 1:
        raise ValueError(
            "FIT_END_FRACTION cannot exceed 1."
        )

    if FIT_START_FRACTION >= FIT_END_FRACTION:
        raise ValueError(
            "FIT_START_FRACTION must be smaller than FIT_END_FRACTION."
        )

    if TRANSPARENT_THRESHOLD < 0:
        raise ValueError(
            "TRANSPARENT_THRESHOLD cannot be negative."
        )

    stability_ratio = get_stability_ratio()

    if MODEL == "continuum" and stability_ratio > 0.25:
        raise ValueError(
            "The continuum solver may be unstable.\n"
            "The required condition is:\n"
            "D * DT / DX^2 <= 0.25\n"
            "Current value: "
            + str(stability_ratio)
        )


# ============================================================
# 14. TEST CONFIGURATION
# ============================================================

if __name__ == "__main__":

    validate_config()

    print("Configuration is valid.")
    print("Model:", MODEL)
    print("Grid:", GRID_WIDTH, "x", GRID_HEIGHT)
    print("Source position:", get_source_position())
    print("Total time:", get_total_time())
    print("Stability ratio:", get_stability_ratio())
    print("Saved frames:", get_number_of_saved_frames())