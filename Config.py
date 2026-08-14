"""
Config.py

Central settings for the 2D diffusion project.

Change values in this file to adjust the simulation.

Other modules should import these settings instead of redefining them.
"""

# 1. MODEL SELECTION: Selects the model for simulation

# Options:
# "analytical"
# "master"
# "brownian"
# "continuum"
MODEL = "brownian"
# Options:
# True: run Brownian, Master Equation, and Continuum using the same simulation parameters and compare them.
# If False: run only the model selected by MODEL.
COMPARE_MODELS = False
# Options:
# True: test how the Brownian model approaches the Master Equation as the number of particles increases, MODEL must be "brownian"!
# False: only run one simulation
RUN_PARTICLE_CONVERGENCE = False
PARTICLE_COUNTS = [1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]

# 2. CANVAS SETTINGS: Adjust the size and unit length of the coordinate system
#CANVAS SIZE
GRID_WIDTH = 101
GRID_HEIGHT = 101

# GRID CELL LENGTH: Physical distance between neighboring grid points
DX = 1.0


# 3. TIME SETTINGS

NUM_STEPS = 200
DT = 0.1

# Save one frame every SAVE_EVERY simulation steps
SAVE_EVERY = 1


# 4. DIFFUSION CEFFICIENT

DIFFUSION_COEFFICIENT = 2.0

TOTAL_MASS = 1.0


# 5. INITIAL CONDITION

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

# DIRECTIONAL BIAS
USE_BIAS = True

# Positive x -> right
# Negative x -> left
BIAS_X = 0.03

# Positive y -> up
# Negative y -> down
BIAS_Y = 0.0

# Use None to place the source at the center
SOURCE_X = None
SOURCE_Y = None

# Width of the Gaussian source
GAUSSIAN_SIGMA = 3.0

# 6. BROWNIAN MODEL SETTINGS

NUM_PARTICLES = 1000000
RANDOM_SEED = 1
# Lattice movement distance per time step
# PARTICLE_STEP_SIZE = 1


# 7. BOUNDARY CONDITIONS
# Options:
# "reflecting"
# "absorbing"
OUTER_BOUNDARY = "reflecting"

# Solid obstacles will use no-flux behavior
OBSTACLE_BOUNDARY = "no_flux"


# 9. ANALYSIS SETTINGS

CALCULATE_MSD = True
CALCULATE_MASS_HISTORY = True
CALCULATE_EFFECTIVE_DIFFUSIVITY = True

# Fractions of the MSD data used for fitting D_eff
FIT_START_FRACTION = 0.10
FIT_END_FRACTION = 0.70




# 10. VISUALIZATION SETTINGS

# Options:
# "slider"

VISUALIZATION_TYPE = "slider"


SHOW_HEATMAP = True
SHOW_MSD_PLOT = True
SHOW_MASS_PLOT = True

# Values below this are treated as transparent
TRANSPARENT_THRESHOLD = 1e-12

# Keep one color scale for all frames
FIXED_COLOR_SCALE = True


# 11. OUTPUT SETTINGS

SAVE_RESULTS = False
SAVE_FIGURES = False

RESULTS_DIRECTORY = "results"
FIGURES_DIRECTORY = "figures"


# 12. HELPER FUNCTIONS

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


def get_diffusion_ratio():
    """
    Return the dimensionless diffusion ratio:
        r = D * dt / dx^2
    """

    from src.Diffusion_rates import (calculate_diffusion_ratio)
    return calculate_diffusion_ratio(DIFFUSION_COEFFICIENT,DT,DX)

def get_stability_ratio():
    """
    Backward-compatible name for get_diffusion_ratio().

    For the explicit 2D continuum solver,
    this same diffusion ratio determines stability.
    """

    return get_diffusion_ratio()


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
        "master",  
        "brownian",
        "continuum"
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
        raise ValueError("MODEL must be analytical, brownian, or continuum.")

    if SOURCE_TYPE not in allowed_sources:
        raise ValueError("SOURCE_TYPE must be point, gaussian, uniform, or gradient.")

    if OUTER_BOUNDARY not in allowed_boundaries:
        raise ValueError("OUTER_BOUNDARY is not recognized.")

    if VISUALIZATION_TYPE not in allowed_visualizations:
        raise ValueError("VISUALIZATION_TYPE must be static, slider, or animation.")

    if GRID_WIDTH < 3 or GRID_HEIGHT < 3:
        raise ValueError("GRID_WIDTH and GRID_HEIGHT must be at least 3.")

    if DX <= 0:
        raise ValueError("DX must be positive.")

    if DT <= 0:
        raise ValueError("DT must be positive.")

    if NUM_STEPS < 1:
        raise ValueError("NUM_STEPS must be at least 1.")

    if type(SAVE_EVERY) != int:
        raise TypeError("SAVE_EVERY must be int.")

    if SAVE_EVERY < 1:
        raise ValueError("SAVE_EVERY must be at least 1.")

    if DIFFUSION_COEFFICIENT <= 0:
        raise ValueError("DIFFUSION_COEFFICIENT must be greater than zero.")

    if TOTAL_MASS < 0:
        raise ValueError("SOURCE_STRENGTH cannot be negative.")

    if GAUSSIAN_SIGMA <= 0:
        raise ValueError("GAUSSIAN_SIGMA must be greater than zero.")

    if NUM_PARTICLES < 1:
        raise ValueError("NUM_PARTICLES must be at least 1.")


    source_x, source_y = get_source_position()

    if source_x < 0 or source_x >= GRID_WIDTH:
        raise ValueError("SOURCE_X is outside the grid.")

    if source_y < 0 or source_y >= GRID_HEIGHT:
        raise ValueError("SOURCE_Y is outside the grid.")

    if FIT_START_FRACTION < 0:
        raise ValueError("FIT_START_FRACTION cannot be negative.")

    if FIT_END_FRACTION > 1:
        raise ValueError("FIT_END_FRACTION cannot exceed 1.")

    if FIT_START_FRACTION >= FIT_END_FRACTION:
        raise ValueError("FIT_START_FRACTION must be smaller than FIT_END_FRACTION.")

    if TRANSPARENT_THRESHOLD < 0:
        raise ValueError("TRANSPARENT_THRESHOLD cannot be negative.")


    from src.Diffusion_rates import (validate_2d_diffusion_ratio)

    r = get_diffusion_ratio()
    validate_2d_diffusion_ratio(r)

