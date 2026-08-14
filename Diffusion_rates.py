"""
Diffusion_rates.py

Calculates the r value, which is the lattice diffusion transition probability

The main relationships are:

    k = D / dx^2
    r = k * dt

so:
    r = D * dt / dx^2

where:
    D  = diffusion coefficient
    dx = lattice spacing
    dt = timestep
    k  = hopping rate to one neighboring site
    r  = transition probability to one neighboring site
         during one timestep

For an unbiased 2D square lattice, there are four possible neighboring moves:
    right, left, up, down
Each has probability r.

Therefore, probability of staying still is:
    p_stay = 1 - 4r

and we require:
    0 <= r <= 0.25
"""
import Config

# HOPPING RATE

def calculate_hopping_rate(diffusion_coefficient, dx):
    """
    Calculate the hopping rate to one neighboring site.

        k = D / dx^2

    Parameters: 
    diffusion_coefficient : float, Diffusion coefficient D.
    dx : float, Lattice spacing.

    Returns: 
    hopping_rate : float, Hopping rate k, with units of 1 / time.
    """

    if diffusion_coefficient <= 0:
        raise ValueError("diffusion_coefficient must be positive.")

    if dx <= 0:
        raise ValueError("dx must be positive.")

    hopping_rate = (diffusion_coefficient / dx**2)

    return hopping_rate


# TRANSITION PROBABILITY

def calculate_transition_probability(hopping_rate, dt):
    """
    Convert a hopping rate into a transition probability.
        r = k * dt
    r is the probability of moving to one neighboring
    lattice site during one timestep.
    """

    if hopping_rate < 0:
        raise ValueError("hopping_rate cannot be negative.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    r = hopping_rate * dt

    return r

def calculate_transition_probabilities_2d(diffusion_ratio):
    """
    Convert the base diffusion probability r into directional transition probabilities.
    Without bias:
        right = r
        left  = r
        up    = r
        down  = r
        stay  = 1 - 4r

    With bias:
        right = r + bias_x
        left  = r - bias_x
        up    = r + bias_y
        down  = r - bias_y
        stay  = 1 - sum(all above)
    """

    r = diffusion_ratio

    validate_2d_diffusion_ratio(r)


    # Get bias settings from Config.py
    if Config.USE_BIAS:
        bias_x = Config.BIAS_X
        bias_y = Config.BIAS_Y

    else:
        bias_x = 0.0
        bias_y = 0.0


    # Validate bias
    if abs(bias_x) > r:
        raise ValueError("Absolute BIAS_X cannot be larger than the diffusion ratio.")
    if abs(bias_y) > r:
        raise ValueError("Absolute BIAS_Y cannot be larger than the diffusion ratio.")


    # Directional probabilities
    p_right = r + bias_x
    p_left = r - bias_x
    p_up = r + bias_y
    p_down = r - bias_y

    p_stay = (calculate_stay_probability_2d(r))

    return {
        "right": p_right,
        "left": p_left,
        "up": p_up,
        "down": p_down,
        "stay": p_stay
    }

# DIFFUSION RATIO

def calculate_diffusion_ratio(diffusion_coefficient, dt, dx):
    """
    Calculate the dimensionless diffusion ratio:

        r = D * dt / dx^2

    This same quantity appears as:

    1. The movement probability to each neighboring site in the unbiased Brownian lattice model.

    2. The probability-transfer coefficient in the Master Equation.

    3. The finite-difference coefficient in the continuum diffusion solver.
    """

    hopping_rate = calculate_hopping_rate(diffusion_coefficient, dx)

    r = calculate_transition_probability(hopping_rate, dt)

    return r

# GENERAL LATTICE VALIDATION

def validate_lattice_probability(r, number_of_neighbors):
    """
    Check whether r gives valid probabilities for
    a lattice with a specified number of neighbors.

    The total movement probability is:
        number_of_neighbors * r

    Therefore:
        number_of_neighbors * r <= 1
    """

    if r < 0:
        raise ValueError("Transition probability r cannot be negative.")

    if number_of_neighbors <= 0:
        raise ValueError("number_of_neighbors must be positive.")

    if number_of_neighbors * r > 1:
        raise ValueError("Transition probabilities are invalid: number_of_neighbors * r must be <= 1.")


# STAY PROBABILITY

def calculate_stay_probability(r, number_of_neighbors):
    """
    Calculate the probability of remaining at
    the current lattice site.

        p_stay = 1 - number_of_neighbors * r
    """

    validate_lattice_probability(r, number_of_neighbors)

    stay_probability = (1 - number_of_neighbors * r)

    return stay_probability

# Biased Case
def calculate_biased_probabilities_2d(diffusion_ratio,bias_x=0.0,bias_y=0.0):
    """
    Calculate directional transition probabilities for
    a 2D random walk with optional directional bias.

    Parameters
    diffusion_ratio : float, Base probability r of moving in each direction for unbiased diffusion.
    bias_x : float, Bias in the horizontal direction.
        positive: favors right
        negative: favors left
    bias_y : float, Bias in the vertical direction.
        positive: favors up
        negative: favors down

    Returns
    Dictionary containing probabilities for:
        right
        left
        up
        down
        stay
    """

    r = diffusion_ratio
    validate_2d_diffusion_ratio(r)

    if Config.USE_BIAS:
        bias_x = Config.BIAS_X
        bias_y = Config.BIAS_Y
        if abs(bias_x) > r:
            raise ValueError("BIAS_X is too large, its absolute value cannot exceed the diffusion ratio.")
        if abs(bias_y) > r:
            raise ValueError("BIAS_Y is too large, tts absolute value cannot exceed the diffusion ratio.")
    else:
        # Setting both biases to zero automatically
        # reproduces the original unbiased random walk.
        bias_x = 0.0
        bias_y = 0.0

    # Directional probabilities
    right_probability = (r + bias_x)
    left_probability = (r - bias_x)
    up_probability = (r + bias_y)
    down_probability = (r - bias_y)


    # Bias redistributes movement probability between opposite directions, so the total movement probability is still 4r.
    stay_probability = (calculate_stay_probability_2d(r))

    probabilities = [
        right_probability,
        left_probability,
        up_probability,
        down_probability,
        stay_probability
    ]

    return {
        "right": right_probability,
        "left": left_probability,
        "up": up_probability,
        "down": down_probability,
        "stay": stay_probability
    }



# 2D CONVENIENCE FUNCTIONS

def validate_2d_diffusion_ratio(r):
    """
    Validate r for a 2D square lattice.

    There are four neighboring sites:

        right
        left
        up
        down

    so:
        r <= 0.25
    """

    validate_lattice_probability(r, number_of_neighbors=4)


def calculate_stay_probability_2d(r):
    """
    Calculate staying probability for a 2D four-neighbor lattice.
        p_stay = 1 - 4r
    """

    return calculate_stay_probability(r, number_of_neighbors=4)
