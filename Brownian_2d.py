"""
brownian_2d.py

Particle-based 2D diffusion using an unbiased lattice random walk.
At each timestep, every particle can move:

    right
    left
    up
    down
    stay

The four movement directions have equal probability.

The probabilities are chosen so that the random walk represents the configured diffusion coefficient:


Array conventions
-----------------
positions[particle_index, coordinate]

coordinate:
    0 = x
    1 = y

frames[time_index, y, x]
"""
import Config
import numpy as np
from src.Diffusion_rates import (
    calculate_hopping_rate,
    calculate_diffusion_ratio,
    validate_2d_diffusion_ratio,
    calculate_stay_probability_2d,
    calculate_transition_probability,
    calculate_transition_probabilities_2d,
    calculate_biased_probabilities_2d
)
from src.Analysis import (fit_diffusion_coefficient_from_msd)


def initialize_particles(n_particles, source_x, source_y):
    """
    Place all particles at the point-source location.

    Returns
    positions : numpy.ndarray with Shape (n_particles, 2).
        coordinate 0 = x
        coordinate 1 = y
    """

    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")

    positions = np.zeros((n_particles, 2), dtype=int)
    #This part creates a large vertical array (maybe something like 100000 by 2), each row corresponds to a particle and the two columns representing x and y coords
    #This allows us to not iterate by 100000 times, but doing only array modifications
    positions[:, 0] = source_x
    positions[:, 1] = source_y

    return positions



def brownian_step(
    positions,
    p_right,
    p_left,
    p_up,
    p_down,
    width,
    height,
    rng):
    """
    Perform one random-walk timestep.
    Particles can move right, left, up, down, or stay.
    Reflecting outer boundaries are used:
    """

    n_particles = len(positions)

    # Random number between 0 and 1 for every particle
    random_values = rng.random(n_particles)

    new_positions = positions.copy()

    right_limit = (p_right)
    left_limit = (right_limit+ p_left)
    up_limit = (left_limit+ p_up)
    down_limit = (up_limit+ p_down)

    move_right = (random_values < right_limit)
    move_left = ((random_values >= right_limit)& (random_values < left_limit))
    move_up = ((random_values >= left_limit)& (random_values < up_limit))
    move_down = (random_values >= up_limit)& (random_values < down_limit)
    #=====================================================================================================================================================================================================================
    #This part controls the movement of the particle in a rather independent way, guaranteeing that at every time, four potential movements have the same probability being taken
    #However, it eliminates the possibility of moving diagonally, which might be addressed by dividing tick into two, allowing more pending ticks to be taken
    # If random_value >= 4*r, it's automatically staying still
    #=====================================================================================================================================================================================================================
    #The above  part wwas for original unbiased case
    #Now, each direction has independent probability

    # Remaining particles stay where they are.
    new_positions[move_right, 0] += 1
    new_positions[move_left, 0] -= 1
    new_positions[move_up, 1] += 1
    new_positions[move_down, 1] -= 1

    # Reflecting outer boundaries
    outside = (
        (new_positions[:, 0] < 0)
        | (new_positions[:, 0] >= width)
        | (new_positions[:, 1] < 0)
        | (new_positions[:, 1] >= height)
    )

    # Reject moves that leave the grid
    new_positions[outside] = positions[outside]

    return new_positions


def positions_to_concentration(
    positions,
    width,
    height,
    dx,
    total_mass
):
    """
    Convert particle positions into a 2D concentration field.

    Each particle represents an equal fraction of the total mass.

    Returns
    concentration : numpy.ndarray
        Shape (height, width).
    """

    n_particles = len(positions)

    if n_particles == 0:
        raise ValueError("There must be at least one particle.")

    particle_mass = (total_mass / n_particles)

    counts = np.zeros((height, width), dtype=int)

    x = positions[:, 0]
    y = positions[:, 1]

    np.add.at(counts, (y, x), 1)
    #This is what makes this part works: np.add, it would count number of rows in the huge array that have the same (y, x) and record it to the corresponding location in counts
    #This is equivalent to number of particles on each coordinate, allowing us to get the concentration at each spot 
    concentration = (counts * particle_mass / dx**2)

    return concentration

def calculate_msd(
    positions,
    source_x,
    source_y,
    dx
):
    """
    Calculate mean squared displacement from the
    current particle positions.
    """

    displacement_x = (positions[:, 0] - source_x) * dx
    displacement_y = (positions[:, 1] - source_y) * dx
    squared_displacement = (displacement_x**2 + displacement_y**2)
    msd = np.mean(squared_displacement)

    return msd


def simulate_brownian_2d(
    n_particles,
    steps,
    width,
    height,
    dx,
    dt,
    diffusion_coefficient,
    total_mass,
    source_x,
    source_y,
    probabilities,
    save_every=1,
    seed=Config.RANDOM_SEED
):
    """
    Simulate 2D Brownian diffusion using random walkers.

    Only the current particle positions are stored.
    Concentration fields are saved at selected timesteps.

    Returns
    frames : numpy.ndarray, Concentration fields with shape: [time_index, y, x]
    times : numpy.ndarray, Physical times corresponding to saved frames.
    """

    if steps <= 0:
        raise ValueError("n_steps must be positive.")

    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    rng = np.random.default_rng(seed)

    positions = initialize_particles(n_particles, source_x, source_y)
    p_right = probabilities["right"]
    p_left = probabilities["left"]
    p_up = probabilities["up"]
    p_down = probabilities["down"]
    p_stay = probabilities["stay"]

    frames = []
    times = []
    msd_history = []
    mass_history = []
    # Save initial condition at t = 0

    concentration = positions_to_concentration(
        positions,
        width,
        height,
        dx,
        total_mass
    )

    frames.append(concentration)
    times.append(0.0)
    msd_history.append(0.0)
    mass_history.append(
    np.sum(concentration) * dx**2)
    # Time evolution

    for step in range(1, steps + 1):
        positions = brownian_step(
            positions,
            p_right,
            p_left,
            p_up,
            p_down,
            width,
            height,
            rng
        )

        if step % save_every == 0:
        #This is not optional! Such simulation will create many many frames, meaning 100000 cmaps, occupying memory and increasing runtime
        #So don't put save_every = 1!!! Be nice to your computer!
            concentration = positions_to_concentration(positions,width,height,dx,total_mass)
            current_time = step * dt
            current_msd = calculate_msd(positions,source_x,source_y,dx)
            current_mass = (np.sum(concentration)* dx**2)

            frames.append(concentration)
            times.append(current_time)
            msd_history.append(current_msd)
            mass_history.append(current_mass)

    frames = np.array(frames)
    times = np.array(times)
    msd_history = np.array(msd_history)
    mass_history = np.array(mass_history)
    return frames, times, msd_history, mass_history


def run_from_config():
    """
    Run the Brownian simulation using config.py.
    """

    Config.validate_config()
    #Calculate Transition Probabilities
    r = calculate_diffusion_ratio(Config.DIFFUSION_COEFFICIENT, Config.DT, Config.DX)
    validate_2d_diffusion_ratio(r)
    probabilities = calculate_transition_probabilities_2d(r)
    p_right = probabilities["right"]
    p_left = probabilities["left"]
    p_up = probabilities["up"]
    p_down = probabilities["down"]
    p_stay = probabilities["stay"]
    source_x, source_y = (Config.get_source_position())

    (frames, times, msd, mass_history) = simulate_brownian_2d(
        n_particles=Config.NUM_PARTICLES,
        steps=Config.NUM_STEPS,
        width=Config.GRID_WIDTH,
        height=Config.GRID_HEIGHT,
        dx=Config.DX,
        dt=Config.DT,
        diffusion_coefficient=Config.DIFFUSION_COEFFICIENT,
        total_mass=Config.TOTAL_MASS,
        source_x=source_x,
        source_y=source_y,
        probabilities=probabilities,
        save_every=Config.SAVE_EVERY,
        seed=Config.RANDOM_SEED
    )
    from src.Analysis import (fit_diffusion_coefficient_from_msd,calculate_percent_error)    

    (measured_D, msd_slope, msd_intercept) = fit_diffusion_coefficient_from_msd(
        times,
        msd,
        start_fraction=Config.FIT_START_FRACTION,
        end_fraction=Config.FIT_END_FRACTION
    )
    percent_error = calculate_percent_error(measured_D,Config.DIFFUSION_COEFFICIENT)
    hopping_rate = calculate_hopping_rate(Config.DIFFUSION_COEFFICIENT, Config.DX)
    r = calculate_diffusion_ratio(Config.DIFFUSION_COEFFICIENT, Config.DT, Config.DX)
    stay_probability = (calculate_stay_probability_2d(r))


    result = {
        "model": "brownian",
        "times": times,
        "frames": frames,
        "msd": msd,
        "mass": mass_history,
        "move_probability": r,
        "right_probability": p_right,
        "left_probability":p_left,
        "up_probability":p_up,
        "down_probability":p_down,
        "stay_probability":p_stay,
        "source_position": (source_x,source_y),
        "diffusion_coefficient":Config.DIFFUSION_COEFFICIENT,
        "measured_diffusion_coefficient":measured_D,
        "msd_slope":msd_slope,
        "msd_intercept":msd_intercept,
        "hopping_rate": hopping_rate,
        "stay_probability": stay_probability,
        "total_mass":Config.TOTAL_MASS,
        "num_particles":Config.NUM_PARTICLES,
        "dx":Config.DX,
        "dt":Config.DT,
        "seed":Config.RANDOM_SEED,
        "diffusion_coefficient":Config.DIFFUSION_COEFFICIENT,
        "measured_diffusion_coefficient":measured_D,
        "diffusion_percent_error":percent_error,
    }

    return result

