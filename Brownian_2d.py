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

The probabilities are chosen so that the random walk represents
the configured diffusion coefficient:

    move probability in each direction = D * dt / dx^2

Array conventions
-----------------
positions[particle_index, coordinate]

coordinate:
    0 = x
    1 = y

frames[time_index, y, x]
"""

import numpy as np

def calculate_move_probability(
    diffusion_coefficient,
    dt,
    dx
):
    """
    Calculate the probability of moving in each direction
    during one timestep.

    Returns
    -------
    move_probability : float
        Probability of moving in EACH of the four directions.
    """

    if diffusion_coefficient <= 0:
        raise ValueError(
            "diffusion_coefficient must be positive."
        )

    if dt <= 0:
        raise ValueError(
            "dt must be positive."
        )

    if dx <= 0:
        raise ValueError(
            "dx must be positive."
        )

    move_probability = (diffusion_coefficient * dt / dx**2)

    if move_probability > 0.25:
        raise ValueError(
            "D * dt / dx^2 must be <= 0.25."
        )

    return move_probability


def initialize_particles(
    n_particles,
    source_x,
    source_y
):
    """
    Place all particles at the point-source location.

    Returns
    -------
    positions : numpy.ndarray
        Shape (n_particles, 2).

        coordinate 0 = x
        coordinate 1 = y
    """

    if n_particles <= 0:
        raise ValueError(
            "n_particles must be positive."
        )

    positions = np.zeros(
        (n_particles, 2),
        dtype=int
    )
    #This part creates a large vertical array (maybe something like 100000 by 2), each row corresponds to a particle and the two columns representing x and y coords
    #This allows us to not iterate by 100000 times, but doing only array modifications
    positions[:, 0] = source_x
    positions[:, 1] = source_y

    return positions

def brownian_step(
    positions,
    move_probability,
    width,
    height,
    rng
):
    """
    Perform one random-walk timestep.

    Particles can move right, left, up, down, or stay.

    Reflecting outer boundaries are used:
    an attempted move outside the grid is rejected.
    """

    n_particles = len(positions)

    # Random number between 0 and 1 for every particle
    random_values = rng.random(
        n_particles
    )

    new_positions = positions.copy()

    p = move_probability


    move_right = (random_values < p)
    move_left = ((random_values >= p) & (random_values < 2 * p))
    move_up = ((random_values >= 2 * p) & (random_values < 3 * p))
    move_down = ((random_values >= 3 * p) & (random_values < 4 * p))
    #This part controls the movement of the particle in a rather independent way, guaranteeing that at every time, four potential movements have the same probability being taken
    #However, it eliminates the possibility of moving diagonally, which might be addressed by dividing tick into two, allowing more pending ticks to be taken
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
        raise ValueError(
            "There must be at least one particle."
        )

    particle_mass = (
        total_mass / n_particles
    )

    counts = np.zeros(
        (height, width),
        dtype=int
    )

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

    displacement_x = (
        positions[:, 0]
        - source_x
    ) * dx

    displacement_y = (
        positions[:, 1]
        - source_y
    ) * dx

    squared_displacement = (
        displacement_x**2
        +
        displacement_y**2
    )

    msd = np.mean(
        squared_displacement
    )

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
    save_every=1,
    seed=None
):
    """
    Simulate 2D Brownian diffusion using random walkers.

    Only the current particle positions are stored.
    Concentration fields are saved at selected timesteps.

    Returns
    -------
    frames : numpy.ndarray
        Concentration fields with shape:

            [time_index, y, x]

    times : numpy.ndarray
        Physical times corresponding to saved frames.
    """

    if steps <= 0:
        raise ValueError("n_steps must be positive.")

    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    rng = np.random.default_rng(seed)

    move_probability = (calculate_move_probability(diffusion_coefficient, dt, dx))

    positions = initialize_particles(n_particles, source_x, source_y)

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
    np.sum(concentration) * dx**2
)
    # Time evolution

    for step in range(1, steps + 1):
        positions = brownian_step(
            positions,
            move_probability,
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

    import Config

    from src.Analysis import (fit_diffusion_coefficient_from_msd)

    Config.validate_config()

    source_x, source_y = (Config.get_source_position())

    (frames, times, msd, mass_history) = simulate_brownian_2d(
        n_particles=Config.NUM_PARTICLES,
        steps=Config.NUM_STEPS,
        width=Config.GRID_WIDTH,
        height=Config.GRID_HEIGHT,
        dx=Config.l,
        dt=Config.DT,
        diffusion_coefficient=Config.DIFFUSION_COEFFICIENT,
        total_mass=Config.TOTAL_MASS,
        source_x=source_x,
        source_y=source_y,
        save_every=Config.SAVE_EVERY,
        seed=Config.RANDOM_SEED
    )

    (measured_D, msd_slope, msd_intercept) = fit_diffusion_coefficient_from_msd(
        times,
        msd,
        start_fraction=Config.FIT_START_FRACTION,
        end_fraction=Config.FIT_END_FRACTION
    )

    move_probability = (
        calculate_move_probability(
            Config.DIFFUSION_COEFFICIENT,
            Config.DT,
            Config.l
        )
    )

    result = {
        "model": "brownian",
        "times": times,
        "frames": frames,
        "msd": msd,
        "mass": mass_history,

        "source_position": (
            source_x,
            source_y
        ),

        "diffusion_coefficient":
            Config.DIFFUSION_COEFFICIENT,

        "measured_diffusion_coefficient":
            measured_D,

        "msd_slope":
            msd_slope,

        "msd_intercept":
            msd_intercept,

        "move_probability":
            move_probability,

        "total_mass":
            Config.TOTAL_MASS,

        "num_particles":
            Config.NUM_PARTICLES,

        "l":
            Config.l,

        "dt":
            Config.DT,

        "seed":
            Config.RANDOM_SEED
    }

    return result