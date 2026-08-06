import numpy as np

def Grid_Creator(width=101, height=101, l=1.0):

    # Create a 2D coordinate grid centered at (0, 0)

    # Parameters
    # width : int, Number of grid points in the x direction

    # height : int, Number of grid points in the y direction

    # l: float, Spacing between grid points

    x = (np.arange(width) - width // 2) * l
    y = (np.arange(height) - height // 2) * l
    # This creates centeralized 1-D arrays for x, y coordinates, with 0 as the origin and spacing of l


    X, Y = np.meshgrid(x, y)
    # X, Y: 2-D Grid as the "Blank Canvas", with each coordinate having a corresponding concentration value that can be assigned

    return X, Y, x, y

def Analytical_Solution_at_a_time (width = 101, height = 101, t = 1.0, D = 1.0, M = 1.0, l = 1.0):
    #Calculates the concentration at all coordinates on the canvas at each time stamp
    #t: float, "unit time"
    #
    X, Y, x, y = Grid_Creator(width, height, l)
    r_squared = X**2 + Y**2
    C = (M / (4 * np.pi * D * t)) * np.exp(-r_squared / (4 * D * t))
    return C, X, Y, x, y

def analytical_diffusion_frames(width=101, height=101, times=None, D=1.0, M=1.0, l=1.0):
    # Generates "heatmaps" for the entire canvas at each time stamp and combines them into a set

    # Parameters
    # width : int, Number of grid points in x direction.
    # height : int, Number of grid points in y direction.
    # times : array-like, List or array of time values. All must be greater than 0.

    # D : float
    #     Diffusion coefficient.

    # M : float
    #     Total amount of concentration/mass.

    # dx : float
    #     Grid spacing.

    # Returns
    # frames : ndarray
    #     Array with shape (num_times, height, width).

    # times : ndarray
    #     Array of time values.


    if times is None:
        times = np.linspace(1, 100, 100)

    times = np.asarray(times, dtype=float)

    frames = []

    for t in times:
        C, X, Y, x, y = Analytical_Solution_at_a_time(
            width=width,
            height=height,
            t=t,
            D=D,
            M=M,
            l=l
        )
        frames.append(C)

    return np.array(frames), times

def total_mass(C, l=1.0):
    """
    Estimate total mass/concentration in a 2D concentration field.

    For a continuous integral:
        M ≈ sum(C) * dx^2

    Parameters
    ----------
    C : ndarray
        2D concentration field.

    dx : float
        Grid spacing.

    Returns
    -------
    mass : float
        Estimated total mass.
    """

    return np.sum(C) * l**2