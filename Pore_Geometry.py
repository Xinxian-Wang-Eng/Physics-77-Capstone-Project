"""
Pore_Geometry.py

Purpose
-------
This file defines and visualizes the porous geometry used in the project.

In this project, the porous medium is represented as a 2-D Boolean grid:

    True  = open pore space
    False = solid obstacle

The porosity is the fraction of cells that are open pore space.

This file is responsible for:
    1. Generating porous grids with a chosen porosity.
    2. Forcing the center/source cell to remain open.
    3. Measuring the actual porosity of a generated grid.
    4. Plotting the porous geometry.
    5. Generating multiple porous grids for different porosity values.

Notes for other project files
-----------------------------
Brownian_2d.py should use this grid so that particles can only move through
cells where porous_grid[y, x] == True.

Continuum_Walking_2d.py should use the same grid so that concentration only
diffuses through cells where porous_grid[y, x] == True.

Both models should use the same porous grid for a given porosity so that their
results can be compared under identical pore geometry.

Indexing convention
-------------------
The grid uses NumPy array indexing:

    porous_grid[y, x]

where:
    y = row index / vertical coordinate
    x = column index / horizontal coordinate
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_porous_grid(width=101, height=101, porosity=0.75, seed=None, force_center_open=True):
    """
    Generate a 2-D porous medium as a Boolean grid.

    Parameters
    ----------
    width : int
        Number of columns in the grid.

    height : int
        Number of rows in the grid.

    porosity : float
        Fraction of cells that are open pore space.
        porosity = 1.0 means all cells are open.
        porosity = 0.0 means all cells are solid.

    seed : int or None
        Random seed for reproducibility.

    force_center_open : bool
        If True, force the center cell to be open so particles or concentration
        can start from the center of the domain.

    Returns
    -------
    porous_grid : ndarray of bool
        Boolean grid with shape (height, width).
        True means open pore space.
        False means solid obstacle.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")

    if porosity < 0 or porosity > 1:
        raise ValueError("porosity must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    porous_grid = rng.random((height, width)) < porosity

    if force_center_open:
        center_y = height // 2
        center_x = width // 2
        porous_grid[center_y, center_x] = True

    return porous_grid


def actual_porosity(porous_grid):
    """
    Calculate the actual porosity of a generated porous grid.

    Parameters
    ----------
    porous_grid : ndarray of bool
        True means open pore space.
        False means solid obstacle.

    Returns
    -------
    float
        Fraction of cells that are open.
    """

    porous_grid = np.asarray(porous_grid)

    if porous_grid.ndim != 2:
        raise ValueError("porous_grid must be a 2-D array.")

    return np.mean(porous_grid)


def plot_porous_grid(porous_grid, title="Porous Geometry"):
    """
    Plot a 2-D porous medium.

    Parameters
    ----------
    porous_grid : ndarray of bool
        True means open pore space.
        False means solid obstacle.

    title : str
        Plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    porous_grid = np.asarray(porous_grid)

    if porous_grid.ndim != 2:
        raise ValueError("porous_grid must be a 2-D array.")

    fig, ax = plt.subplots(figsize=(6, 5))

    img = ax.imshow(
        porous_grid,
        origin="lower",
        interpolation="none"
    )

    ax.set_title(title + f"\nActual porosity = {actual_porosity(porous_grid):.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Open pore = 1, solid obstacle = 0")

    plt.show()

    return fig, ax


def generate_porosity_series(width=101, height=101, porosities=None, seed=None):
    """
    Generate several porous grids with different porosity values.

    Parameters
    ----------
    width : int
        Number of columns in each grid.

    height : int
        Number of rows in each grid.

    porosities : list or None
        List of porosity values to generate.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    grids : dict
        Dictionary where keys are target porosity values and values are porous grids.
    """

    if porosities is None:
        porosities = [0.90, 0.75, 0.60, 0.50, 0.25]

    grids = {}

    for i, porosity in enumerate(porosities):
        grid_seed = None if seed is None else seed + i

        grids[porosity] = generate_porous_grid(
            width=width,
            height=height,
            porosity=porosity,
            seed=grid_seed,
            force_center_open=True
        )

    return grids
