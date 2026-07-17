import numpy as np
import scipy as scy
import matplotlib.pyplot as plt
from  matplotlib.widgets import Slider
import random as rand

def simplified_1D_Brownian (X_range, time, source_location, source_concentration = 1):
    X_axis = np.arrange(-X_range, X_range+1, 1)
    X = np.zeros(len(X_axis))
    X[source_location] = source_concentration
    dot_locations = []
    for t in range(time+1):
        dot_index = np.where(X != 0)[0][0]



def simplified_brownian(X, time):
    X = X.copy()
    L = len(X)

    x_axis = np.arange(L) - L // 2

    dot_locations = []

    for t in range(time + 1):
        dot_index = np.where(X == 1)[0][0]
        dot_location = x_axis[dot_index]
        dot_locations.append(dot_location)

        step = rand.choice([-1, 1])
        new_index = dot_index + step

        if 0 <= new_index < L:
            X[dot_index] = 0
            X[new_index] = 1
        else:
            pass
    return np.array(dot_locations)


def Brownian_1D(x_range, time):
    # Create domain from -x_range to x_range - 1
    X = np.zeros(2 * x_range)

    # Put particle at center, which represents x = 0
    X[x_range] = 1

    dot_locations = simplified_brownian(X, time)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(range(time + 1), dot_locations, s=15)
    ax.plot(range(time + 1), dot_locations, linewidth=1)

    ax.axhline(0, linestyle="--", label="x = 0")

    ax.set_ylim(-x_range, x_range)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Particle location")
    ax.set_title("1D Brownian motion")
    ax.grid(True)
    ax.legend()

    plt.show()
    sum = np.sum(dot_locations)
    return fig, sum
Brownian_1D(10, 50)
