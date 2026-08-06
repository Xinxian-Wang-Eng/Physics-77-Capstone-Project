from src.Analytical_2d import Analytical_Solution_at_a_time, analytical_diffusion_frames, total_mass
from src.visualization import plot_heatmap, plot_centerline, show_heatmap_slider
import numpy as np

# Generate one analytical solution frame
C, X, Y, x, y = Analytical_Solution_at_a_time(
    width=101,
    height=101,
    t=1.0,
    D=5.0,
    M=1.0,
    l=1.0
)

print("Total mass:", total_mass(C, l=1.0))

plot_heatmap(C, x=x, y=y, title="Analytical 2D Diffusion at t = 10")
plot_centerline(x, C, title="Centerline Profile at t = 10")


# Generate multiple frames for slider
frames, times = analytical_diffusion_frames(
    width=101,
    height=101,
    times= np.linspace(1, 100, 100),
    D=5.0,
    M=100.0,
    l=1.0
)

show_heatmap_slider(
    frames,
    times=times,
    x=x,
    y=y,
    title="Analytical 2D Diffusion"
)