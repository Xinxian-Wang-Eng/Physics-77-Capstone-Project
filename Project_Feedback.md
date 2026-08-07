# Checkpoint Feedback

You responded well to the proposal feedback. The checkpoint now explains how you will generate porous media, vary porosity, impose no-flux boundaries, and validate both models under free-diffusion conditions before adding obstacles. The scientific plan is much clearer than it was in the proposal.

The main concern is that the checkpoint describes research and planned methods but does not show evidence of a working implementation or preliminary numerical results. You now need to move from background research to code, testing, and quantitative comparison.

## What is going well

- You have a clear comparison between a microscopic particle model and a macroscopic continuum model.
- You plan to validate both methods in the simpler free-diffusion case before introducing porous media.
- You identified a specific continuum method: finite-difference time stepping.
- You identified a boundary condition for solid obstacles: no flux through the solid boundary.
- You plan to vary porosity systematically rather than study only one obstacle arrangement.
- Your planned outputs—particle trajectories, concentration heatmaps, mean-square displacement, and effective diffusion coefficients—can support meaningful quantitative conclusions.

## Define both numerical models precisely

Your checkpoint explains the models in words, but the final report must state the governing diffusion equation, the finite-difference approximation, and the random-walk rules. Define every variable, parameter, index, initial condition, and boundary condition you use.

Do not copy a discretization from a source without understanding it. Every team member should be able to explain:

- What concentration and the diffusion coefficient represent
- How a spatial derivative is approximated on a grid
- How the concentration is updated from one time step to the next
- How the grid spacing and time step affect the calculation
- How the random walk produces diffusion when many trajectories are averaged

For the connection between random walks and diffusion, consult the relevant material in [MIT OpenCourseWare's Fields, Forces, and Flows in Biological Systems](https://ocw.mit.edu/courses/20-430j-fields-forces-and-flows-in-biological-systems-fall-2015/) and [Random Walks and Diffusion](https://ocw.mit.edu/courses/18-366-random-walks-and-diffusion-fall-2006/). Locate the sections that derive diffusion from a random walk and connect mean-square displacement with the diffusion coefficient.

## Check the finite-difference stability requirement

An explicit finite-difference diffusion solver can become unstable when the time step is too large relative to the grid spacing and diffusion coefficient. An unstable result may oscillate, become negative, or grow without a physical cause.

Use a numerical-methods reference to find and derive the stability condition for your particular 2D update. The [NIST FiPy documentation](https://pages.nist.gov/fipy/en/stable/) discusses diffusion solvers, discretization, time-step stability, analytical comparisons, and boundary conditions. Be careful about dimensionality: a stability limit stated for a 1D problem should not be copied directly into a 2D implementation.

In your report:

1.  State the diffusion coefficient, grid spacing, and time step.
2.  Show that your chosen values satisfy the appropriate 2D stability condition.
3.  Check that total concentration remains approximately constant.
4.  Check that the solver does not produce significant negative concentrations.
5.  Repeat the calculation at finer spatial and temporal resolutions to confirm that the result converges.

## Use consistent physical parameters in both models

The particle and continuum simulations must describe the same physical diffusion process. The particle step length and time interval determine an effective diffusion coefficient. Derive this relationship from the random-walk literature and use it to choose parameters consistent with the continuum solver.

Document the particle rules precisely:

- Which directions can a particle move?
- Are all directions equally probable?
- Is the step length fixed?
- Can particles move diagonally?
- What happens at the edge of the domain?
- What happens when a particle attempts to enter an obstacle?

These choices change the model and may change its effective diffusion coefficient.

## Validate free diffusion before adding obstacles

### Particle model

1.  Begin with particles at a clearly defined initial location or distribution.
2.  Run many independent random walks.
3.  Calculate and plot mean-square displacement (MSD) versus time.
4.  Fit the portion that should be linear.
5.  Use its slope to estimate the diffusion coefficient.
6.  Compare the measured value with the value predicted from the step length and time interval.
7.  Repeat the experiment with more particles to show how statistical noise changes.

Record the random-number seed so that the results can be reproduced.

### Continuum model

1.  Begin with an initial concentration profile that has a known analytical solution.
2.  Evolve it without obstacles.
3.  Compare numerical and analytical concentration profiles at several times.
4.  Calculate an error measure, such as root-mean-square error.
5.  Repeat the test with smaller grid spacing and time steps and demonstrate that the error decreases.

The examples in the [NIST FiPy documentation](https://pages.nist.gov/fipy/en/stable/) show how numerical diffusion problems specify their grids, coefficients, boundary conditions, time steps, and validation. You do not need to use FiPy, but you can use its documentation as a model for describing your own implementation.

Do not proceed to complicated porous geometries until both free-diffusion tests behave correctly.

## Define the porous geometry carefully

Changing the percentage of open cells is a reasonable starting point, but porosity alone does not completely describe a geometry. Two grids can have the same fraction of open cells while one contains a path across the domain and the other contains disconnected pockets.

Specify:

1.  How obstacle cells are selected
2.  Whether obstacles are single cells or larger shapes
3.  Whether the pore space must connect across the domain
4.  Which porosity values will be tested
5.  How many independent random geometries will be tested at each porosity

Use several geometries at each porosity and report the mean and variation of the effective diffusion coefficient. Save the obstacle mask and random seed for every run. Both numerical models must use the exact same obstacle mask for a fair comparison.

## Implement no-flux boundaries correctly

For the particle model, decide whether a step into a solid cell is rejected or reflected. State and justify the rule.

For the continuum model, do not set the concentration in obstacle cells to zero at every step without further consideration. That can behave like an absorbing boundary and incorrectly remove material. Implement a zero-flux condition at each fluid–solid interface and verify it by checking conservation of total concentration.

The [NIST FiPy documentation](https://pages.nist.gov/fipy/en/stable/) explains flux-based discretization and boundary conditions. Even if you implement finite differences rather than use FiPy, this material can help you understand what a no-flux boundary means physically and numerically.

## Compare the models using common measurements

Use the same grid, initial condition, obstacle mask, diffusion coefficient, boundary conditions, and output times in both models.

Convert particle locations into a concentration field by binning them into the same grid cells used by the continuum model. Then compare:

- Concentration heatmaps at identical times
- Concentration slices through the same location
- MSD as a function of time
- Effective diffusion coefficient as a function of porosity
- A numerical error or difference between the two concentration fields
- Conservation of particle number and total concentration

When the methods disagree, investigate whether the difference comes from finite particle statistics, grid resolution, time-step error, boundary handling, domain size, or inconsistent initialization.

## Required progress for the next stage

Aim to complete:

1.  A working 1D random walk with the expected MSD behavior
2.  A working 2D free random walk
3.  A working 2D continuum solver that satisfies its stability requirement
4.  A free-diffusion validation plot for each model
5.  A shared obstacle-mask generator
6.  One simple porous test completed with both models
7.  A table listing simulation parameters, units, and random seeds
8.  A clear division of implementation, testing, analysis, and writing responsibilities

Your project is well scoped and has a strong scientific question. The next priority is to demonstrate that each numerical method works independently under known conditions. Only after that validation should you use the models to draw conclusions about diffusion in porous media.

## Recommended references

- [MIT OpenCourseWare: Fields, Forces, and Flows in Biological Systems](https://ocw.mit.edu/courses/20-430j-fields-forces-and-flows-in-biological-systems-fall-2015/)
- [MIT OpenCourseWare: Random Walks and Diffusion](https://ocw.mit.edu/courses/18-366-random-walks-and-diffusion-fall-2006/)
- [NIST FiPy Documentation](https://pages.nist.gov/fipy/en/stable/)
