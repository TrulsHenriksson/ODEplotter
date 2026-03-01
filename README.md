# Purpose
This is a project designed to help visualize solutions to ODEs, by giving simple interfaces to get numerical solutions, as well as plotting and animating them.

![Lorenz attractor in 3D](/assets/Lorenz%20attractor%20rotating.gif)

# Background
An Ordinary Differential Equation (ODE) is an equation on the form
$$
\begin{cases}
    \displaystyle\frac{\mathrm{d}y}{\mathrm{d}t}(t) = f(t, y(t)), \\[8pt]
    y(t_{\text{start}}) = y_{\text{start}}
\end{cases}
$$
given a function $f: I\times\mathbb{R}^n \to \mathbb{R}^n$ and some initial condition $y_{\text{start}} \in \mathbb{R}^n$. A solution to an ODE is a function $y: I \to \mathbb{R}^n$ with domain $I = [t_{\text{start}}, t_{\text{end}}]$ that satisfies the differential equation and initial condition.

# Main idea
This project is written in Python only, which means we trade speed for flexibility. Here are some things that are possible:
- Solving ODEs:
    - Solve ODEs in any number of dimensions
    - Solve until some $t_{\text{end}}$, do something else, then resume and solve further
    - Easily solve linear ODEs and planar (2D) ODEs using convenient abstractions
    - Interpolate linearly between points of the discrete solution
    - Solve ODEs while respecting user-defined obstacles with custom behavior
    - Solve ODEs with JIT (just-in-time) compiled methods for significant speedups
- Plotting solutions:
    - Plot given coordinates of a solution in 2D or 3D
    - Plot a solution using a custom projection function into 2D or 3D
    - Plot the step size over time for adaptive methods
- Animating solutions:
    - Animate any number of solutions together in 2D or 3D
    - Animate a solution as a point with a trail (phase diagram)
    - Animate a solution in a custom way using any matplotlib Artist
    - Easily overlay text displaying the time $t$
    - Animate indefinitely in interactive mode
    - Save animation as .mp4 (requires ffmpeg) or .gif
    - Insert pauses in the animation
    - Animate in "virtual time", jumping forward a fixed number of solution steps instead of a fixed length of time per frame (useful for adaptive methods)

This library is better suited to visualize solutions for relative short durations, rather than solve and analyze them for long time spans. Not impossible, but there are better tools to do that.

# Important components
## ODE
Class for representing the differential equation. Only needs the right hand side function $f$ (called `derivative`) to instantiate it, so a single `ODE` can be solved for different initial values.

An ODE can be also be defined by creating a subclass of `ODE`.

Methods:
- `solve`: Give initial conditions, method name, and method parameters. Returns a `DiscreteSolution`.
- `solve_single`: Like `solve`, but only returns the next `(t, y)` point from the solution method.
- `draw_vector_field`: Plot a 2D projection of the vector field defined by $f$.

## DiscreteSolution
Class for representing a discrete solution to an `ODE`. Has the attributes `ts` and `ys`, which are lists of $t$ and $y$ values approximating the exact solution ($y_k \approx y(t_k)$ where $y$ is the exact solution). Includes methods to query and linearly interpolate between these points. These lists are extended as needed using the `point_gen` it gets when instantiated (usually by `ODE.solve`).

Methods:
- `load`, `load_until`: Load new points by solving for them.
- `__call__`: Evaluate the linear interpolation at one or several times $t$.
- `__getitem__`, `get_arrays`, `get_arrays_between`: Convenience functions to get the `ts`, `ys` lists as arrays.
- `plot`, `plot_3d`: Plot two or three coordinates of the vectors, or using an optional custom projection.
- `plot_time_steps`: Plot the time steps $h_k$ over time. Only useful for adaptive-step methods.

## SolutionAnimator
Class for animating several `DiscreteSolution`s in parallel. Uses `matplotlib.animation.FuncAnimation` as a backend, and so reuses the language of update functions and init functions. To animate anything, only a `DiscreteSolution`, an update function, and an optional init function need to be provided, which makes for a very general animation system.

Methods:
- `animate`: Add a `DiscreteSolution` to be animated, along with an update function and optional init function that define how the solution vectors should be visualized. The update function is called once per frame with the `ts` and `ys` that are new for that frame. The init function is called when the animation is started or restarted.
- `animate_time_text`, `animate_phase_diagram`, `animate_phase_diagram_3d`: Convenience functions for common animation types. The `phase_diagram` variants behave similarly to `DiscreteSolution.plot` and `.plot_3d`.
- `render`: Assemble all animations into a `FuncAnimation object`. Can be animated indefinitely, or saved to a gif or mp4.

## Obstacles
Class for defining interruptions to the solution of ODEs. Maybe you want to detect the precise moment a double pendulum flips over, or three bodies in the three-body problem form a syzygy, or something else. If you also want the solution to behave differently after that point, `Obstacle`s are the way to do this. An `Obstacle` is defined by its position in phase space, given as a signed distance function, and its effect on a solution's state vector when hit.

An `Obstacle`'s methods shouldn't be called by the user. If used when solving, the `ODE` class handles it.
