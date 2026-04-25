import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count, chain, repeat
from functools import wraps

from typing import overload, Iterable, Callable
from .utils.types import *

from .discrete_solution import DiscreteSolution


def flattened_once[T](list_of_lists: Iterable[Iterable[T]]) -> list[T]:
    return [val for sublist in list_of_lists for val in sublist]

# Marginally faster than np.concat((array, [value])), much faster than np.append
def append(array: np.ndarray, value) -> np.ndarray:
    return np.concat((array, (value,)))


def cannot_call_after_rendering(method):
    """Stop the user from doing things after `SolutionAnimator.render` has been called."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "animation"):
            raise RuntimeError("Cannot do this after rendering an animation")
        return method(self, *args, **kwargs)
    return wrapper


class SolutionAnimator:
    solutions: list[DiscreteSolution]
    update_functions: list[tuple[Callable[[TimeArray, VectorArray], tuple[Artist, ...]], int]]
    no_solution_update_functions: list[Callable[[TimeArray], tuple[Artist, ...]]]
    init_functions: list[Callable[[Time], tuple[Artist, ...]]]

    def __init__(self):
        """An animation object that animates DiscreteSolutions in parallel."""
        self.solutions = []
        # Lists of various types of update functions
        self.update_functions = []
        self.no_solution_update_functions = []
        # List of init functions
        self.init_functions = []

    @overload
    def animate(
        self,
        solution: DiscreteSolution,
        update_function: Callable[[TimeArray, VectorArray], tuple[Artist, ...]],
        init_function: Callable[[Time], tuple[Artist, ...]] | None = None,
    ): ...

    @overload
    def animate(
        self,
        solution: None,
        update_function: Callable[[TimeArray], tuple[Artist, ...]],
        init_function: Callable[[Time], tuple[Artist, ...]] | None = None,
    ): ...

    @cannot_call_after_rendering
    def animate(self, solution, update_function, init_function=None):
        """Add a solution to be animated.

        Arguments
        ---------

        solution : DiscreteSolution, or None
            A solution to take time and vector values from and give to update_function. If `None`, only the time
            is passed to `update_function`.
        update_function : (TimeArray, VectorArray) or (TimeArray,) -> tuple of `Artist`s
            Function that takes the times and vectors from an animation frame, updates some `Artist`s (e.g. lines),
            and returns a tuple of those `Artist`s.
        init_function : (Time) -> tuple of `Artist`s
            Function that is called with the start time before the first animation frame, resets some
            `Artist`s (e.g. lines), and returns a tuple of those `Artist`s.
        """
        if solution is None:
            self.no_solution_update_functions.append(update_function)
        else:
            if not solution in self.solutions:
                self.solutions.append(solution)
            index = self.solutions.index(solution)
            self.update_functions.append((update_function, index))
        # Add init_function if it was given
        if init_function is not None:
            self.init_functions.append(init_function)

    @cannot_call_after_rendering
    def animate_time_text(
        self,
        ax: Axes | Axes3D | None = None,
        loc: str = "upper left",
        decimals: int = 2,
        t_max: float = np.inf
    ):
        """Animate a text object with the current time.

        Arguments
        ---------

        ax : Axes or Axes3D or None
            Which axes to animate the time text on. Defaults to `plt.gca()`.
        loc : str, any combination of "upper"/"lower" and "left"/"right".
            Which corner to put the text in.
        decimals : int
            How many decimals of the time to show.
        t_max : float
            Which time to stop updating the time text at.
        """
        if ax is None:
            ax = plt.gca()
        time_text = SolutionAnimator.get_time_text(ax, loc)
        updater = SolutionAnimator.time_text_updater(time_text, decimals, t_max)
        resetter = SolutionAnimator.time_text_resetter(time_text, decimals)
        self.animate(None, updater, resetter)

    @cannot_call_after_rendering
    def animate_phase_diagram(
        self,
        sol: DiscreteSolution,
        ax: Axes | None = None,
        xcoord: int = -1,
        ycoord: int = 0,
        trail_length: int = 0,
        skip_last: bool = False,
        projection: Callable[[TimeArray, VectorArray], tuple[np.ndarray, np.ndarray]] | None = None,
        **line_kwargs
    ):
        """Animate two coordinates of the solution vectors.

        Every frame, a number of new `(t, y)` points are solved for. These points are projected on the
        x- and y-axis by plotting `(*y, t)[xcoord]` on the x-axis and `(*y, t)[ycoord]` on the y-axis.

        Arguments
        ---------
        sol : DiscreteSolution
            Solution as gotten from `ODE.solve()`, for example.
        ax : Axes
            Which axes to animate on. Defaults to `plt.gca()`.
        xcoord, ycoord : int (default: -1, 0)
            Which index of the phase vector `(*y, t)` to plot on the x- and y-axis. The default represents plotting `y[0]` against `t`.
        trail_length : int (default: 0)
            How many points to keep from the end of the line. Default: all.
        **line_kwargs
            Keyword arguments forwarded to `SolutionAnimator.get_line`.
        """
        if ax is None:
            ax = plt.gca()
        line = SolutionAnimator.get_line(ax, **line_kwargs)
        updater = SolutionAnimator.phase_diagram_updater(line, xcoord, ycoord, trail_length, skip_last, projection)
        resetter = SolutionAnimator.line_resetter(line)
        self.animate(sol, updater, resetter)

    @cannot_call_after_rendering
    def animate_phase_diagram_3d(
        self,
        sol: DiscreteSolution,
        ax: Axes3D,
        xcoord: int = 0,
        ycoord: int = 1,
        zcoord: int = 2,
        trail_length: int = 0,
        skip_last: bool = False,
        projection: Callable[[TimeArray, VectorArray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        **line_kwargs
    ):
        """Animate three coordinates of the solution vectors.

        Every frame, a number of new `(t, y)` points are solved for. These points are projected on the
        x-, y-, and z-axis by plotting `(*y, t)[xcoord]` on the x-axis and likewise for the y- and z-axis.

        Arguments
        ---------
        sol : DiscreteSolution
            Solution as gotten from `ODE.solve()`, for example.
        ax : Axes3D
            Which axes to animate on.
        xcoord, ycoord, zcoord : int (default: 0, 1, 2)
            Which index of the phase vector `(*y, t)` to plot on the x-, y-, and z-axis. The default
            represents plotting the first three coordinates of `y` as x, y, z.
        trail_length : int (default: 0)
            How many points to keep from the end of the line. Default: all.
        **line_kwargs
            Keyword arguments forwarded to `SolutionAnimator.get_line_3d`.
        """
        line = SolutionAnimator.get_line_3d(ax, **line_kwargs)
        updater = SolutionAnimator.phase_diagram_updater_3d(line, xcoord, ycoord, zcoord, trail_length, skip_last, projection)
        resetter = SolutionAnimator.line_resetter_3d(line)
        self.animate(sol, updater, resetter)

    # Internal update functions

    def __get_infinite_update_function(self, load: bool = True) -> Callable[[TimeArray], list[Artist]]:
        """Internal update function capable of running infinitely."""
        def infinite_update_function(time_span: TimeArray) -> list[Artist]:
            t_steps: list[TimeArray] = []
            y_steps: list[VectorArray] = []
            for sol in self.solutions:
                (ts,), (ys,) = sol.get_arrays_between(time_span, load=load, closed_side='right')
                t_steps.append(ts)
                y_steps.append(ys)
            updated_artists = flattened_once(
                update_function(t_steps[solution_index], y_steps[solution_index])
                for update_function, solution_index in self.update_functions
            )
            for no_solution_update_function in self.no_solution_update_functions:
                updated_artists.extend(no_solution_update_function(time_span[-1:]))
            return updated_artists
        return infinite_update_function

    def __get_interval_steps(
        self,
        t_start: Time,
        t_end: Time,
        time_step_per_frame: Time,
        fps: int,
        pauses: list[tuple[Time, Time]]
    ) -> tuple[int, TimeArray]:
        steps = int((t_end - t_start) // time_step_per_frame)
        t_intervals = np.concatenate((np.array([t_start]), np.linspace(t_start, t_end, steps+1))) # Make sure first frame is at t_start
        if not len(pauses):
            return len(t_intervals) - 1, t_intervals  # -1 because it counts the number of intervals
        # At each pause, insert the needed number of frames.
        pause_times, durations = zip(*pauses)
        frames_per_pause = [int(duration * fps) for duration in durations]
        pause_indices = np.searchsorted(t_intervals, np.array(pause_times))
        all_pause_indices = list(chain.from_iterable(repeat(index, num_frames) for index, num_frames in zip(pause_indices, frames_per_pause)))
        all_pause_times = list(chain.from_iterable(repeat(pause_time, num_frames) for pause_time, num_frames in zip(pause_times, frames_per_pause)))
        t_intervals = np.insert(t_intervals, all_pause_indices, all_pause_times)
        return len(t_intervals) - 1, t_intervals

    def __get_interval_update_function(self, t_intervals: TimeArray, load: bool = True) -> Callable[[int], list[Artist]]:
        """Internal update function that uses predefined intervals.

        Update function that plots the exact (t, y) points in the intervals for each
        solution.
        """
        all_solution_points = tuple(sol.get_arrays_between(t_intervals, load=load) for sol in self.solutions)
        solution_ts = tuple(solution_points[0] for solution_points in all_solution_points)
        solution_ys = tuple(solution_points[1] for solution_points in all_solution_points)
        paused_indices = np.diff(t_intervals) < 1e-15
        interval_end_ts: TimeArray = t_intervals[1:]
        interval_end_ys = np.array([sol(interval_end_ts, load=load) for sol in self.solutions])
        def interval_update_function(interval_index: int) -> list[Artist]:
            if paused_indices[interval_index]:
                return []
            updated_artists: list[Artist] = []
            # Update with the exact points first, then one interpolated value
            for update_function, solution_index in self.update_functions:
                updated_artists.extend(
                    update_function(
                        append(solution_ts[solution_index][interval_index], interval_end_ts[interval_index]),
                        append(solution_ys[solution_index][interval_index], interval_end_ys[solution_index, interval_index])
                    )
                )
            for no_solution_update_function in self.no_solution_update_functions:
                updated_artists.extend(
                    no_solution_update_function(
                        interval_end_ts[interval_index : interval_index+1]
                    )
                )
            return updated_artists
        return interval_update_function

    def __get_virtual_time_update_function(self, t_start: Time, t_end: Time, steps_per_frame: int = 1, load: bool = True):
        """Internal update function that takes a fixed number of steps per frame.

        Update function that, without regard for actual t values, updates steps_per_frame steps per frame.
        Only the first animated solution is rendered, which is often undesirable, but has the benefit of
        showing the process of adaptive-step methods better.
        """
        update_function, solution_index = self.update_functions[0]
        solution = self.solutions[solution_index]
        if t_end != np.inf:
            if load:
                solution.load_until(t_end)
            max_index = len(solution.ts)
            frames = range(max_index // steps_per_frame)
        else:
            max_index = None
            frames = count()
        index_offset = int(np.searchsorted(np.array(solution.ts), t_start))

        def virtual_time_update_function(frame: int) -> list[Artist]:
            start_index = index_offset + frame * steps_per_frame
            end_index = start_index + steps_per_frame
            new_ts, new_ys = solution.get_arrays(start_index, min(end_index, max_index) if max_index is not None else end_index, load=load)

            updated_artists = list(update_function(new_ts, new_ys))
            for no_solution_update_function in self.no_solution_update_functions:
                updated_artists.extend(no_solution_update_function(new_ts))
            return updated_artists

        return frames, virtual_time_update_function

    def __get_init_function(self, t_start):
        # init_func takes no arguments and returns all the updated objects from all the init_functions
        def init_function():
            # Get a list of all the objects that each init_function returns
            updated_artists = flattened_once(init_function(t_start) for init_function in self.init_functions)
            return updated_artists
        return init_function

    @cannot_call_after_rendering
    def render(
        self,
        fig: Figure,
        t_start: Time,
        t_end: Time,
        *,
        fps: int = 30,
        speed: float = 1.0,
        pauses: list[tuple[Time, Time]] = [],
        load: bool = True,
        real_time: bool = True,
        steps_per_frame: int = 1,
        repeat: bool = True,
        repeat_delay: int = 2000,
        blit: bool = False,
        filename: str | None = None,
        dpi: int = 150,
    ):
        """Return the finished animation object and save if filename is given.

        If `t_end` is finite, all the `update_functions` are called for all
        solution points between the last frame and the next, as well as the solution
        point at exactly the frame's t value.

        If `t_end` is infinite, all the `update_functions` are called for `steps_per_frame` t values
        between the last frame's and the next's.

        Arguments
        ---------

        fig : Figure
            Which matplotlib Figure to render the animation on.
        t_start : Time
            Which t value to start animating from.
        t_end: Time
            Until which t value to animate before looping or stopping. Can be infinite.
        fps : int (default: 30)
            How many frames to draw per second.
        speed : float (default: 1.0)
            How many time units pass per second. Not used if `real_time` is false.
        pauses : list of tuples like (pause_time, pause_duration)
            List of times and durations of pauses that will be included in the rendered animation.
        load : bool (default: True)
            Whether to load new solution points from the solutions as needed while animating.
        real_time : bool (default: True)
            If false, animates with a fixed number of steps per frame, instead of a fixed amount of time per frame. Only
            animates the first solution.
        steps_per_frame : int (default: 1)
            How many solution points are included in the animation per frame. Only used if `real_time` is false.
        repeat : bool (default: True)
            Whether to repeat the animation when it ends.
        repeat_delay : int (default: 2000)
            How many milliseconds to wait before repeating the animation after ending.
        blit : bool (default: False)
            Whether to animate with blitting. This can speed up animation if few Artists are updated per frame,
            but may also produce artifacts due to zorder caveats. Other than that, there should be no difference
            in the saved animation.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html.
        filename : str or None (default: None)
            If str is supplied, what the animation is saved as. Adds .gif to the end if it doesn't already end with .gif or .mp4. To save
            to mp4, remember to supply the path to ffmpeg using `plt.rc('animation', ffmpeg_path=...)`.
        dpi : int (default: 150)
            Which dpi to save the animation with. Only used if `filename` is supplied.
        """
        t_start, t_end = to_time(t_start), to_time(t_end)
        # Check for errors
        if t_end <= t_start:
            raise ValueError('t_end must be greater than t_start')
        if not isinstance(steps_per_frame, int):
            raise TypeError(f'steps_per_frame must be int, not {type(steps_per_frame).__name__}')
        if steps_per_frame < 1:
            raise ValueError('steps_per_frame must be positive')
        if filename is not None:
            if t_end == np.inf:
                raise ValueError('Cannot save gif when t_end=np.inf')
            if not (filename.endswith('.gif') or filename.endswith('.mp4')):
                filename = filename + '.gif'

        # Get frames and main_update_function
        if real_time:
            time_step_per_frame = speed / fps
            if t_end == np.inf:
                frames = count(start=np.array([t_start - time_step_per_frame, t_start]), step=time_step_per_frame)
                main_update_function = self.__get_infinite_update_function(load)
            else:
                frames, t_intervals = self.__get_interval_steps(t_start, t_end, time_step_per_frame, fps, pauses)
                main_update_function = self.__get_interval_update_function(t_intervals, load)
        else:
            frames, main_update_function = self.__get_virtual_time_update_function(t_start, t_end, steps_per_frame, load)

        animation = FuncAnimation(
            fig,
            func=main_update_function,
            init_func=self.__get_init_function(t_start),
            frames=frames,
            interval=1000/fps,
            cache_frame_data=(t_end != np.inf),
            repeat=repeat,
            repeat_delay=repeat_delay,
            blit=blit
        )
        if filename is not None:
            animation.save(filename, fps=fps, dpi=dpi)

        self.animation = animation  # Save so it (hopefully) isn't garbage collected
        return animation

    def plot_frame(self, t_start: Time, t_end: Time, *, fps: int | None = None, speed=1.0, steps_per_frame=1, load=True):
        """Plot the frame as it looks in the rendered animation at time t.

        If fps is not None, it emulates the behaviour of animating with t_end=np.inf. It then
        uses fps, speed, and steps_per_frame to animate the interpolated points. Otherwise if
        fps is None, it only animates the exact points, without interpolating."""
        # Initialize all
        self.__get_init_function(t_start)()
        # Update all
        if fps is None:
            # Animate exact points
            self.__get_interval_update_function(np.array([t_start, t_end]), load=load)(0)
        else:
            # Animate interpolated points
            frame_step = speed / fps
            frames = int((t_end - t_start) // frame_step)
            t_end = t_start + frames * frame_step # t rounded down to the nearest frame
            t_steps: TimeArray = append(np.linspace(t_start, t_end, num=frames*steps_per_frame), t_end)
            self.__get_infinite_update_function(load=load)(t_steps)

    @staticmethod
    def time_text_updater(text: Text, decimals=2, t_max=np.inf) -> Callable[[TimeArray], tuple[Artist, ...]]:
        """Update function that writes the current time to a text Artist every frame.

        If `t_max` is set, it stops updating past `t_max` and only displays that value.
        """
        def update_time_text(t_steps: TimeArray) -> tuple[Artist, ...]:
            if not len(t_steps):
                return ()
            t = t_steps[-1] if t_steps[-1] < t_max else t_max
            text.set_text(f'$t = {t:.{decimals}f}$')
            return text,
        return update_time_text

    @staticmethod
    def time_text_resetter(text: Text, decimals=2) -> Callable[[Time], tuple[Artist, ...]]:
        """Reset the time text to show the initial time."""
        def reset_time_text(t_start: Time) -> tuple[Artist, ...]:
            text.set_text(f'$t = {t_start:.{decimals}f}$')
            return text,
        return reset_time_text

    @staticmethod
    def phase_diagram_updater(
        line: Line2D,
        xcoord=-1,
        ycoord=0,
        trail_length=0,
        skip_last=False,
        projection: Callable[[TimeArray, VectorArray], tuple[np.ndarray, np.ndarray]] | None = None
    ) -> Callable[[TimeArray, VectorArray], tuple[Artist, ...]]:
        """Update function that plots two coordinates of the y vector each frame.

        Every frame, a number of new `(t, y)` points are solved for. These points are projected on the
        x- and y-axis by plotting `(*y, t)[xcoord]` on the x-axis and `(*y, t)[ycoord]` on the y-axis.

        Arguments
        ---------
        line : Line2D
            Matplotlib line (as gotten from `plt.plot()`, for example) to add points to.
        xcoord, ycoord : int (default: -1, 0)
            Which index of the phase vector `(*y, t)` to plot on the x- and y-axis. The default represents plotting `y[0]` against `t`.
        trail_length : int (default: 0)
            How many points to keep from the end of the line. Default: all.
        skip_last : bool (default: False)
            Whether to skip using the last `(t, y)` point (which is from linear interpolation).
        projection : (TimeArray, VectorArray) -> (xs, ys) (default: None)
            Optional function to skip using `xcoord, ycoord` and instead use a custom projection.
        """
        def update_phase_diagram(t_steps: TimeArray, y_steps: VectorArray) -> tuple[Artist, ...]:
            if skip_last:
                t_steps = t_steps[:-1]
                y_steps = y_steps[:-1]
            if len(t_steps) == 0:
                return ()
            if projection is None:
                new_xdata = y_steps[:, xcoord] if xcoord != -1 else t_steps
                new_ydata = y_steps[:, ycoord] if ycoord != -1 else t_steps
            else:
                new_xdata, new_ydata = projection(t_steps, y_steps)
            xdata = np.concatenate((line.get_xdata(), new_xdata))
            ydata = np.concatenate((line.get_ydata(), new_ydata))
            line.set_xdata(xdata[-trail_length:] if trail_length else xdata)
            line.set_ydata(ydata[-trail_length:] if trail_length else ydata)
            return line,
        return update_phase_diagram

    @staticmethod
    def phase_diagram_updater_3d(
        line: Line3D,
        xcoord=0,
        ycoord=1,
        zcoord=2,
        trail_length=0,
        skip_last=False,
        projection: Callable[[TimeArray, VectorArray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ) -> Callable[[TimeArray, VectorArray], tuple[Artist, ...]]:
        """Update function that plots three coordinates of the y vector each frame.

        Every frame, a number of new `(t, y)` points are solved for. These points are projected on the
        x-, y- and z-axis by plotting `(*y, t)[xcoord]` on the x-axis, and likewise for the y- and z-axis.

        Arguments
        ---------
        line : Line3D
            Matplotlib line (as gotten from `plt.plot()`, for example) to add points to.
        xcoord, ycoord, zcoord : int (default: 0, 1, 2)
            Which index of the phase vector `(*y, t)` to plot on the x-, y-, and z-axis. The default
            represents plotting the first three coordinates of `y` as x, y, z.
        trail_length : int (default: 0)
            How many points to keep from the end of the line. Default: all.
        skip_last : bool (default: False)
            Whether to skip using the last `(t, y)` point (which is from linear interpolation).
        projection : (TimeArray, VectorArray) -> (xs, ys, zs) (default: None)
            Optional function to skip using `xcoord, ycoord, zcoord` and instead use a custom projection.
        """
        def update_phase_diagram(t_steps: TimeArray, y_steps: VectorArray) -> tuple[Artist, ...]:
            if skip_last:
                t_steps = t_steps[:-1]
                y_steps = y_steps[:-1]
            if len(t_steps) == 0:
                return ()
            if projection is None:
                new_xdata = y_steps[:, xcoord] if xcoord != -1 else t_steps
                new_ydata = y_steps[:, ycoord] if ycoord != -1 else t_steps
                new_zdata = y_steps[:, zcoord] if zcoord != -1 else t_steps
            else:
                new_xdata, new_ydata, new_zdata = projection(t_steps, y_steps)
            new_data = np.array((new_xdata, new_ydata, new_zdata))
            data = np.concatenate((np.asarray(line.get_data_3d()), new_data), axis=1)
            line.set_data_3d(data[:, -trail_length:] if trail_length else data)
            return line,
        return update_phase_diagram

    @staticmethod
    def scalar_function_updater(
        line: Line2D,
        scalar_function: Callable[[TimeArray, VectorArray], tuple[TimeArray, np.ndarray]],
    ) -> Callable[[TimeArray, VectorArray], tuple[Artist, ...]]:
        """Update function that plots a scalar across the time.

        Arguments
        ---------
        line : Line2D
            Matplotlib line (as gotten from `plt.plot()`, for example) to add points to.
        scalar_function : (TimeArray, VectorArray) -> (ts, values)
            Function that takes the new `(t, y)` values from the current frame and returns one array of times and
            one of scalar values corresponding the times. Example: `lambda ts, ys: (ts, np.linalg.norm(ys, axis=1))`.
        """
        def update_scalar_function(t_steps: TimeArray, y_steps: VectorArray) -> tuple[Artist, ...]:
            returned_ts, function_values = scalar_function(t_steps, y_steps)
            if not len(returned_ts):
                return ()
            line.set_xdata(np.concatenate((line.get_xdata(), returned_ts)))
            line.set_ydata(np.concatenate((line.get_ydata(), function_values)))
            return line,
        return update_scalar_function

    @staticmethod
    def line_resetter(line: Line2D) -> Callable[[Time], tuple[Artist, ...]]:
        """Initialization function that resets a Line2D."""
        def init_line(t_start) -> tuple[Artist, ...]:
            line.set_xdata([])
            line.set_ydata([])
            return line,
        return init_line

    @staticmethod
    def line_resetter_3d(line: Line3D) -> Callable[[Time], tuple[Artist, ...]]:
        """Initialization function that resets a Line3D."""
        def init_line(t_start) -> tuple[Artist, ...]:
            line.set_data_3d(np.empty((3, 0)))
            return line,
        return init_line

    @staticmethod
    def get_time_text(ax: Axes | Axes3D, loc='upper left') -> Text:
        locations = {
            "upper left": (0.05, 0.95, "left", "top"),
            "lower left": (0.05, 0.05, "left", "bottom"),
            "lower right": (0.95, 0.05, "right", "bottom"),
            "upper right": (0.95, 0.95, "right", "top"),
        }
        x, y, ha, va = locations[loc]
        if isinstance(ax, Axes3D):
            return ax.text2D(x, y, '', ha=ha, va=va, transform=ax.transAxes)
        return ax.text(x, y, '', ha=ha, va=va, transform=ax.transAxes)

    @staticmethod
    def get_line(ax: Axes | None = None, style='o-', markevery=[-1], **kwargs) -> Line2D:
        if ax is None:
            ax = plt.gca()
        return ax.plot([], [], style, markevery=markevery, **kwargs)[0]

    @staticmethod
    def get_line_3d(ax: Axes3D, style='o-', markevery=[-1], **kwargs) -> Line3D:
        return ax.plot([], [], [], style, markevery=markevery, **kwargs)[0]  # type: ignore (matplotlib thinks it returns a Line2D)


"""
TODO:
+ Write docstrings
- Make infinite animation work with pauses
- Make infinite animation work with exact solution points
"""
