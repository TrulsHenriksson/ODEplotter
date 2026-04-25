from .method_data import *

from .solution_method import SolutionMethod
from .linear_multistep.euler import EulersMethod
from .linear_multistep.implicit_euler import ImplicitEulersMethod
from .linear_multistep.trapezoidal_rule import TrapezoidalRule
from .linear_multistep.adams_bashforth import AdamsBashforth
from .linear_multistep.adams_moulton import AdamsMoulton
from .linear_multistep.bdf import BackwardDifferentialFormula
from .runge_kutta.rk4 import RungeKutta4
from .runge_kutta.runge_kutta import RungeKutta
from .runge_kutta.adaptive_runge_kutta import AdaptiveRungeKutta
from .runge_kutta.adaptive_runge_kutta_PI import AdaptiveRungeKuttaPI
from .runge_kutta.rk43 import RungeKutta43


__all__ = ["METHODS"]


# Explicit linear multistep methods

EULER = EulersMethod()

AB1 = AdamsBashforth([1.0])
AB2 = AdamsBashforth([3/2, -1/2])
AB3 = AdamsBashforth([23/12, -16/12, 5/12])
AB4 = AdamsBashforth([55/24, -59/24, 37/24, -9/24])
AB5 = AdamsBashforth([1901/720, -2774/720, 2616/720, -1274/720, 251/720])

# Implicit linear multistep methods

IMPLICIT_EULER = ImplicitEulersMethod()

TRAPEZOIDAL_RULE = TrapezoidalRule()
AM0 = AdamsMoulton([1.0], predictor="AB1")
AM1 = AdamsMoulton([0.5, 0.5], predictor="AB2")
AM2 = AdamsMoulton([5/12, 8/12, -1/12], predictor="AB3")
AM3 = AdamsMoulton([9/24, 19/24, -5/24, 1/24], predictor="AB4")
AM4 = AdamsMoulton([251/720, 646/720, -264/720, 106/720, -19/720], predictor="AB5")

BDF1 = BackwardDifferentialFormula([1.0], 1.0)
BDF2 = BackwardDifferentialFormula([4/3, -1/3], 2/3)
BDF3 = BackwardDifferentialFormula([18/11, -9/11, 2/11], 6/11)
BDF4 = BackwardDifferentialFormula([48/25, -36/25, 16/25, -3/25], 12/25)
BDF5 = BackwardDifferentialFormula([300/137, -300/137, 200/137, -75/137, 12/137], 60/137)
BDF6 = BackwardDifferentialFormula([360/147, -450/147, 400/147, -225/147, 72/147, -10/147], 60/147)

# Fixed-step Runge-Kutta methods

HEUN = RungeKutta(HEUN_NODES, HEUN_WEIGHTS, HEUN_MATRIX)
MIDPOINT = RungeKutta(MIDPOINT_NODES, MIDPOINT_WEIGHTS, MIDPOINT_MATRIX)
RALSTON = RungeKutta(RALSTON_NODES, RALSTON_WEIGHTS, RALSTON_MATRIX)
RK3 = RungeKutta(RK3_NODES, RK3_WEIGHTS, RK3_MATRIX)
RK4 = RungeKutta4()
RK4_38 = RungeKutta(THREE_EIGHTS_NODES, THREE_EIGHTS_WEIGHTS, THREE_EIGHTS_MATRIX)
RKH10 = RungeKutta(RKH10_NODES, RKH10_WEIGHTS, RKH10_MATRIX)
RKZ10 = RungeKutta(RKZ10_NODES, RKZ10_WEIGHTS, RKZ10_MATRIX)
RK12 = RungeKutta(RKFEAGIN12_NODES, RKFEAGIN12_WEIGHTS, RKFEAGIN12_MATRIX)

# Adaptive-step Runge-Kutta methods

# TODO: Check that these orders are accurate
HEUN_EULER = AdaptiveRungeKutta(HEUN_EULER_NODES, HEUN_EULER_WEIGHTS, HEUN_EULER_MATRIX, HEUN_EULER_ERROR, order=2)
BOGACKI_SHAMPINE = AdaptiveRungeKutta(BOGACKI_SHAMPINE_NODES, BOGACKI_SHAMPINE_WEIGHTS, BOGACKI_SHAMPINE_MATRIX, BOGACKI_SHAMPINE_ERROR, order=3)
RK43 = RungeKutta43()
RKF = AdaptiveRungeKutta(RKF_NODES, RKF_WEIGHTS, RKF_MATRIX, RKF_ERROR, order=4)
RK45 = AdaptiveRungeKutta(RK45_NODES, RK45_WEIGHTS, RK45_MATRIX, RK45_ERROR, order=5)
RKCK = AdaptiveRungeKutta(RKCK_NODES, RKCK_WEIGHTS, RKCK_MATRIX, RKCK_ERROR, order=5)
DOPRI = AdaptiveRungeKutta(DOPRI_NODES, DOPRI_WEIGHTS, DOPRI_MATRIX, DOPRI_ERROR, order=5)
DVERK = AdaptiveRungeKutta(DVERK_NODES, DVERK_WEIGHTS, DVERK_MATRIX, DVERK_ERROR, order=6)
RKF78 = AdaptiveRungeKutta(RKF78_NODES, RKF78_WEIGHTS, RKF78_MATRIX, RKF78_ERROR, order=7)
RK10_8 = AdaptiveRungeKutta(RK10_8_NODES, RK10_8_WEIGHTS, RK10_8_MATRIX, RK10_8_ERROR, order=10)
RK12_10 = AdaptiveRungeKutta(RK12_10_NODES, RK12_10_WEIGHTS, RK12_10_MATRIX, RK12_10_ERROR, order=12)
RK14_12 = AdaptiveRungeKutta(RK14_12_NODES, RK14_12_WEIGHTS, RK14_12_MATRIX, RK14_12_ERROR, order=14)


METHODS: dict[str, SolutionMethod] = {
    "euler": EULER,
    "ab1": AB1,
    "ab2": AB2,
    "ab3": AB3,
    "ab4": AB4,
    "ab5": AB5,
    "implicit_euler": IMPLICIT_EULER,
    "trapezoidal_rule": TRAPEZOIDAL_RULE,
    "am0": AM0,
    "am1": AM1,
    "am2": AM2,
    "am3": AM3,
    "am4": AM4,
    "bdf1": BDF1,
    "bdf2": BDF2,
    "bdf3": BDF3,
    "bdf4": BDF4,
    "bdf5": BDF5,
    "bdf6": BDF6,
    "heun": HEUN,
    "midpoint": MIDPOINT,
    "ralston": RALSTON,
    "rk3": RK3,
    "rk4": RK4,
    "rk4_38": RK4_38,
    "rkh10": RKH10,
    "rkz10": RKZ10,
    "rk12": RK12,
    "heun_euler": HEUN_EULER,
    "bogacki_shampine": BOGACKI_SHAMPINE,
    "rkf": RKF,
    "rk43": RK43,
    "rk45": RK45,
    "rkck": RKCK,
    "dopri": DOPRI,
    "dverk": DVERK,
    "rkf78": RKF78,
    "rk10_8": RK10_8,
    "rk12_10": RK12_10,
    "rk14_12": RK14_12,
}

ALIASES: dict[str, list[str]] = {
    "euler": ["explicit_euler"],
    "heun": ["rkh2"],
    "ralston": ["rkr2"],
    "trapezoidal_rule": ["trapezoidal", "trapezoid"],
}

# Add the aliased names to solution_methods as well
for method_name, aliases in ALIASES.items():
    method = METHODS[method_name]
    METHODS.update({alias: method for alias in aliases})
