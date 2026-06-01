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

AB1 = AdamsBashforth(AB1_WEIGHTS)
AB2 = AdamsBashforth(AB2_WEIGHTS)
AB3 = AdamsBashforth(AB3_WEIGHTS)
AB4 = AdamsBashforth(AB4_WEIGHTS)
AB5 = AdamsBashforth(AB5_WEIGHTS)

# Implicit linear multistep methods

IMPLICIT_EULER = ImplicitEulersMethod()

TRAPEZOIDAL_RULE = TrapezoidalRule()
AM0 = AdamsMoulton(AM0_WEIGHTS, predictor_weights=AB1_WEIGHTS)
AM1 = AdamsMoulton(AM1_WEIGHTS, predictor_weights=AB2_WEIGHTS)
AM2 = AdamsMoulton(AM2_WEIGHTS, predictor_weights=AB3_WEIGHTS)
AM3 = AdamsMoulton(AM3_WEIGHTS, predictor_weights=AB4_WEIGHTS)
AM4 = AdamsMoulton(AM4_WEIGHTS, predictor_weights=AB5_WEIGHTS)

BDF1 = BackwardDifferentialFormula(BDF1_Y_WEIGHTS, BDF1_DERIVATIVE_WEIGHT)
BDF2 = BackwardDifferentialFormula(BDF2_Y_WEIGHTS, BDF2_DERIVATIVE_WEIGHT)
BDF3 = BackwardDifferentialFormula(BDF3_Y_WEIGHTS, BDF3_DERIVATIVE_WEIGHT)
BDF4 = BackwardDifferentialFormula(BDF4_Y_WEIGHTS, BDF4_DERIVATIVE_WEIGHT)
BDF5 = BackwardDifferentialFormula(BDF5_Y_WEIGHTS, BDF5_DERIVATIVE_WEIGHT)
BDF6 = BackwardDifferentialFormula(BDF6_Y_WEIGHTS, BDF6_DERIVATIVE_WEIGHT)

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
    "euler": ["explicit_euler", "forward_euler"],
    "implicit_euler": ["backward_euler"],
    "heun": ["rkh2"],
    "ralston": ["rkr2"],
    "trapezoidal_rule": ["trapezoidal", "trapezoid"],
}

# Add the aliased names to solution_methods as well
for method_name, aliases in ALIASES.items():
    method = METHODS[method_name]
    METHODS.update({alias: method for alias in aliases})
