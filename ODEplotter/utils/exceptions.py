class ODEStopSolvingException(Exception):
    pass

class StepSizeTooSmallError(ODEStopSolvingException):
    pass

class ObstacleStopSolving(ODEStopSolvingException):
    pass
