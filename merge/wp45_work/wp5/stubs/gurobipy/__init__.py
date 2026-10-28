"""Minimal stand-in for gurobipy, for local checks only.

It lets ``base.py`` be imported on a computer without Gurobi, so that its
pure-Python functions (no Gurobi object involved) can be called. Any attempt
to build or solve a model raises ``RuntimeError``.
"""


class GurobiError(Exception):
    pass


class _GRB:
    BINARY = "B"
    CONTINUOUS = "C"
    INTEGER = "I"
    MINIMIZE = 1
    MAXIMIZE = -1
    INFINITY = 1e100
    UNDEFINED = 1e101
    OPTIMAL = 2
    INFEASIBLE = 3
    INF_OR_UNBD = 4
    UNBOUNDED = 5
    ITERATION_LIMIT = 7
    NODE_LIMIT = 8
    TIME_LIMIT = 9
    SOLUTION_LIMIT = 10
    INTERRUPTED = 11
    SUBOPTIMAL = 13
    GENCONSTR_INDICATOR = 7

    class Callback:
        MIP = 3
        RUNTIME = 6001
        MIP_OBJBST = 3000
        MIP_OBJBND = 3001
        MIP_NODCNT = 3002
        MIP_SOLCNT = 3003


GRB = _GRB


class tupledict(dict):
    pass


class _NoGurobi:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("gurobipy stub: no model can be built on this computer")


Model = _NoGurobi
Var = _NoGurobi
LinExpr = _NoGurobi
QuadExpr = _NoGurobi


def quicksum(items):
    raise RuntimeError("gurobipy stub: quicksum needs Gurobi")


class gurobi:
    @staticmethod
    def version():
        return (0, 0, 0)
