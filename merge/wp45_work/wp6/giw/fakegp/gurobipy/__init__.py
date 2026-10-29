"""A tiny linear emulator of gurobipy, for local row checks only (not Gurobi).

It records variables, linear rows and indicator rows; ``optimize`` solves the
LP that remains when every binary variable is fixed (LB == UB), with scipy.
Written for the executor of WP5 Task A; lives in the scratchpad, never in the
repository.

WP6 GIW copy (merge/wp6_plan_giw.md): products of two linear expressions and
``addQConstr`` are recorded, so that a model with quadratic rows can be built
and inspected; ``optimize`` refuses such a model with NotImplementedError
("needs Gurobi"). ``getConstrs`` and ``getQConstrs`` return objects with
``ConstrName``, ``RHS``, ``Sense`` and ``QCName``, like Gurobi.
"""
import itertools
import math

import numpy as np
from scipy.optimize import linprog


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


def _expr(obj):
    if isinstance(obj, LinExpr):
        return obj
    if isinstance(obj, Var):
        return LinExpr({obj: 1.0}, 0.0)
    return LinExpr({}, float(obj))


class _Arith:
    def __add__(self, other):
        if isinstance(other, QuadExpr):
            return other + self
        a, b = _expr(self), _expr(other)
        terms = dict(a.terms)
        for v, c in b.terms.items():
            terms[v] = terms.get(v, 0.0) + c
        return LinExpr(terms, a.const + b.const)

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        if isinstance(other, QuadExpr):
            return self + (-other)
        return self + (-_expr(other))

    def __rsub__(self, other):
        return _expr(other) + (-_expr(self))

    def __mul__(self, other):
        if isinstance(other, QuadExpr):
            raise NotImplementedError("fake gurobipy: no cubic terms")
        if isinstance(other, (Var, LinExpr)):
            return QuadExpr(LinExpr(), [(1.0, _expr(self), _expr(other))])
        k = float(other)
        a = _expr(self)
        return LinExpr({v: c * k for v, c in a.terms.items()}, a.const * k)

    __rmul__ = __mul__

    def __le__(self, other):
        if isinstance(other, QuadExpr):
            return TempConstr(QuadExpr(_expr(self)) - other, "<")
        return TempConstr(_expr(self) - _expr(other), "<")

    def __ge__(self, other):
        if isinstance(other, QuadExpr):
            return TempConstr(QuadExpr(_expr(self)) - other, ">")
        return TempConstr(_expr(self) - _expr(other), ">")

    def __eq__(self, other):
        return TempConstr(_expr(self) - _expr(other), "=")

    __hash__ = object.__hash__


class QuadExpr:
    """lin + sum of coef * a * b, with a and b linear expressions (recorded, not solved)."""

    def __init__(self, lin=None, products=None):
        self.lin = LinExpr() if lin is None else lin
        self.products = [] if products is None else products

    def __add__(self, other):
        if isinstance(other, QuadExpr):
            return QuadExpr(self.lin + other.lin, self.products + other.products)
        return QuadExpr(self.lin + other, list(self.products))

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, (Var, LinExpr, QuadExpr)):
            raise NotImplementedError("fake gurobipy: no cubic terms")
        k = float(other)
        return QuadExpr(self.lin * k, [(c * k, a, b) for c, a, b in self.products])

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __le__(self, other):
        return TempConstr(self - other, "<")

    def __ge__(self, other):
        return TempConstr(self - other, ">")


class LinExpr(_Arith):
    def __init__(self, terms=None, const=0.0):
        self.terms = {} if terms is None else terms
        self.const = float(const)


class Var(_Arith):
    def __init__(self, idx, lb, ub, vtype, name):
        self.idx, self.LB, self.UB, self.VType, self.VarName = idx, lb, ub, vtype, name
        self.X = None
        self.Start = None


class TempConstr:
    def __init__(self, expr, sense):
        self.expr, self.sense = expr, sense


def quicksum(items):
    total = LinExpr()
    for item in items:
        total = total + item
    return total


class _Params:
    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)


class Model:
    def __init__(self, name=""):
        self.ModelName = name
        self.Params = _Params()
        self.Params.OutputFlag = 0
        self.Params.TimeLimit = 1e100
        self.Params.MIPGap = 1e-4
        self.Params.Threads = 0
        self.Params.Seed = 0
        self.vars = []
        self.rows = []
        self.qrows = []
        self.indicators = []
        self.objective = LinExpr()
        self.Status = None
        self.SolCount = 0
        self.ObjVal = None
        self.ObjBound = None
        self.MIPGap = 0.0
        self.Runtime = 0.0
        self.NodeCount = 0

    def setParam(self, key, value):
        setattr(self.Params, key, value)

    def _add_var(self, lb, ub, vtype, name):
        v = Var(len(self.vars), lb, ub, vtype, name)
        self.vars.append(v)
        return v

    def addVar(self, lb=0.0, ub=1e100, vtype="C", name=""):
        if vtype == "B":
            lb, ub = 0.0, 1.0
        return self._add_var(lb, ub, vtype, name)

    def addVars(self, *dims, lb=0.0, ub=1e100, vtype="C", name=""):
        if len(dims) == 1 and not isinstance(dims[0], int):
            keys = [k if isinstance(k, tuple) else (k,) for k in dims[0]]
        else:
            keys = list(itertools.product(*[range(d) for d in dims]))
        out = tupledict()
        for key in keys:
            label = f"{name}[{','.join(str(i) for i in key)}]"
            out[key if len(key) > 1 else key[0]] = self.addVar(lb, ub, vtype, label)
        return out

    def addConstr(self, tc, name=""):
        assert isinstance(tc, TempConstr), "fake gurobipy: addConstr needs a linear row"
        assert not isinstance(tc.expr, QuadExpr), "fake gurobipy: use addQConstr for a quadratic row"
        self.rows.append((tc, name))
        return tc

    def addQConstr(self, tc, name=""):
        assert isinstance(tc, TempConstr) and isinstance(tc.expr, QuadExpr)
        self.qrows.append((tc, name))
        return tc

    def addGenConstrIndicator(self, binvar, binval, tc, name=""):
        assert isinstance(tc, TempConstr)
        assert not isinstance(tc.expr, QuadExpr), "fake gurobipy: indicator rows are linear"
        self.indicators.append((binvar, bool(binval), tc, name))

    def setObjective(self, expr, sense=1):
        self.objective = _expr(expr)
        self.sense = sense

    def update(self):
        pass

    def dispose(self):
        pass

    def getConstrs(self):
        out = []
        for tc, name in self.rows:
            sense = {"<": "<", ">": ">", "=": "="}[tc.sense]
            out.append(type("Constr", (), {"ConstrName": name, "RHS": -tc.expr.const,
                                           "Sense": sense})())
        return out

    def getQConstrs(self):
        return [type("QConstr", (), {"QCName": name})() for _, name in self.qrows]

    def getVars(self):
        return list(self.vars)

    # --- size attributes, as Gurobi reads them after update() ---
    NumVars = property(lambda self: len(self.vars))
    NumIntVars = property(lambda self: sum(v.VType in ("B", "I") for v in self.vars))
    NumBinVars = property(lambda self: sum(v.VType == "B" for v in self.vars))
    NumConstrs = property(lambda self: len(self.rows))
    NumGenConstrs = property(lambda self: len(self.indicators))
    NumQConstrs = property(lambda self: len(self.qrows))
    NumNZs = property(lambda self: sum(len(tc.expr.terms) for tc, _ in self.rows))
    IterCount = 0.0
    BarIterCount = 0.0
    Work = 0.0

    def getGenConstrs(self):
        return [type("GenConstr", (), {"GenConstrType": GRB.GENCONSTR_INDICATOR})() for _ in self.indicators]

    def optimize(self, callback=None):
        if self.qrows:
            raise NotImplementedError("fake gurobipy: quadratic rows cannot be solved; needs Gurobi")
        n = len(self.vars)
        for v in self.vars:
            if v.VType == "B" and v.LB != v.UB:
                if v.Start is None:
                    raise RuntimeError(f"fake gurobipy: binary {v.VarName} is free and has no Start; fix it first")
                v.LB = v.UB = float(v.Start)        # emulation: a complete MIP start fixes the binaries
        cons = [tc for tc, _ in self.rows]
        for binvar, binval, tc, _ in self.indicators:
            if (binvar.LB > 0.5) == binval:
                cons.append(tc)
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for tc in cons:
            row = np.zeros(n)
            for v, c in tc.expr.terms.items():
                row[v.idx] += c
            rhs = -tc.expr.const
            if tc.sense == "<":
                A_ub.append(row); b_ub.append(rhs)
            elif tc.sense == ">":
                A_ub.append(-row); b_ub.append(-rhs)
            else:
                A_eq.append(row); b_eq.append(rhs)
        c = np.zeros(n)
        for v, k in self.objective.terms.items():
            c[v.idx] += k
        if getattr(self, "sense", 1) == -1:
            c = -c
        bounds = [(v.LB, None if v.UB >= 1e99 else v.UB) for v in self.vars]
        res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=b_ub or None,
                      A_eq=np.array(A_eq) if A_eq else None, b_eq=b_eq or None,
                      bounds=bounds, method="highs")
        if res.status == 0:
            self.Status = GRB.OPTIMAL
            self.SolCount = 1
            for v, val in zip(self.vars, res.x):
                v.X = float(val)
            self.ObjVal = float(res.fun if getattr(self, "sense", 1) == 1 else -res.fun) + self.objective.const
            self.ObjBound = self.ObjVal
        else:
            self.Status = GRB.INFEASIBLE if res.status == 2 else GRB.INF_OR_UNBD
            self.SolCount = 0
            for attribute in ("ObjVal", "ObjBound", "MIPGap"):
                self.__dict__.pop(attribute, None)    # like Gurobi: no value without a solution


class gurobi:
    @staticmethod
    def version():
        return (0, 0, 0)
