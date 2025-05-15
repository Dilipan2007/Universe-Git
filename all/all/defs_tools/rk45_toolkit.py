import sympy
import numpy as np
from scipy.integrate import solve_ivp
import defs_tools.modifications as md
from typing import Sequence


def make_rk_sys(
    expr: sympy.Function, variables: Sequence[sympy.Function], t: sympy.Symbol, order=2
):
    u = np.array(
        [
            np.array([sympy.symbols(f"u_{i}_{j}]") for j in range(order)])
            for i in range(3)
        ]
    )  # Creates (u0, u1, ..., u_{n-1})
    # variables=[x,y,z]
    # Substitution map: y(t), y'(t), ... => u0, u1, ...
    rhs = []
    sub = {}
    for i in range(3):
        subs = {sympy.diff(variables[i], t, j): u[i, j] for j in range(order)}
        subs[variables[i]] = u[i, 0]
        sub.update(subs)
    for i in range(3):
        # Construct the system: dy/dt = [u1, u2, ..., f(t, u0,...,u_{n-1})]
        rhs_exprs = list(u[i, 1:])  # u1, u2, ..., u_{n-2}
        rhs_exprs.append(expr[i].subs(sub))  # u_{n-1}' = f(t, u0,...)

        rhs.append(rhs_exprs)
    # Convert to numeric function
    rhs_func = sympy.lambdify((t, md.col_flatten(u)), rhs, "numpy")

    # Return function suitable for solve_ivp
    def sys(t, y):
        return np.array(rhs_func(t, y), dtype=float).flatten()

    return sys


def solve(
    system,
    ix,
    iv,
    itime,
    ftime,
):
    t = (itime, ftime)
    sol = solve_ivp(system, t, [ix, iv, itime], method="RK45")

    return sol
