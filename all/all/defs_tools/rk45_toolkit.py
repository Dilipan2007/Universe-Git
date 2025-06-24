import sympy
import numpy as np
from scipy.integrate import solve_ivp
import defs_tools.modifications as md
from typing import Sequence


def make_rk_sys(
    expr: Sequence[sympy.Function],
    variables: Sequence[Sequence[sympy.Function]],
    t: sympy.Symbol,
    order=2,
):

    # k for particle, i for axis, j for degree
    u = np.array(
        [
            np.array(
                [
                    np.array(
                        [sympy.symbols(f"u_{k}_{i}_{j}") for j in range(order)]
                    )  # k for kth particle
                    for i in range(3)
                ]
            )
            for k in range(len(expr))
        ]
    )
    # Creates [[[u_0 for velocity, u_1 for acceleraruion]* three coordinates]* N paerticles]
    # variables=[[x_k,y_k,z_k]* for N paerticles]
    dy = []  # symbolliacly dy=d(y)/dt
    sub = {}
    y = (
        []
    )  # to have [..., u_k_0(i)_0, u_k_1_0, u_k_2_0, u_k_0_1, u_k_1_1, u_k_2_1,... for n particles]
    for k in range(len(expr)):
        for i in range(3):
            subs = {sympy.diff(variables[k, i], t, j): u[k, i, j] for j in range(order)}
            sub.update(subs)

    for k in range(len(expr)):
        v = []
        a = []
        y.extend(*list([md.col_flatten(u[k])]))  # flattening
        for i in range(3):
            v.append(*u[k, i, 1:])
            a.append(expr[k, i].subs(sub))

        dy.extend(v + a)
    # Convert to numeric function
    rhs_func = sympy.lambdify((t, *y), dy, "numpy")

    # Return function suitable for solve_ivp
    def sys(t, y):
        return np.array(rhs_func(t, *y), dtype=float)

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
