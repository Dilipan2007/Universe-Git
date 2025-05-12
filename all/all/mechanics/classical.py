from defs.vectors import v3
from core.particle import particle
import sympy
import numpy as np


def eq_of_motion(
    partic: particle, V: sympy.Function, generalized_coordinates="cartesian"
):  # potential is symbolic func
    if generalized_coordinates == "cartesian":
        m, i, t = sympy.symbols("mass inertia time")
        x = sympy.Function("x pos")(t)
        y = sympy.Function("y pos")(t)
        z = sympy.Function("z pos")(t)
        v1 = sympy.diff(x, t)
        v2 = sympy.diff(y, t)
        v3 = sympy.diff(z, t)
        av1 = sympy.Function("angular velocity x")(t)
        av2 = sympy.Function("angular velocity y")(t)
        av3 = sympy.Function("angular velocity z")(t)
        v = sympy.Matrix([v1, v2, v3])
        av = sympy.Matrix([av1, av2, av3])
        T = (m * (v.dot(v)) + i * (av.dot(av))) / 2
        L = T - V
        eq1 = sympy.diff(L, t, v1) + sympy.diff(L, x)
        eq2 = sympy.diff(L, t, v2) + sympy.diff(L, y)
        eq3 = sympy.diff(L, t, v3) + sympy.diff(L, z)
        # following are specific
        x_acc = sympy.solve(eq1, sympy.diff(x, t, 2))
        y_acc = sympy.solve(eq2, sympy.diff(y, t, 2))
        z_acc = sympy.solve(eq3, sympy.diff(z, t, 2))
        return np.array([x_acc, y_acc, z_acc])
