from mechanics.classical import eq_of_motion
from defs.vectors import v3
from core.particle import particle
import sympy


p = v3(0, 0, 0)
v = v3(1, 1, 1)
w = v3(0, 0, 0)
m = 5
r = 1

part = particle(5, p, v, w, r)
t = sympy.symbols("t")
x = sympy.Function("x pos")(t)
V = sympy.Function("pot")(x)
V = x**2
eq_of_motion(part, V)
