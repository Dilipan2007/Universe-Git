from defs_tools.vectors import v3
from core.particle import particle
import sympy
import numpy as np
from mechanics import classical
import matplotlib.pyplot as plt


p = v3(0, 0, 0)
v = v3(1, 1, 1)
w = v3(0, 0, 0)
m = 5.9722e24
M = 1.989e30
G = 6.67430e-11
r = 1

part = particle(5, p, v, w, r)
n = 2  # mass=1, pos=(0,1,2) vel=(1,0,0), w=(0,0,0), r=1
particles = np.array(
    [particle(i + 1, v3(0, 0, 0), v3(0, -10, 0), v3(0, 0, 0), 1) for i in range(n)]
)
p1 = particle(
    1,
    v3(0, 1, 0),
    v3(1, -2, 0),
    v3(7.2921159 * 10 ** (-5), np.pi, 0.4091, "p"),
    0.1,
)  # for earth inn summer
p2 = particle(1, v3(0, 0, 0), v3(1, 1, 0), p, 0.1)
t = sympy.symbols("time")
p_vectors = np.array(
    [
        np.array([sympy.Function(f"r_{partic}_{axis}")(t) for axis in range(1, 4)])
        for partic in range(1, n + 1)
    ]
)
v_vectors = np.array(
    [
        np.array([sympy.diff(p_vectors[partic, axis]) for axis in range(3)])
        for partic in range(n)
    ]
)


V = (p_vectors[0, 1] - p_vectors[1, 1]) ** 2
test = classical.evolve([p1, p2], V, 0, 10, p_vectors, v_vectors, t)
yo = test.run(n=1000)
plt.scatter(yo[0][0], yo[0][1], s=0.5)
plt.scatter(yo[1][0], yo[1][1], s=0.5)
plt.show()
