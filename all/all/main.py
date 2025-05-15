from defs_tools.vectors import v3
from core.particle import particle
import sympy
import numpy as np
from mechanics import classical
import matplotlib.pyplot as plt


p = v3(0, 0, 0)
v = v3(1, 1, 1)
w = v3(0, 0, 0)
m = 5
r = 1

part = particle(5, p, v, w, r)
n = 1  # mass=1, pos=(0,1,2) vel=(1,0,0), w=(0,0,0), r=1
particles = np.array(
    [particle(i + 1, (0, 0, 0), (1, 0, 0), (0, 0, 0), 1) for i in range(n)]
)
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

x = sympy.Function("x pos")(t)
V = sympy.Function("pot")(*(p_vectors.flatten() + v_vectors.flatten()))
V = 0
test = classical.evolve(particles, V, 0, 10, p_vectors, v_vectors, t)
yo = test.run(n=10)
plt.scatter(yo[1], yo[0])
