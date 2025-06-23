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
n = 1  # mass=1, pos=(0,1,2) vel=(1,0,0), w=(0,0,0), r=1
particles = np.array(
    [particle(i + 1, v3(0, 0, 0), v3(0, -10, 0), v3(0, 0, 0), 1) for i in range(n)]
)
p1 = particle(
    m,
    v3(1.519 * (10**11), 0, 0),
    v3(181, 29300, 0),
    v3(7.2921159 * 10 ** (-5), np.pi, 0.4091, "p"),
    0.2,
)  # for earth inn summer
p2 = particle(1, v3(10, -10, 0), v3(-1, 1, 0), p, 0.2)
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


V = sympy.Function("pot")(*(p_vectors.flatten() + v_vectors.flatten()))
# earth data

V = (-G * M * m) * (
    p_vectors[0, 0] ** 2 + p_vectors[0, 1] ** 2 + p_vectors[0, 2] ** 2
) ** (-1 / 2)
test = classical.evolve([p1], V, 0, 700 * 24 * 60 * 60, p_vectors, v_vectors, t)
yo = test.run(n=700)


def fun(i, j, o):
    if i in [0, 2] and j in [0, 2] and i == j and o == 0:
        return 1
    elif i in [0, 2] and j in [0, 2] and i != j and o == 1:
        return (-1) ** (i / 2)
    elif i == 1 and j == 1 and o == 2:
        return 1
    else:
        return 0


# For analemma


def T(v):
    a = v3(1, np.pi / 2 - 0.4091, 0, "p")
    b = v3(0, 1, 0)
    c = v3(1, 0.4091, np.pi, "p")
    li = a * v.v1 + b * v.v2 + c * v.v3
    return li


def Q(v, t, y=0):
    c = v3(1, (np.pi / 2) - y, v3.norm(p1.avel) * t + np.pi, "p")
    b = v3(1, np.pi / 2, v3.norm(p1.avel) * t + np.pi / 2, "p")
    a = v3(1, y, v3.norm(p1.avel) * t, "p")
    A = np.array([a.v, b.v, c.v])
    Ain = np.linalg.inv(A)
    li = v3(*Ain[0]) * v.v1 + v3(*Ain[1]) * v.v2 + v3(*Ain[2]) * v.v3
    return li


space = []
final = [[], []]
lis = [[], []]
op = [[], []]
for r in range(len(yo[1][0])):
    rad = T(yo[1][0][r])  # in earths frame
    t = (r + 1) * 24 * 60**2
    rad2 = Q(rad, t)  # in my frame
    p = rad2 * (1 / (rad2 @ v3(0, 0, 1))) - v3(0, 0, 1)
    space.append(p)
    op[0].append(rad2.v1)
    op[1].append(rad2.v2)
a = v3.unit(space[0])
b = v3.unit(space[200] - a * (a @ space[200]))
for i in space[188:540]:
    final[0].append(a @ i)
    final[1].append(b @ i)

plt.figure(figsize=(6, 6))
plt.scatter(final[1], final[0], s=0.5)
plt.axis("equal")

plt.show()
