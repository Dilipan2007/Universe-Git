from types import SimpleNamespace as planet_data
from defs_tools.vectors import v3
from core.particle import particle
import numpy as np
from mechanics import classical
import matplotlib.pyplot as plt
import sympy

# at summer solistice
Earth = planet_data(
    mass=5.9722e24,
    tilt=0.4091,
    sid_day=86164.0905,
    avg_day=24 * 60**2,
    pos=v3(1.519 * (10**11), 0, 0),
    vel=v3(181, 29300, 0),
    radius=6371000,
    days=366,
    range=[188, 540],
)

Mars = planet_data(
    mass=6.417e23,
    tilt=np.pi * 25.19 / 180,
    sid_day=88642.7,
    avg_day=88775.2,
    pos=v3(2.492e11, 0, 0),
    vel=v3(0, 2.197e4, 0),
    radius=3.3895e6,
    days=668.6,
    range=[333, 1000],
)


def analemma(planet: planet_data, lat=np.pi / 4):  # lat is latitude
    sun_mass = 1.989e30
    G = 6.67430e-11
    part = particle(
        planet.mass,
        planet.pos,
        planet.vel,
        v3(2 * np.pi / planet.sid_day, planet.tilt, np.pi, "p"),
        planet.radius,
    )
    # creating variables and defining the field
    t = sympy.symbols("time")
    x = [sympy.Function(f"x_{i}")(t) for i in range(1, 4)]
    v = [sympy.diff(x[i]) for i in range(3)]
    V = -(G * sun_mass * part.mass) * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (-1 / 2)

    # collecting data for an year or more
    evo = classical.evolve(
        np.array(
            [
                part,
            ]
        ),
        V,
        0,
        (planet.range[1]) * planet.avg_day,
        np.array(
            [
                x,
            ]
        ),
        np.array(
            [
                v,
            ]
        ),
        t,
    )
    semi_data = evo.run(n=planet.range[1])

    data = [
        semi_data[i] for i in range(len(semi_data))
    ]  # has the particle state at each day

    # defining transformations

    def T(v):  # from orbital to planet's frame w=Tv
        a = v3(1, np.pi / 2 - planet.tilt, 0, "p")
        b = v3(0, 1, 0)
        c = v3(1, planet.tilt, np.pi, "p")
        w = a * v.v1 + b * v.v2 + c * v.v3
        return w

    def Q(v, t, y=0.7106):  # from planet's to my frame w=Fv
        freq = 2 * np.pi / planet.sid_day
        c = v3(1, (np.pi / 2) - y, freq * t + np.pi, "p")
        b = v3(1, np.pi / 2, freq * t + np.pi / 2, "p")
        a = v3(1, y, freq * t, "p")
        A = np.array([a.v, b.v, c.v])
        Ain = np.linalg.inv(A)
        w = v3(*Ain[0]) * v.v1 + v3(*Ain[1]) * v.v2 + v3(*Ain[2]) * v.v3
        return w

    # creating a 2D vector space
    space = []
    c = 0
    for r in data:  # r for radial vector
        time = (c + 1) * planet.avg_day
        q = Q(T(v3.unit(r)), time)  # q = QTr
        space.append(q * (1 / (q @ v3(0, 0, 1))) - v3(0, 0, 1))
        c += 1

    # points to plot
    to_plot = [[], []]
    a = v3.unit(space[0])
    b = v3.unit(space[200] - a * (a @ space[200]))  # spanning vectors
    for p in space:
        to_plot[0].append(p @ a)
        to_plot[1].append(p @ b)

    # plotting
    plt.scatter(
        to_plot[1][planet.range[0] : planet.range[1]],
        to_plot[0][planet.range[0] : planet.range[1]],
        s=0.5,
    )
    plt.axis("equal")
    plt.show()


analemma(Earth)
