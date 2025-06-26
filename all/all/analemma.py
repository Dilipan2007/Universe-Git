from types import SimpleNamespace as planet_data
from defs_tools.vectors import v3
from core.particle import particle
import numpy as np
from mechanics import classical
import matplotlib.pyplot as plt
import sympy
from skyfield.api import load
from astropy import units as u
import datetime
import pytz

# with skyfield
data_ = load("de421.bsp")
ts = load.timescale()

# at summer solistice
Earth = planet_data(
    name="earth",
    mass=5.9722e24,
    sid_day=86164.09053,
    avg_day=24 * 60**2,
    radius=6371000,
    days=365.256,
    range=[0, 366],
    spin=v3(0, 0, 1),
)

Mars = planet_data(
    name="mars",
    mass=6.417e23,
    sid_day=88642.663,
    avg_day=88775.244,
    radius=3.3895e6,
    days=686.980,
    range=[0, 1000],
    spin=v3(1, np.pi / 2 - np.radians(52.8865), np.radians(317.6814), "p"),
)

Moon = planet_data(
    name="moon",
    mass=7.34767309e22,
    sid_day=2360591.5,
    avg_day=2551443.1,
    radius=1737.4e3,
    days=27.321661,
    range=[0, 30],
    spin=v3(1, (np.pi / 2) - np.radians(66.5392), np.radians(269.9949), "p"),
)

Venus = planet_data(name="venus", mass=4.867e24)

Jupiter = planet_data(
    name="jupiter barycenter",
    mass=1.898e27,
    sid_day=35730.6,
    avg_day=35730.6,
    days=10476,
    range=[0, 11000],
    spin=v3(1, np.pi / 2 - np.radians(67.15105), np.radians(272.73911), "p"),
)

Uranus = planet_data(
    name="uranus",
    mass=8.681e25,
    sid_day=62064,
    avg_day=62064,
    days=30589,
    range=[0, 30600],
    spin=v3(1, np.pi / 2 + np.radians(15.175), np.radians(257.31), "p"),
)


def initialize(obj, time):  # obj is any celestial body
    body = data_[obj.name]
    sun = data_["sun"]
    r = v3(*list((body.at(time) - sun.at(time)).position.km * 1000))
    v = v3(*list((body.at(time) - sun.at(time)).velocity.km_per_s * 1000))
    obj.pos = r
    obj.vel = v
    return obj


def analemma(
    body: planet_data, planet=False, lat=np.pi / 4
):  # lat is latitude , if planet is given then body is moon
    sun_mass = 1.989e30
    G = 6.67430e-11
    part = particle(
        body.mass,
        body.pos,
        body.vel,
        body.spin * (2 * np.pi / body.sid_day),
        1,
    )  # creating particle for the body

    t = sympy.symbols("time")

    # data will have body's position vector over body.range[1] days
    if planet:
        part_planet = particle(
            planet.mass,
            planet.pos,
            planet.vel,
            planet.spin * (2 * np.pi / planet.sid_day),
            planet.radius,
        )  # creating planet particle
        x = np.array(
            [[sympy.Function(f"x__{j}_{i}")(t) for i in range(1, 4)] for j in range(2)]
        )
        v = np.array([[sympy.diff(x[j][i]) for i in range(3)] for j in range(2)])
        V = (-G * sun_mass) * (
            part_planet.mass * (x[1][0] ** 2 + x[1][1] ** 2 + x[1][2] ** 2) ** (-1 / 2)
            + part.mass * (x[0][0] ** 2 + x[0][1] ** 2 + x[0][2] ** 2) ** (-1 / 2)
        ) + (-G * part.mass * part_planet.mass) * (
            (x[1][0] - x[0][0]) ** 2
            + (x[1][1] - x[0][1]) ** 2
            + (x[1][2] - x[0][2]) ** 2
        ) ** (
            -1 / 2
        )

        evo = classical.evolve(
            np.array([part, part_planet]), V, 0, body.range[1] * body.avg_day, x, v, t
        )
        x = 1  # factor for resolution, for better accuracy and to avoid crash
        semi_data = evo.run(n=body.range[1] * x + 1)
        data = [v3(*list(semi_data[0][i * 1][0])) for i in range(semi_data[0])]
    else:
        x = [sympy.Function(f"x_{i}")(t) for i in range(1, 4)]
        v = [sympy.diff(x[i]) for i in range(3)]
        V = -(G * sun_mass * part.mass) * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (
            -1 / 2
        )

        # collecting data for an year or more
        evo = classical.evolve(
            np.array(
                [
                    part,
                ]
            ),
            V,
            0,
            (body.range[1]) * body.avg_day,
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
        semi_data = evo.run(n=body.range[1] + 1)
        data = [v3(*list(semi_data[0][i][0])) for i in range(len(semi_data[0]))]

    # defining transformations

    def P(
        v, align=body.spin, inv=False
    ):  # from celestial J2000 to planet's, z to spin vector with rigid* transformation of basis
        if align.v1 == 0 and align.v2 == 0 and (align.v3 == 1 or align.v3 == -1):
            return v
        a1 = v3(0, 0, 1)
        c1 = v3.unit(v3.cross(a1, align))
        b1 = v3.cross(c1, a1)
        B = np.array(
            [a1.v, b1.v, c1.v]
        )  # transpose is inverse, B is inermediate frame with no significance
        B_inv = B.T
        basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # intermediate vectors
        i_v = [v3(*list(B @ np.array(basis[i]))) for i in range(3)]
        # partially transformed, by rotating by a plane as seen from J2000
        p_t = [v3(1, i_v[i].theta, i_v[i].phi + align.theta, "p") for i in range(3)]
        # fully transformed basis
        a, b, c = (B_inv @ p_t[i].v for i in range(3))
        A = np.array([a, b, c])
        if inv:
            return v3(*list(A.T @ v.v))
        return v3(*list(A @ v.v))

    def Q(v, t, y=lat, inv=False):  # from planet's to my frame w=Fv
        freq = 2 * np.pi / body.sid_day
        c = v3(1, (np.pi / 2) - y, freq * t + np.pi, "p")  #
        b = v3(1, np.pi / 2, freq * t + np.pi / 2, "p")
        a = v3(1, y, freq * t, "p")
        A = np.array([a.v, b.v, c.v])
        if inv:
            return v3(*list(A.T @ v.v))
        return v3(*list(A @ v.v))

    # creating a 3D vector space
    space = []
    c = 0
    avg = v3(0, 0, 0)
    for r in data:  # r for radial vector
        time = (c + 1) * body.avg_day
        q = Q(P(v3.unit(r)), time)  # q = QPr
        space.append(q)
        avg = avg + q
        c += 1
    avg = v3.unit(avg)

    project = []  # is 2D vector space
    for q in space:
        p = P(q, avg)
        project.append((p * (1 / (p @ v3(0, 0, 1))) - v3(0, 0, 1)))

    # points to plot
    to_plot = [[], []]
    for p in project:
        to_plot[0].append(p.v1)
        to_plot[1].append(p.v2)

    # plotting
    plt.scatter(
        to_plot[0][body.range[0] : body.range[1]],
        to_plot[1][body.range[0] : body.range[1]],
        s=0.002,
        c="red",
    )
    plt.axis("equal")


n = 10  # number of observers for better accuarcy
# converting local to Universal Coordinated Time(UTC)
ist = pytz.timezone("Asia/Kolkata")  # indian standard time
for i in range(n):
    min = 24 * 60
    del_min = min / n
    mins = 46 + i * del_min
    my_time = datetime.datetime(
        2025,
        6,
        25 + (0 + int(mins) // 60) // 24,
        (0 + int(mins) // 60) % 24,
        int(mins % 60),
        tzinfo=ist,
    )
    print(my_time)
    utc_time = my_time.astimezone(pytz.utc)
    t = ts.from_datetime(utc_time)
    analemma(initialize(Earth, t))
plt.show()
