from defs.vectors import v3
from mechanics.classical import eq_of_motion
import sympy


class particle:
    def __init__(
        self, mass, pos: v3, vel: v3, avel: v3, r, shape="sphere"
    ):  # pos, vel and angular vel(avel) are of v3; r is radius for now
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.avel = avel
        self.r = r
        self.s = shape
        self.shape(self.s)

    def shape(self, s):  # shall be updated for different shapes
        if s == "sphere":
            self.inertia = (2 * self.mass * (self.r) ** 2) / 5


class evolve:
    def __init__(self, particles, V: sympy.Function, itime, ftime):
        self.particles = particles
        self.V = V
        self.itime = itime
        self.ftime = ftime
        self.acc = eq_of_motion(self.particles[0], self.V)
