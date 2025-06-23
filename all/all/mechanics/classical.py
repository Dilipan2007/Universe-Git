from defs_tools.vectors import v3
from defs_tools import rk45_toolkit
from core.particle import particle
import sympy
import numpy as np
from typing import Sequence
from scipy.integrate import solve_ivp


class system:  # to return symbolic lagrangian equations for rk45 solve only in terms of coordinate axes and its derivatives
    def __init__(self, particles: Sequence[particle], V: sympy.Function, p_v, v_v, t):
        self.particles = particles
        self.V = V
        self.p_v = p_v
        self.v_v = v_v
        self.t = t

    def lagrangian(self):
        lag = np.array(
            [
                (
                    np.array(
                        [
                            (
                                (
                                    self.particles[partic].mass
                                    * (self.v_v[partic, axis] ** 2)
                                )
                                / 2
                            )
                            - self.V
                            for axis in range(3)
                        ]
                    )
                )
                for partic in range(len(self.particles))
            ]
        )
        return lag

    def eqs_of_motion(self):
        lag = self.lagrangian()

        expr = np.array(
            [
                (
                    np.array(
                        [
                            sympy.diff(
                                sympy.diff(
                                    lag[partic, axis],
                                    self.v_v[partic, axis],
                                ),
                                self.t,
                            )
                            - sympy.diff(lag[partic, axis], self.p_v[partic, axis])
                            for axis in range(3)
                        ]
                    )
                )
                for partic in range(len(self.particles))
            ]
        )
        acceleration = np.array(
            [
                (
                    np.array(
                        [
                            sympy.solve(
                                expr[partic, axis],
                                sympy.diff(self.p_v[partic, axis], self.t, 2),
                            )[0]
                            for axis in range(3)
                        ]
                    )
                )
                for partic in range(len(self.particles))
            ]
        )
        lambdified_acc = np.array(
            [
                (
                    np.array(
                        [
                            sympy.lambdify(
                                (
                                    self.p_v[partic, axis],
                                    self.v_v[partic, axis],
                                    self.t,
                                ),
                                acceleration[partic, axis],
                                "numpy",
                            )
                            for axis in range(3)
                        ]
                    )
                )
                for partic in range(len(self.particles))
            ]
        )
        return acceleration
        # returning d2x/dt2(x,dx/dt,t) of all coordinates


class evolve:
    def __init__(
        self,
        particles: Sequence[particle],
        V: sympy.Function,
        itime,
        ftime,
        p_v,
        v_v,
        t: sympy.Symbol,
    ):
        self.particles = particles
        self.V = V
        self.itime = itime
        self.ftime = ftime
        self.t = t
        self.p_v = p_v
        sys = system(particles, V, p_v, v_v, self.t)
        self.eqs = sys.eqs_of_motion()

    def collide(self, p1: particle, p2: particle):
        # refer all\all\core\refs\Colliding balls
        rhat = v3.unit(p2.pos - p1.pos)
        normal_v1 = rhat * (
            ((p1.mass - p2.mass) * (p1.vel @ rhat) + 2 * p2.mass * (p2.vel @ rhat))
            / (p1.mass + p2.mass)
        )
        normal_v2 = rhat * (
            ((p2.mass - p2.mass) * (p2.vel @ rhat) + 2 * p1.mass * (p1.vel @ rhat))
            / (p1.mass + p2.mass)
        )
        q1 = p1.vel - rhat * (p1.vel @ rhat)
        q2 = p2.vel - rhat * (p2.vel @ rhat)
        J1 = (
            q2
            - q1
            - ((v3.cross(p1.avel, rhat)) * p1.r + (v3.cross(p2.avel, rhat)) * p2.r)
        ) * (2 * p1.mass / 7)
        J2 = -(
            q2
            - q1
            - ((v3.cross(p1.avel, rhat)) * p1.r + (v3.cross(p2.avel, rhat)) * p2.r)
        ) * (2 * p2.mass / 7)
        tang_v1 = q1 + (
            q2
            - q1
            - ((v3.cross(p1.avel, rhat)) * p1.r + (v3.cross(p2.avel, rhat)) * p2.r)
        ) * (2 / 7)
        tang_v2 = q2 + tang_v1 - q1
        # updating
        p1.vel, p2.vel = normal_v1 + tang_v1, normal_v2 + tang_v2
        p1.avel += (v3.cross(J1, rhat)) * (5 / (2 * p1.r))
        p2.avel += (v3.cross(J2, rhat)) * (5 / (2 * p2.r))
        return (p1, p2)

    def run(self, n=5):
        # n = int((self.ftime - self.itime) / 0.1)
        intervals = np.linspace(self.itime, self.ftime, n)
        rk_systems = np.array(
            [
                rk45_toolkit.make_rk_sys(self.eqs[partic], self.p_v[partic], self.t)
                for partic in range(len(self.particles))
            ]
        )

        plot1 = [[], []]
        plot2 = [[], []]
        ret = {}
        for i in range(len(intervals) - 1):
            sol = np.array(
                [
                    solve_ivp(
                        rk_systems[partic],
                        [intervals[i], intervals[i + 1]],
                        [
                            *list(self.particles[partic].pos.v),
                            *list(self.particles[partic].vel.v),
                        ],
                        t_eval=[intervals[i + 1]],
                    )
                    for partic in range(len(self.particles))
                ]
            )

            # sol contains: .y[0:3] position, .y[3:6] velocity and higher derivatives
            # updating
            for partic in range(len(self.particles)):
                self.particles[partic].pos = v3(
                    *[sol[partic].y[axis][0] for axis in range(3)]
                )
                self.particles[partic].vel = v3(
                    *[sol[partic].y[axis][0] for axis in range(3, 6)]
                )

            # Colliding balls
            for k in range(len(self.particles)):
                for j in range(k + 1, len(self.particles)):
                    if (
                        v3.norm(self.particles[k].pos - self.particles[j].pos)
                        <= self.particles[k].r + self.particles[j].r
                    ):

                        self.particles[k], self.particles[j] = self.collide(
                            self.particles[k], self.particles[j]
                        )
            plot1[0].append(self.particles[0].pos.v[0])
            plot1[1].append(self.particles[0].pos.v[1])
            plot2[0].append(v3.unit(self.particles[0].pos) * -1)
            plot2[1].append(
                v3(
                    1,
                    (np.pi / 4),
                    np.pi + v3.norm(self.particles[0].avel) * intervals[i + 1],
                    "p",
                )
            )
            ret[i] = self.particles[0].pos
        # should return list of particles at each time for analemma
        return ret
