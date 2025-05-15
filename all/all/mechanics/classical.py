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

    def collide(p1: particle, p2: particle):
        pass


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

    def run(self, n=5):
        # n = int((self.ftime - self.itime) / 0.1)
        intervals = np.linspace(self.itime, self.ftime, n)
        rk_systems = np.array(
            [
                rk45_toolkit.make_rk_sys(self.eqs[partic], self.p_v[partic], self.t)
                for partic in range(len(self.particles))
            ]
        )

        updated_pos = np.random.rand(len(self.particles), 3)
        updated_vel = np.random.rand(len(self.particles), 3)
        plot = [[], []]
        for i in range(len(intervals) - 1):
            if i == 0:
                ini_pos = np.array(
                    [
                        np.array(self.particles[partic].pos)
                        for partic in range(len(self.particles))
                    ]
                )
                ini_vel = np.array(
                    [
                        np.array(self.particles[partic].vel)
                        for partic in range(len(self.particles))
                    ]
                )
            else:
                ini_pos = updated_pos
                ini_vel = updated_vel

            sol = np.array(
                [
                    solve_ivp(
                        rk_systems[partic],
                        [intervals[i], intervals[i + 1]],
                        list(ini_pos[partic]) + list(ini_vel[partic]),
                        t_eval=[intervals[i + 1]],
                    )
                    for partic in range(len(self.particles))
                ]
            )
            # sol contains: .y[0:3] position, .y[3:6] velocity and higher derivatives
            updated_pos = np.array(
                [
                    np.array([sol[partic].y[axis][0] for axis in range(3)])
                    for partic in range(len(self.particles))
                ]
            )
            updated_vel = np.array(
                [
                    np.array([sol[partic].y[axis][0] for axis in range(3)])
                    for partic in range(len(self.particles))
                ]
            )
            print(updated_pos, intervals[i + 1])
            plot[0].append(updated_pos[0, 2])
            plot[1].append(updated_pos[0, 0])
        return plot
