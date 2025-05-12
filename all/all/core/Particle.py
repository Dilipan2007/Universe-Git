from defs import vectors as vec


class particle:
    def __init__(self, mass, pos, vel, shape):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.s = shape

    def shape(self, s):
        pass
