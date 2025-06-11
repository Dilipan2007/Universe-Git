import numpy as np


class v3:
    def __init__(self, v1, v2, v3, polor="np"):  # np- not polor
        if polor == "p":
            self.v1 = v1 * np.sin(v2) * np.cos(v3)
            self.v2 = v1 * np.sin(v2) * np.sin(v3)
            self.v3 = v1 * np.cos(v2)
        elif polor == "np":
            self.v1 = v1
            self.v2 = v2
            self.v3 = v3
        self.v = np.array([self.v1, self.v2, self.v3])

    def norm(self):
        return np.linalg.norm(self.v)

    def __add__(self, o):
        return v3(self.v1 + o.v1, self.v2 + o.v2, self.v3 + o.v3)

    def __sub__(self, o):
        return self + -o

    def __mul__(self, scalar):
        return v3(self.v1 * scalar, self.v2 * scalar, self.v3 * scalar)

    def __matmul__(self, other):
        # Dot product  @
        return self.v1 * other.v1 + self.v2 * other.v2 + self.v3 * other.v3

    def cross(self, other):
        a = np.cross(self.v, other.v)
        return v3(a[0], a[1], a[2])

    def __neg__(self):
        return v3(-self.v1, -self.v2, -self.v3)

    def unit(self):
        return v3(self.v1 / self.norm(), self.v2 / self.norm(), self.v3 / self.norm())
