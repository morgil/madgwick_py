
import numpy as np
import numbers


class Quaternion:
    """
    A simple class for unit quaternions.
    """
    # TODO initialize class with array
    def __init__(self, w_or_q, x=None, y=None, z=None):
        self._q = None

        if x is not None and y is not None and z is not None:
            w = w_or_q
            q = np.array([w, x, y, z])
        else:
            q = np.array(w_or_q)
            if len(q) != 4:
                raise ValueError("Expecting a 4-element array or w x y z as parameters")

        self._set_q(q)

    def _set_q(self, q):
        self._q = q

    def _get_q(self):
        return self._q

    q = property(_get_q, _set_q)

    def __getitem__(self, item):
        return self._q[item]

    def inv(self):
        return Quaternion(self[0], -self[1], -self[2], -self[3])

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self[3]*other[0] - self[2]*other[1] + self[1]*other[2] + self[0]*other[3]
            x = self[2]*other[0] + self[3]*other[1] - self[0]*other[2] + self[1]*other[3]
            y = -self[1]*other[0] + self[0]*other[1] + self[3]*other[2] + self[2]*other[3]
            z = -self[0]*other[0] - self[1]*other[1] - self[2]*other[2] + self[3]*other[3]

            return Quaternion(w, x, y, z)
        elif isinstance(other, numbers.Number):
            q = self._q * other
            return Quaternion(q)

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            if len(other) != 4:
                raise TypeError("Quaternions must be added to other quaternions or a 4-element array")
            q = self.q + other
        else:
            q = self.q + other.q

        return Quaternion(q)
