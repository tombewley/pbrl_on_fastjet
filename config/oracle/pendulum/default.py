from torch import pi, acos, clamp
from rlutils.observers.pbrl.interfaces import OracleInterface

MAX_TORQUE = 2.

def oracle(tr):
    def angle_normalize(a): return ((a + pi) % (2 * pi)) - pi
    return - (angle_normalize(acos(clamp(tr[:,0], -1, 1))) ** 2 + \
            .1 * tr[:,2] ** 2 + \
            .001 * (clamp(tr[:,3], -MAX_TORQUE, MAX_TORQUE) ** 2))

P = {"pbrl": {"interface": {"class": OracleInterface, "oracle": oracle}}}
