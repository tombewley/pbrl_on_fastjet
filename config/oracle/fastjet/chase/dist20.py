from torch import abs
from rlutils.rewards.pbrl.interfaces import OracleInterface
from ....features.fastjet.default import *

def oracle(s, a, ns):
    return - (abs(dist(s, a, ns, None) -20.) + 5. * roll_rate(s, a, ns, None) + 100. * (alt(s, a, ns, None) < 50))

P = {"pbrl": {"interface": {"class": OracleInterface, "oracle": oracle}}}
