from rlutils.rewards.pbrl.interfaces import OracleInterface
from ....features.fastjet.default import *

def oracle(s, a, ns):
    d = dist(s, a, ns, None)
    return - (d + 0.05 * closing_speed(s, a, ns, {"dist": d}) + 10. * up_error(s, a, ns, None))

P = {"pbrl": {"interface": {"class": OracleInterface, "oracle": oracle}}}
