from rlutils.observers.pbrl.interfaces import OracleInterface
from ...features.fastjet_default import *

def oracle(tr):
    d = dist(tr, None)
    return - (d + 0.05 * closing_speed(tr, {"dist": d}) + 10. * up_error(t, None))

P = {"pbrl": {"interface": {"class": OracleInterface, "oracle": oracle}}}
