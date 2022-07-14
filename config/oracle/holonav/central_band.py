from torch import logical_and
from rlutils.rewards.pbrl.interfaces import OracleInterface

def oracle(tr):
    _, _, _, _, x, y = tr.T
    return logical_and(x >= .25, x <= .75).float()

P = {"pbrl": {"interface": {"class": OracleInterface, "oracle": oracle}}}
