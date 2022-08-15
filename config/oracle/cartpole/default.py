from torch import pi, logical_and
from rlutils.rewards.interfaces import OracleInterface

x_threshold = 2.4
theta_threshold_radians = 12 * 2 * pi / 360

def oracle(tr):
    _, _, _, _, _, x, x_dot, theta, theta_dot = tr.T # NOTE: Uses next state
    
    return logical_and(
           logical_and(-x_threshold <= x, x <= x_threshold),
           logical_and(-theta_threshold_radians <= theta, theta <= theta_threshold_radians)
           ).float()

#     return 1 if not done else 0

P = {"pbrl": {"interface": {"class": OracleInterface, "oracle": oracle}}}
