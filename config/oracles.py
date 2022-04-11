from math import pi
from .features import *

def target_pose_tree(t):
    d, f, u = dist(t, None), fwd_error(t, None), up_error(t, None)
    c = closing_speed(t, {"dist": d})
    reward_d = (-1. * (d > 20).float()) + (-1. * (d > 50).float())
    reward_c = (-1 * (c > 0).float())
    reward_f = (-1. * (f > pi/4).float()) + (-1. * (f > pi/2).float())
    reward_u = (-1. * (u > pi/4).float()) + (-1. * (u > pi/2).float())
    constant = -1.
    return reward_d + reward_c + reward_f + reward_u + constant

def target_pose_linear(t): 
    """
    Linear function of angular and positional deviation from target pose.
    """
    return - (fwd_error(t, None) + up_error(t, None) + 0.075*dist(t, None))
    
def dist_only(t):
    return -dist(t, None)

def dist_pose_when_close(t):
    d = dist(t, None)
    return - (d + (d < 30).float() * (fwd_error(t, None) + up_error(t, None)))

def dist_closing_uperr(t):
    d = dist(t, None)
    return - (d + 0.1 * closing_speed(t, {"dist": d}) + 30. * up_error(t, None))

def dist_closing_uperr_v2(t):
    d = dist(t, None)
    return - (d + 0.05 * closing_speed(t, {"dist": d}) + 10. * up_error(t, None))

if False: # TODO: generic function to visualise 2D reward
    from numpy import zeros
    import matplotlib.pyplot as plt
    lims = (425, pi)
    RES = 100
    dx, dy = lims[0]/RES, lims[1]/RES
    r = zeros((RES, RES))
    for x in range(RES):
        for y in range(RES):
            r[x, y] = - (0.01*x*dx + y*dy)
    fig, ax = plt.subplots(figsize=(12,8))
    im = ax.imshow(r, cmap="Reds_r", origin="lower", 
                   extent=(-dx, lims[0]+dx, -dy, lims[1]+dy), aspect="auto")
    ax.set_title("True reward")
    ax.set_xlabel("dist")
    ax.set_ylabel("fwd_error")
    plt.colorbar(mappable=im)
    ax.autoscale_view()
    plt.savefig("True_reward.png")