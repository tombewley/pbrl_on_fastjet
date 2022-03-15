from numpy import pi
from torch import tensor
from .features import F

def target_pose_tree(tr):
    def reward(d, v, f, u):
        constant = -1.
        reward_d   = 0. if d < 20      else (-1. if d < 50      else -2.) 
        reward_v   = 0. if v < 0       else -1.
        reward_f   = 0. if f < pi/4 else (-1. if f < pi/2 else -2.)
        reward_u   = 0. if u < pi/4 else (-1. if u < pi/2 else -2.)
        return constant + reward_d + reward_v + reward_f + reward_u
    return tensor([reward(d, v, f, u) for d, v, f, u in zip(
           F["dist"](tr), F["closing_speed"](tr), F["fwd_error"](tr), F["up_error"](tr))],
           device=tr.device)

def target_pose_linear(tr): 
    """
    Linear function of angular and positional deviation from target pose.
    """
    return - (F["fwd_error"](tr) + F["up_error"](tr) + 0.01*F["dist"](tr))
    
def negative_dist_to_target(tr):
    return -F["dist"](tr)

def dist_closing_uperr_when_close(tr):
    d = F["dist"](tr)
    return - (d + F["closing_speed"](tr) + 30. * F["up_error"](tr) * (d < 30.))

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