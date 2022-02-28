import numpy as np
from .features import F

def target_pose_linear(tr): 
    """
    Linear function of angular and positional deviation from target pose.
    """
    return - (F["fwd_error"](tr) + F["up_error"](tr) + 0.01*F["dist"](tr)).sum()

def target_pose_tree(tr):
    def reward(d, d_d, f, u):
        constant = -1.
        reward_d   = 0. if d < 20      else (-1. if d < 50      else -2.) 
        reward_d_d = 0. if d_d < 0     else -1.
        reward_f   = 0. if f < np.pi/4 else (-1. if f < np.pi/2 else -2.)
        reward_u   = 0. if u < np.pi/4 else (-1. if u < np.pi/2 else -2.)
        return constant + reward_d + reward_d_d + reward_f + reward_u
    return [reward(d, d_d, f, u) for d, d_d, f, u in zip(
            F["dist"](tr), F["delta_dist"](tr), F["fwd_error"](tr), F["up_error"](tr))]
    
if False: # TODO: generic function to visualise 2D reward
    import numpy as np
    import matplotlib.pyplot as plt
    lims = (425, np.pi)
    RES = 100
    dx, dy = lims[0]/RES, lims[1]/RES
    r = np.zeros((RES, RES))
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