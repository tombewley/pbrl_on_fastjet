import torch
from torch.nn.functional import cosine_similarity as cosim

# TODO: Action rewards?
print("TODO: New feature: line of sight error between fwd and vector to target")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_VEC = torch.tensor([0,9.81,0], device=device)

def preprocessor(s, a, ns):
    """Normalise fwd/up components for both ego and target to make them unit vectors."""
    for low, high in ((12,15), (15,18), (31,34), (34,37)):
        s [...,low:high] /= torch.linalg.norm(s [...,low:high], axis=-1).unsqueeze(-1)
        ns[...,low:high] /= torch.linalg.norm(ns[...,low:high], axis=-1).unsqueeze(-1)
    return s, a, ns

def dist(s, a, ns, f):              return torch.linalg.norm(ns[...,0:3] - ns[...,19:22], axis=-1)
def closing_speed(s, a, ns, f):     return f["dist"] - dist(None, None, s, None)

def alt(s, a, ns, f):               return ns[...,1]
def target_alt(s, a, ns, f):        return ns[...,20]
def alt_error(s, a, ns, f):         return torch.abs(f["alt"] - f["target_alt"])
def delta_alt_error(s, a, ns, f):   return f["alt_error"] - torch.abs(s[...,1] - s[...,20])

def dist_xz(s, a, ns, f):           return (((ns[...,0] - ns[...,19])**2) + ((ns[...,2] - ns[...,21])**2))**0.5
def delta_dist_xz(s, a, ns, f):     return f["dist_xz"] - dist_xz(None, None, s, None)

def pitch(s, a, ns, f):             return torch.asin(ns[...,13])
def abs_pitch(s, a, ns, f):         return torch.abs(f["pitch"])
def target_pitch(s, a, ns, f):      return torch.asin(ns[...,32])
def pitch_error(s, a, ns, f):       return torch.abs(f["pitch"] - f["target_pitch"])
def delta_pitch_error(s, a, ns, f): return f["pitch_error"] - torch.abs(pitch(None, None, s, None) - target_pitch(None, None, s, None))

# TODO: Compute these properly
# f["roll"]              = lambda t: torch.atan2(torch.sum(___, axis=-1), torch.sum(___, axis=-1))
# f["abs_roll"]          = lambda t: torch.abs(f["roll"](s, a, ns, f))
# f["target_roll"]       = lambda t: torch.atan2(torch.sum(___, axis=-1), torch.sum(___, axis=-1)) 
# f["roll_error"]        = lambda t: torch.abs(f["roll"](s, a, ns, f) - f["target_roll"](s, a, ns, f))
# f["delta_roll_error"]  = lambda t: f["roll_error"](s, a, ns, f) - torch.abs(torch.atan2(torch.sum(___, axis=-1), torch.sum(___, axis=-1)) - f["target_roll"](s, a, ns, f))

# NOTE: Due to symmetry, no reason for absolute heading and target heading to be meaningful
def hdg_error(s, a, ns, f):         return torch.abs(torch.atan2(ns[...,14], ns[...,12]) - torch.atan2(ns[...,33], ns[...,31]))
def delta_hdg_error(s, a, ns, f):   return f["hdg_error"] - hdg_error(None, None, s, None)

def _target_fwd(s):                 return s[...,31:34]
def fwd_error(s, a, ns, f):         return torch.acos(cosim(ns[...,12:15], _target_fwd(ns), dim=-1))
def delta_fwd_error(s, a, ns, f):   return f["fwd_error"] - fwd_error(None, None, s, None)

def _target_up(s):                  return s[...,34:37]
def up_error(s, a, ns, f):          return torch.acos(cosim(ns[...,15:18], _target_up(ns), dim=-1))
def delta_up_error(s, a, ns, f):    return f["up_error"] - up_error(None, None, s, None)

def right_error(s, a, ns, f):       return torch.acos(cosim(torch.cross(ns[...,12:15], ns[...,15:18]), torch.cross(_target_fwd(ns), _target_up(ns)), dim=-1))
def delta_right_error(s, a, ns, f): return f["right_error"] - right_error(None, None, s, None)

def abs_vel(s, a, ns, f):           return torch.linalg.norm(ns[...,3:6], axis=-1)
def g_force(s, a, ns, f):           return torch.linalg.norm(ns[...,6:9]+G_VEC, axis=-1) / 9.81 # NOTE: Includes gravity
def pitch_rate(s, a, ns, f):        return torch.abs(ns[...,9])
def roll_rate(s, a, ns, f):         return torch.abs(ns[...,10]) 
def yaw_rate(s, a, ns, f):          return torch.abs(ns[...,11])
def thrust(s, a, ns, f):            return ns[...,18]
def delta_thrust(s, a, ns, f):      return torch.abs(f["thrust"] - thrust(None, None, s, None))

features = [
    dist,
    closing_speed,
    alt,
    target_alt,
    alt_error,
    delta_alt_error,
    dist_xz,
    delta_dist_xz,
    pitch,
    abs_pitch,
    target_pitch,
    pitch_error,
    delta_pitch_error,
    hdg_error,
    delta_hdg_error,
    fwd_error,
    delta_fwd_error,
    up_error,
    delta_up_error,
    right_error,
    delta_right_error,
    abs_vel,
    g_force,
    pitch_rate,
    roll_rate,
    yaw_rate,
    thrust,
    delta_thrust
]

P = {
    "pbrl": {
        "model": {
            "featuriser": {
                "preprocessor": preprocessor,
                "features": features
            }
        }
    }
}

"""
STATE
 0 | ego.pos.x
 1 | ego.pos.y
 2 | ego.pos.z
 3 | ego.vel.x
 4 | ego.vel.y
 5 | ego.vel.z
 6 | ego.acc.x
 7 | ego.acc.y
 8 | ego.acc.z
 9 | ego.r_vel.x
10 | ego.r_vel.y
11 | ego.r_vel.z
12 | ego.axis.x
13 | ego.axis.y
14 | ego.axis.z
15 | ego.up.x
16 | ego.up.y
17 | ego.up.z
18 | ego.thrust
19 | target.pos.x
20 | target.pos.y
21 | target.pos.z
22 | target.vel.x
23 | target.vel.y
24 | target.vel.z
25 | target.acc.x
26 | target.acc.y
27 | target.acc.z
28 | target.r_vel.x,
29 | target.r_vel.y,
30 | target.r_vel.z,
31 | target.axis.x, 
32 | target.axis.y,
33 | target.axis.z,
34 | target.up.x,
35 | target.up.y,
36 | target.up.z
ACTION
 0 | demanded_pitch
 1 | demanded_roll
 2 | demanded_yaw
 3 | demanded_thrust

In original reward machines:

Features used in rules:
- pitch (and abs(pitch) but equivalent to two symmetric rules)
- y - path_alt (and abs)
- abs(vel_y)
- mag(r_vel)
- abs(roll)
- thrust
- up_y
- G force NOTE: Includes gravity vector
- abs(r_vel.roll)
- diff_angle(axis, path_axis_fw)

Features with linear coefficients:
- diff_angle(axis_left, path_axis_left) (and abs(pi/2 - this angle))
- diff_angle(axis_left, path_axis_right)
- abs(pi/2 - diff_angle(axis, path_axis_left))
- diff_angle(axis, path_axis_back)
- pitch (and abs)
- abs(y - path_alt)
- diff_angle(axis, prev_axis_offset): 20 deg offset upwards
- abs(vel_y)
- abs(roll) (and pi - abs(roll))
- diff_angle(axis_xz, path_axis_fw): Heading only
- diff_angle(axis, path_axis_fw)
- mag(r_vel)
- diff_angle(axis_roll, path_axis_roll): 45 deg offset left
- abs(0.5 - r_vel.roll): Encourage smooth roll
"""
