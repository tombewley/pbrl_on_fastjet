import torch
from torch.nn.functional import cosine_similarity as cosim

# NOTE: Makes more sense to evaluate performance w.r.t. *previous* target
# TODO: Action rewards?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_VEC = torch.tensor([0,9.81,0], device=device)

def preprocessor(t):
    """Normalise fwd/up components for both ego and target to make them unit vectors."""
    for low, high in ((12,15), (15,18), (31,34), (34,37), (53,56), (56,59), (72,75), (75,78)):
        t[...,low:high] /= torch.linalg.norm(t[...,low:high], axis=-1).unsqueeze(-1)
    return t

def dist(t, f):              return torch.linalg.norm(t[...,41:44] - t[...,19:22], axis=-1)
def closing_speed(t, f):     return f["dist"] - torch.linalg.norm(t[...,0:3] - t[...,19:22], axis=-1)

def alt(t, f):               return t[...,42]
def target_alt(t, f):        return t[...,20]
def alt_error(t, f):         return torch.abs(f["alt"] - f["target_alt"])
def delta_alt_error(t, f):   return f["alt_error"] - torch.abs(t[...,1] - f["target_alt"])

def dist_xz(t, f):           return (((t[...,41] - t[...,19])**2) + ((t[...,43] - t[...,21])**2))**0.5
def delta_dist_xz(t, f):     return f["dist_xz"] - (((t[...,0] - t[...,19])**2) + ((t[...,2] - t[...,21])**2))**0.5

def pitch(t, f):             return torch.asin(t[...,54])
def abs_pitch(t, f):         return torch.abs(f["pitch"])
def target_pitch(t, f):      return torch.asin(t[...,32])
def pitch_error(t, f):       return torch.abs(f["pitch"] - f["target_pitch"])
def delta_pitch_error(t, f): return f["pitch_error"] - torch.abs(torch.asin(t[...,13]) - f["target_pitch"])

# TODO: Compute these properly
# f["roll"]              = lambda t: torch.atan2(torch.sum(___, axis=-1), torch.sum(___, axis=-1))
# f["abs_roll"]          = lambda t: torch.abs(f["roll"](t, f))
# f["target_roll"]       = lambda t: torch.atan2(torch.sum(___, axis=-1), torch.sum(___, axis=-1)) 
# f["roll_error"]        = lambda t: torch.abs(f["roll"](t, f) - f["target_roll"](t, f))
# f["delta_roll_error"]  = lambda t: f["roll_error"](t, f) - torch.abs(torch.atan2(torch.sum(___, axis=-1), torch.sum(___, axis=-1)) - f["target_roll"](t, f))

def _target_hdg(t, f):       return torch.atan2(t[...,33], t[...,31]) # NOTE: Due to symmetry, no reason for absolute heading and target heading to be meaningful
def hdg_error(t, f):         return torch.abs(torch.atan2(t[...,55], t[...,53]) - _target_hdg(t, f))
def delta_hdg_error(t, f):   return f["hdg_error"] - torch.abs(torch.atan2(t[...,14], t[...,12]) - _target_hdg(t, f))

def fwd_error(t, f):         return  torch.acos(cosim(t[...,53:56], t[...,31:34], dim=-1))
def delta_fwd_error(t, f):   return f["fwd_error"] - torch.acos(cosim(t[...,12:15], t[...,31:34], dim=-1))

def up_error(t, f):          return torch.acos(cosim(t[...,56:59], t[...,34:37], dim=-1))
def delta_up_error(t, f):    return f["up_error"] - torch.acos(cosim(t[...,15:18], t[...,34:37], dim=-1))

def _target_right(t, f):     return torch.cross(t[...,31:34], t[...,34:37])
def right_error(t, f):       return torch.acos(cosim(torch.cross(t[...,53:56], t[...,56:59]), _target_right(t, f), dim=-1))
def delta_right_error(t, f): return f["right_error"] - torch.acos(cosim(torch.cross(t[...,12:15], t[...,15:18]), _target_right(t, f), dim=-1))

def abs_vel(t, f):           return torch.linalg.norm(t[...,44:47], axis=-1)
def g_force(t, f):           return torch.linalg.norm(t[...,47:50]+G_VEC, axis=-1) / 9.81 # NOTE: Includes gravity
def pitch_rate(t, f):        return torch.abs(t[...,50])
def roll_rate(t, f):         return torch.abs(t[...,51]) 
def yaw_rate(t, f):          return torch.abs(t[...,52])
def thrust(t, f):            return t[...,59]
def delta_thrust(t, f):      return torch.abs(f["thrust"] - t[...,18])

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

"""
 t   t+1
 0 | 41  | ego.pos.x
 1 | 42  | ego.pos.y
 2 | 43  | ego.pos.z
 3 | 44  | ego.vel.x
 4 | 45  | ego.vel.y
 5 | 46  | ego.vel.z
 6 | 47  | ego.acc.x
 7 | 48  | ego.acc.y
 8 | 49  | ego.acc.z
 9 | 50  | ego.r_vel.x
10 | 51  | ego.r_vel.y
11 | 52  | ego.r_vel.z
12 | 53  | ego.axis.x
13 | 54  | ego.axis.y
14 | 55  | ego.axis.z
15 | 56  | ego.up.x
16 | 57  | ego.up.y
17 | 58  | ego.up.z
18 | 59  | ego.thrust
19 | 60  | target.pos.x
20 | 61  | target.pos.y
21 | 62  | target.pos.z
22 | 63  | target.vel.x
23 | 64  | target.vel.y
24 | 65  | target.vel.z
25 | 66  | target.acc.x
26 | 67  | target.acc.y
27 | 68  | target.acc.z
28 | 69  | target.r_vel.x,
29 | 70  | target.r_vel.y,
30 | 71  | target.r_vel.z,
31 | 72  | target.axis.x, 
32 | 73  | target.axis.y,
33 | 74  | target.axis.z,
34 | 75  | target.up.x,
35 | 76  | target.up.y,
36 | 77  | target.up.z
37 | --  | demanded_pitch
38 | --  | demanded_roll
39 | --  | demanded_yaw
40 | --  | demanded_thrust
"""

"""
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
