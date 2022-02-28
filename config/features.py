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
import numpy as np
G_VEC = np.array([0,9.81,0])

# NOTE: Makes more sense to evaluate performance w.r.t. *previous* target
F = {}
F["dist"]              = lambda t: np.linalg.norm(t[:,41:44] - t[:,19:22], axis=1)
F["delta_dist"]        = lambda t: F["dist"](t) - np.linalg.norm(t[:,0:3] - t[:,19:22], axis=1)
F["alt"]               = lambda t: t[:,42]
F["target_alt"]        = lambda t: t[:,20]
F["alt_error"]         = lambda t: np.abs(F["alt"](t) - F["target_alt"](t))
F["delta_alt_error"]   = lambda t: F["alt_error"](t) - np.abs(t[:,1] - F["target_alt"](t))
F["dist_xz"]           = lambda t: (((t[:,41] - t[:,19])**2) + ((t[:,43] - t[:,21])**2))**0.5
F["delta_dist_xz"]     = lambda t: F["dist_xz"](t) - (((t[:,0] - t[:,19])**2) + ((t[:,2] - t[:,21])**2))**0.5
F["pitch"]             = lambda t: np.arcsin(t[:,54])
F["abs_pitch"]         = lambda t: np.abs(F["pitch"](t))
F["target_pitch"]      = lambda t: np.arcsin(t[:,32])
F["pitch_error"]       = lambda t: np.abs(F["pitch"](t) - F["target_pitch"](t))
F["delta_pitch_error"] = lambda t: F["pitch_error"](t) - np.abs(np.arcsin(t[:,13]) - F["target_pitch"](t))
# F["roll"]              = lambda t: np.arctan2(np.sum(___, axis=1), np.sum(___, axis=1)) # NOTE: BIT OF A MESS TO COMPUTE
# F["abs_roll"]          = lambda t: np.abs(F["roll"](t))
# F["target_roll"]       = lambda t: np.arctan2(np.sum(___, axis=1), np.sum(___, axis=1)) 
# F["roll_error"]        = lambda t: np.abs(F["roll"](t) - F["target_roll"](t))
# F["delta_roll_error"]  = lambda t: F["roll_error"](t) - np.abs(np.arctan2(np.sum(___, axis=1), np.sum(___, axis=1)) - F["target_roll"](t))
# NOTE: Due to symmetry, no reason for absolute heading and target heading to be meaningful
F["hdg_error"]         = lambda t: np.abs(np.arctan2(t[:,55], t[:,53]) - np.arctan2(t[:,33], t[:,31]))
F["delta_hdg_error"]   = lambda t: F["hdg_error"](t) - np.abs(np.arctan2(t[:,14], t[:,12]) - np.arctan2(t[:,33], t[:,31]))
F["fwd_error"]         = lambda t: np.arccos(np.sum(t[:,53:56]*t[:,31:34], axis=1))
F["delta_fwd_error"]   = lambda t: F["fwd_error"](t) - np.arccos(np.sum(t[:,12:15]*t[:,31:34], axis=1))
F["up_error"]          = lambda t: np.arccos(np.sum(t[:,56:59]*t[:,34:37], axis=1))
F["delta_up_error"]    = lambda t: F["up_error"](t) - np.arccos(np.sum(t[:,15:18]*t[:,34:37], axis=1))
F["right_error"]       = lambda t: np.arccos(np.sum(np.cross(t[:,53:56], t[:,56:59])*np.cross(t[:,31:34], t[:,34:37]), axis=1))
F["delta_right_error"] = lambda t: F["right_error"](t) - np.arccos(np.sum(np.cross(t[:,12:15], t[:,15:18])*np.cross(t[:,31:34], t[:,34:37]), axis=1))
F["abs_vel"]           = lambda t: np.linalg.norm(t[:,44:47], axis=1)
F["g_force"]           = lambda t: np.linalg.norm(t[:,47:50]+G_VEC, axis=1) / 9.81 # NOTE: Includes gravity
F["pitch_rate"]        = lambda t: np.abs(t[:,50])
F["roll_rate"]         = lambda t: np.abs(t[:,51]) 
F["yaw_rate"]          = lambda t: np.abs(t[:,52])
F["thrust"]            = lambda t: t[:,59]
F["delta_thrust"]      = lambda t: np.abs(F["thrust"](t) - t[:,18])

# TODO: Action rewards?

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