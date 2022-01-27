"""
 t   t+1
 0 | 32  | fast_jet.pos.x
 1 | 33  | fast_jet.pos.y
 2 | 34  | fast_jet.pos.z
 3 | 35  | fast_jet.vel.x
 4 | 36  | fast_jet.vel.y
 5 | 37  | fast_jet.vel.z
 6 | 38  | fast_jet.acc.x
 7 | 39  | fast_jet.acc.y
 8 | 40  | fast_jet.acc.z
 9 | 41  | fast_jet.r_vel.x
10 | 42  | fast_jet.r_vel.y
11 | 43  | fast_jet.r_vel.z
12 | 44  | fast_jet.axis.x
13 | 45  | fast_jet.axis.y
14 | 46  | fast_jet.axis.z
15 | 47  | fast_jet.up.x
16 | 48  | fast_jet.up.y
17 | 49  | fast_jet.up.z
18 | 50  | fast_jet.thrust
19 | 51  | target.pos.x
20 | 52  | target.pos.y
21 | 53  | target.pos.z
22 | 54  | target.axis.x
23 | 55  | target.axis.y
24 | 56  | target.axis.z
25 | 57  | target.up.x
26 | 58  | target.up.y
27 | 59  | target.up.z
28 | --  | demanded_pitch
29 | --  | demanded_roll
30 | --  | demanded_yaw
31 | --  | demanded_thrust
"""
import numpy as np

# NOTE: Makes more sense to evaluate performance w.r.t. *previous* target
F = {}
F["dist"]              = lambda t: np.linalg.norm(t[:,32:35] - t[:,19:22], axis=1)
F["delta_dist"]        = lambda t: F["dist"](t) - np.linalg.norm(t[:,0:3] - t[:,19:22], axis=1)
F["alt"]               = lambda t: t[:,33]
F["target_alt"]        = lambda t: t[:,20]
F["alt_error"]         = lambda t: np.abs(F["alt"](t) - F["target_alt"](t))
F["delta_alt_error"]   = lambda t: F["alt_error"](t) - np.abs(t[:,1] - t[:,20])
F["dist_xz"]           = lambda t: (((t[:,32] - t[:,19])**2) + ((t[:,34] - t[:,21])**2))**0.5
F["delta_dist_xz"]     = lambda t: F["dist_xz"](t) - (((t[:,0] - t[:,19])**2) + ((t[:,2] - t[:,21])**2))**0.5
F["pitch"]             = lambda t: np.arcsin(t[:,45])
F["abs_pitch"]         = lambda t: np.abs(F["pitch"](t))
F["target_pitch"]      = lambda t: np.arcsin(t[:,23])
F["pitch_error"]       = lambda t: np.abs(F["pitch"](t) - F["target_pitch"](t))
F["delta_pitch_error"] = lambda t: F["pitch_error"](t) - np.abs(np.arcsin(t[:,13]) - F["target_pitch"](t))
# F["roll"]              = lambda t: np.arctan2(np.sum(___, axis=1), np.sum(___, axis=1)) # NOTE: BIT OF A MESS TO COMPUTE
# F["abs_roll"]          = lambda t: np.abs(F["roll"](t))
# F["target_roll"]       = lambda t: np.arctan2(np.sum(___, axis=1), np.sum(___, axis=1)) 
# F["roll_error"]        = lambda t: np.abs(F["roll"](t) - F["target_roll"](t))
# F["delta_roll_error"]  = lambda t: F["roll_error"](t) - np.abs(np.arctan2(np.sum(___, axis=1), np.sum(___, axis=1)) - F["target_roll"](t))
# Due to symmetry, no reason for absolute heading and target heading to be meaningful
F["hdg_error"]         = lambda t: np.abs(np.arctan2(t[:,46], t[:,44]) - np.arctan2(t[:,24], t[:,22]))
F["delta_hdg_error"]   = lambda t: F["hdg_error"](t) - np.abs(np.arctan2(t[:,14], t[:,12]) - np.arctan2(t[:,24], t[:,22]))
F["fwd_error"]         = lambda t: np.arccos(np.sum(t[:,44:47]*t[:,22:25], axis=1))
F["delta_fwd_error"]   = lambda t: F["fwd_error"](t) - np.arccos(np.sum(t[:,12:15]*t[:,22:25], axis=1))
F["up_error"]          = lambda t: np.arccos(np.sum(t[:,47:50]*t[:,25:28], axis=1))
F["delta_up_error"]    = lambda t: F["up_error"](t) - np.arccos(np.sum(t[:,15:18]*t[:,25:28], axis=1))
F["right_error"]       = lambda t: np.arccos(np.sum(np.cross(t[:,44:47], t[:,47:50])*np.cross(t[:,22:25], t[:,25:28]), axis=1))
F["delta_right_error"] = lambda t: F["right_error"](t) - np.arccos(np.sum(np.cross(t[:,12:15], t[:,15:18])*np.cross(t[:,22:25], t[:,25:28]), axis=1))
F["abs_vel"]           = lambda t: np.linalg.norm(t[:,35:38], axis=1)
F["abs_acc"]           = lambda t: np.linalg.norm(t[:,38:41], axis=1)
F["abs_rot_vel"]       = lambda t: np.linalg.norm(t[:,41:44], axis=1)
F["thrust"]            = lambda t: t[:,50]
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