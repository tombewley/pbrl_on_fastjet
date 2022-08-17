from vpython import canvas, label, slider, vector, dot
import numpy as np
from scipy.interpolate import interp1d

from rlutils.rewards.interfaces import Interface
from fastjet.env import FastJetEnv
from fastjet.fast_jet import dir_to_vec, vec_to_dir


class FastJetInterface(Interface):
    def __init__(self, graph, P):
        Interface.__init__(self, graph)

    def __enter__(self):
        self.envs = []
        for e in range(2):
            # TODO: Need to use correct task?
            self.envs.append(FastJetEnv(task="target_easy", camera_angle="outside_target_bg_skew", show_bbox=False))
            # self.envs[e].jets[0].arrow_length = 20
            self.envs[e].render_mode = "human"
            self.envs[e].reset()
            self.envs[e].render(scene=canvas(width=900, height=600, align="right" if e == 0 else None))
            self.envs[e].ep_num_label = label(canvas=self.envs[e].scene, text="", pixel_pos=True,
                pos=vector(50, self.envs[e].scene.height-50, 0), align="left", height=30)
        self.envs.reverse() # Creation order puts self.envs[0] on the right by default
        self.envs[0].scene.bind("keydown", self.keydown)
        self.last_key = None
        self.slider = slider(bind=(lambda _: None), length=1800, width=50) # No bind needed
        self.print = self.envs[0].scene.append_to_caption
        self.print("\n")

    def __exit__(self, exc_type, exc_value, traceback): 
        self.envs[0].scene.caption = ""
        for env in self.envs: env.close()
        self.slider.delete()

    def __call__(self, i, j, n_interp=3):
        # Gather states, adding final next_state
        states = (np.vstack((self.graph.nodes[i]["states"], self.graph.nodes[i]["next_states"][-1:])),
                  np.vstack((self.graph.nodes[j]["states"], self.graph.nodes[j]["next_states"][-1:])))
        # Extract pose information from states (use hdg, pitch, roll representation) and perform interpolation        
        T = (len(states[0]), len(states[1]))
        poses = tuple((interp1d(range(T[e]), states_to_poses(s), axis=0, 
                kind=
                # "nearest" if e == 0 else # For debugging interpolation; set i = j for side-by-side comparison
                "linear" if len(s) <= 2 else "quadratic" if len(s) == 3 else "cubic")
                for e, s in enumerate(states)))
        t = [0, 0]
        for env, i_or_j in zip(self.envs, (i, j)): 
            env.ep_num_label.text = f"Episode {i_or_j}"
            env.t = 0
        self.slider.value = 0.5
        while True: 
            if self.last_key != " ": break # Press anything other than spacebar to start playback
        while True: 
            for e, env in enumerate(self.envs):
                if t[e] is not None:
                    x, y, z, tx, ty, tz, h, p, r, th, tp, tr = poses[e](t[e])
                    axis, up = dir_to_vec(h, p, r)
                    taxis, tup = dir_to_vec(th, tp, tr)
                    if False: # Sense check at non-interpolated timesteps
                        if env.t % n_interp == 0:
                            idx = int(round(env.t / n_interp))
                            x_or, y_or, z_or, _,_,_,_,_,_,_,_,_, fx, fy, fz, ux, uy, uz, _, \
                                tx_or, ty_or, tz_or, _,_,_,_,_,_,_,_,_, tfx, tfy, tfz, tux, tuy, tuz = states[e][idx]
                            axis_or, up_or, taxis_or, tup_or = \
                                vector(fx, fy, fz), vector(ux, uy, uz), vector(tfx, tfy, tfz), vector(tux, tuy, tuz)
                            pos_error = np.array([x, y, z, tx, ty, tz]) - np.array([x_or, y_or, z_or, tx_or, ty_or, tz_or])
                            assert np.isclose(pos_error, 0.0).all(), pos_error
                            dp = dot(axis, axis_or)
                            assert np.isclose(dp, 1.0)
                            dp = dot(up, up_or)
                            assert np.isclose(dp, 1.0)
                            dp = dot(taxis, taxis_or)
                            assert np.isclose(dp, 1.0)
                            dp = dot(tup, tup_or)
                            assert np.isclose(dp, 1.0)
                    env.jets[0]._set(pos=vector(x,y,z),      axis=axis,  up=up)
                    env.jets[1]._set(pos=vector(tx, ty, tz), axis=taxis, up=tup)
                    env.render()
                    t[e] += 1 / n_interp
                    env.t += 1
                    if t[e] >= T[e]-1: t[e] = None
            if t[0] == t[1] == None: # If different lengths, reset at same time
                t = [0, 0] 
                for env in self.envs: env.t = 0
            if self.last_key == "esc": return "esc" 
            elif self.last_key == " ": return (1. - self.slider.value)

    def keydown(self, ev): 
        if ev.key == "left": self.slider.value = max(0., self.slider.value - 0.5)
        elif ev.key == "right": self.slider.value = min(1., self.slider.value + 0.5)
        self.last_key = ev.key


def states_to_poses(states):
    poses = []
    for x, y, z, _,_,_,_,_,_,_,_,_, fx, fy, fz, ux, uy, uz, _, \
        tx, ty, tz, _,_,_,_,_,_,_,_,_, tfx, tfy, tfz, tux, tuy, tuz in states:
        h,  p,  r  = vec_to_dir(vector(fx, fy, fz),    vector(ux, uy, uz))
        th, tp, tr = vec_to_dir(vector(tfx, tfy, tfz), vector(tux, tuy, tuz))
        poses.append([x, y, z, tx, ty, tz, h, p, r, th, tp, tr])
    poses = np.array(poses)        
    # Implement wraparound on angular dimensions (6:12)
    delta = poses[1:,6:12] - poses[:-1,6:12]
    abs_delta = np.abs(delta)
    wrap = np.zeros_like(delta)
    two_pi = 2*np.pi
    wrap[abs_delta > np.abs(delta+two_pi)] = two_pi
    wrap[abs_delta > np.abs(delta-two_pi)] = -two_pi
    poses[1:,6:12] += np.cumsum(wrap, axis=0)
    return poses

P = {
    "pbrl": {
        "interface": {
            "class": FastJetInterface
        }
    }
}
