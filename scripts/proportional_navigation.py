"""

Baseline: proportional navigation.

See:
    Moran, Inanc, and Turgay Altilar. "Three plane approach for 3D true proportional navigation." 
    In AIAA Guidance, Navigation, and Control Conference and Exhibit, p. 6457. 2005.

"Virtual sliding target"
    https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000692

"""

from vpython import *
import numpy as np
from tqdm import trange
import json
import matplotlib.pyplot as plt

from fastjet.env import FastJetEnv
from fastjet.global_variables import HZ, RENDER_SCALE
from rlutils.observers.pbrl import PbrlObserver, OracleInterface
from config.oracles import target_pose_tree
from config.features import F

RENDER = True

NUM_EPISODES = 10000
MAX_TIMESTEPS_PER_EPISODE = 500
SKIP_FRAMES = 25
N = 3
MAX_ROT_RATE = 0.01

env = FastJetEnv(
    task="target_no_reward", 
    render_mode="human" if RENDER else False,
    skip_frames=SKIP_FRAMES,
    camera_angle="bbox"
    )

pbrl = PbrlObserver({"interface": (OracleInterface, target_pose_tree), "reward_source": "oracle"}, features=F)

dt = env.skip_frames / HZ * RENDER_SCALE
rot_clip = env.skip_frames * MAX_ROT_RATE

reward_sum = []
ep_length = []
completed = []

for ep in trange(NUM_EPISODES):

    while len(env.jets) > 2: env.remove_jet(2) 
    
    obs = env.reset()
    phase_counts = np.array([0 for _ in range(env.num_phases)])
    
    for i in range(0):
        env.add_jet(arrow_length=env.jets[1].arrow_length, show_sphere=True)
        env.jets[-1]._set(pos=env.jets[1].pos - 100*(i+1)*env.jets[1].axis, axis=env.jets[1].axis, up=env.jets[1].up)
    target = len(env.jets)-1

    for _ in range(int(MAX_TIMESTEPS_PER_EPISODE / env.skip_frames)):

        delta   =  env.jets[target].pos                       -  env.jets[0].pos
        delta_n = (env.jets[target].pos + env.jets[1].vel*dt) - (env.jets[0].pos + env.jets[0].vel*dt)
           
        los   = np.arctan2([delta.y,   delta.z,   delta.x  ], [delta.x,   delta.y,   delta.z  ]) # Order is xy, yz, zx 
        los_n = np.arctan2([delta_n.y, delta_n.z, delta_n.x], [delta_n.x, delta_n.y, delta_n.z])

        los_rate = los_n - los
        los_rate = ((los_rate + np.pi) % (2 * np.pi)) - np.pi # https://stackoverflow.com/a/7869457

        env.add_jet()
        env.jets[-1]._set(pos=env.jets[0].pos, axis=env.jets[0].axis, up=env.jets[0].up)

        rot = np.clip(N*los_rate, -rot_clip, rot_clip)

        axis, up = env.jets[0].axis, env.jets[0].up
        axis = axis.rotate(angle=rot[0], axis=vector(0,0,1)) 
        axis = axis.rotate(angle=rot[1], axis=vector(1,0,0)) 
        axis = axis.rotate(angle=rot[2], axis=vector(0,1,0)) 
        up = up.rotate(angle=rot[0], axis=vector(0,0,1)) 
        up = up.rotate(angle=rot[1], axis=vector(1,0,0)) 
        up = up.rotate(angle=rot[2], axis=vector(0,1,0)) 
        env.jets[0]._set(axis=axis, up=up)
        
        action = np.array([0,0,0,0])
        next_obs, _, done, info = env.step(action)
        phase_counts += info["phase"]
        if RENDER: env.render(); #time.sleep(1)

        pbrl.per_timestep(None, None, obs, action, next_obs, None, None, None, None)
        
        if target > 1 and mag(env.jets[target].pos - env.jets[0].pos) < env.jets[target].arrow_length:
            env.remove_jet(target)
            target -= 1
        if done: break    

        obs = next_obs

    reward_sum.append(pbrl.per_episode(ep)["reward_sum_oracle"])
    ep_length.append(int(phase_counts[0]))
    completed.append(int(phase_counts[1]))

with open("pn.json", "w") as f:
    json.dump({"return": reward_sum, "length": ep_length, "completion": completed}, f)

if False:
    reward_sum = np.array(reward_sum)
    ep_length = np.array(ep_length)
    completed = np.array(completed)

    _, axes = plt.subplots(1, 3)
    axes[0].hist(reward_sum)
    axes[0].set_title("Oracle Return")
    axes[1].hist(ep_length)
    axes[1].set_title("Episode Length")
    axes[2].hist(completed)
    axes[2].set_title("Episode Completion")

    print(np.median(reward_sum), reward_sum.mean(), reward_sum.std())
    print(np.median(ep_length), ep_length.mean(), ep_length.std())
    print(np.median(completed), completed.mean(), completed.std())

    plt.show()