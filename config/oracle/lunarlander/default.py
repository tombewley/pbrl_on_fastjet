from rlutils.observers.pbrl.interfaces import OracleInterface
import numpy as np

class old:
    def __call__(self, ti, tj): 
        self.reset()
        ri = sum([self.reward_function(x) for x in ti])
        self.reset()
        rj = sum([self.reward_function(x) for x in tj])
        return ri - rj

    def reset(self): self.prev_shaping = None; self.done = False 

    def reward_function(self, x):
        """Reward function for LunarLanderContinuous-v2."""
        if self.done: return 0
        pos_x, pos_y, vel_x, vel_y, ang, _, left_contact, right_contact, main_engine, side_engine, _, crash, _, land = x
        if crash or abs(pos_x) >= 1.0: self.done = True; reward = -100
        if land: print("LAND"); self.done = True; reward = +100 # NOTE: Needed to overwrite crash?
        else:
            # Shaping reward.
            reward = 0
            shaping = \
                - 100*np.sqrt(pos_x**2 + pos_y**2) \
                - 100*np.sqrt(vel_x**2 + vel_y**2) \
                - 100*abs(ang) \
                + 10*left_contact + 10*right_contact 
            if self.prev_shaping is not None:
                reward = shaping - self.prev_shaping
            self.prev_shaping = shaping
            # Engine usage cost.
            if main_engine > 0.0:
                m_power = (np.clip(main_engine, 0.0,1.0) + 1.0)*0.5  
                assert m_power >= 0.5 and m_power <= 1.0
            else: m_power = 0.0
            if np.abs(side_engine) > 0.5:
                s_power = np.clip(np.abs(side_engine), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else: s_power = 0.0
            reward -= m_power*0.30  
            reward -= s_power*0.03
        return reward

def oracle(tr):
    print(tr.shape)
    raise Exception

# Wrapper to prevent repeated reward after crash/land in LunarLander.
if P["env"] == "LunarLanderContinuous-v2":
    class LlStopRepeatRewardWrapper(gym.Wrapper):
        def __init__(self, env): self.env = env; super().__init__(env)
        def reset(self): self.done = False; return self.env.reset()    
        def step(self, action): 
            next_state, reward, done, info = self.env.step(action)
            if self.done: reward = 0.
            elif self.env.game_over or not(self.env.lander.awake) or abs(next_state[0]) > 1.0:
                self.done = True
            return next_state, reward, done, info
    env = LlStopRepeatRewardWrapper(env)

P = {"pbrl": {"interface": {"class": OracleInterface, "oracle": oracle}}}
