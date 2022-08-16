from torch import sqrt, abs, zeros_like
from rlutils.rewards.interfaces import OracleInterface

def oracle(s, a, ns):
    def shaping(pos_x, pos_y, vel_x, vel_y, ang, left_contact, right_contact):
        """Potential-based shaping"""
        return - 100.*sqrt(pos_x**2 + pos_y**2) \
               - 100.*sqrt(vel_x**2 + vel_y**2) \
               - 100.*abs(ang) \
               + 10.*left_contact + 10.*right_contact
    def engine_usage(main_engine, side_engine):
        """Engine usage penalty"""
        r = zeros_like(main_engine)
        m_act, s_act = main_engine > 0., abs(side_engine) > 0.5
        r[m_act] -= 0.15 * (main_engine[m_act] + 1.)
        r[s_act] -= 0.03 * abs(side_engine[s_act])
        return r
    def descent(pos_x, pos_y):
        """Incentive to move towards the pad"""
        return - 2. * sqrt(pos_x**2 + pos_y**2)
    def landing_bonus(pos_x, left_contact, right_contact):
        """Bonus for being central with both legs on the ground"""
        return 5. * (abs(pos_x) <= 0.2) * left_contact * right_contact

    return shaping(ns[...,0], ns[...,1], ns[...,2], ns[...,3], ns[...,4], ns[...,6], ns[...,7]) - \
           shaping( s[...,0],  s[...,1],  s[...,2],  s[...,3],  s[...,4],  s[...,6],  s[...,7]) + \
           engine_usage(a[...,0], a[...,1]) + \
           descent(ns[...,0], ns[...,1]) + \
           landing_bonus(ns[...,0], ns[...,6], ns[...,7])

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        "save_path": "graphs_and_models/lunarlander/default",
        # "offline_graph_path": "offline_graphs/lunarlander/default/___.graph"
    }
}
