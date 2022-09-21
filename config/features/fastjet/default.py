from . import *

P = {
    "pbrl": {
        "model": {
            "featuriser": {
                "preprocessor": preprocessor,
                "features": [
                    dist,
                    closing_speed,
                    alt,
                    # target_alt,
                    alt_error,
                    delta_alt_error,
                    dist_xz,
                    delta_dist_xz,
                    # pitch,
                    # abs_pitch,
                    # target_pitch,
                    pitch_error,
                    delta_pitch_error,
                    # roll,
                    abs_roll,
                    # target_roll,
                    roll_error,
                    delta_roll_error,
                    hdg_error,
                    delta_hdg_error,
                    fwd_error,
                    delta_fwd_error,
                    up_error,
                    delta_up_error,
                    right_error,
                    delta_right_error,
                    los_error,
                    delta_los_error,
                    # lr_offset,
                    abs_lr_offset,
                    speed,
                    g_force,
                    pitch_rate,
                    roll_rate,
                    yaw_rate,
                    thrust,
                    delta_thrust
                ]
            }
        }
    }
}
