from .agc import (
    compute_frame_rms_db,
    estimate_agc_params,
    compute_agc_gains,
    apply_agc,
    soft_clip_sigmoid,
    plot_gain_curve,
)

__all__ = [
    'compute_frame_rms_db',
    'estimate_agc_params',
    'compute_agc_gains',
    'apply_agc',
    'soft_clip_sigmoid',
    'plot_gain_curve',
]

