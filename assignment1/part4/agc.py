import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def compute_frame_rms_db(
    audio: np.ndarray, 
    sr: int, 
    win_ms: float = 20.0, 
    hop_ms: float = 10.0
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute frame-wise RMS in dB using vectorized framing.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    sr : int
        Sampling rate in Hz.
    win_ms : float
        Window size in milliseconds (default: 20ms).
    hop_ms : float
        Hop size in milliseconds (default: 10ms).
        
    Returns
    -------
    rms : np.ndarray
        RMS per frame (linear).
    rms_db : np.ndarray
        RMS per frame in dB.
    n_fft : int
        Window size in samples.
    hop_length : int
        Hop size in samples.
    """
    n_fft = int(win_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    
    # Vectorized framing using advanced indexing
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    indices = np.arange(n_fft)[None, :] + hop_length * np.arange(num_frames)[:, None]
    frames = audio[indices]
    
    # Compute RMS per frame (vectorized)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    
    # Convert to dB (with small epsilon to avoid log(0))
    eps = 1e-10
    rms_db = 20 * np.log10(rms + eps)
    
    return rms, rms_db, n_fft, hop_length


def estimate_agc_params(
    rms_db: np.ndarray,
    desired_percentile: float = 75.0,
    noise_floor_percentile: float = 20.0
) -> tuple[float, float]:
    """
    Estimate AGC parameters: desired RMS and noise floor threshold.
    
    Parameters
    ----------
    rms_db : np.ndarray
        RMS per frame in dB.
    desired_percentile : float
        Percentile for desired RMS (over speech frames).
    noise_floor_percentile : float
        Percentile for noise floor threshold.
        
    Returns
    -------
    desired_rms_db : float
        Target RMS level in dB.
    noise_floor_db : float
        Noise floor threshold in dB.
    """
    # Noise floor: 20th percentile of all frames
    noise_floor_db = np.percentile(rms_db, noise_floor_percentile)
    
    # Speech frames: frames above noise floor
    speech_mask = rms_db > noise_floor_db
    speech_rms_db = rms_db[speech_mask]
    
    # Desired RMS: 75th percentile of speech frames
    if len(speech_rms_db) > 0:
        desired_rms_db = np.percentile(speech_rms_db, desired_percentile)
    else:
        # Fallback: use median of all frames
        desired_rms_db = np.median(rms_db)
    
    return desired_rms_db, noise_floor_db


def compute_agc_gains(
    rms_db: np.ndarray,
    desired_rms_db: float,
    noise_floor_db: float,
    stats_window_frames: int = 100,
    attack_coef: float = 0.1,
    release_coef: float = 0.01
) -> np.ndarray:
    """
    Compute sequential AGC gains with attack/release smoothing.
    
    Uses a ~1s statistics window (100 frames at 10ms hop) for running RMS estimation.
    
    Parameters
    ----------
    rms_db : np.ndarray
        RMS per frame in dB.
    desired_rms_db : float
        Target RMS level in dB.
    noise_floor_db : float
        Noise floor threshold in dB (frames below this won't be amplified).
    stats_window_frames : int
        Number of frames for running statistics (~1s = 100 frames at 10ms hop).
    attack_coef : float
        Smoothing coefficient for attack (gain decreasing) - faster.
    release_coef : float
        Smoothing coefficient for release (gain increasing) - slower.
        
    Returns
    -------
    gains_db : np.ndarray
        Smoothed gain per frame in dB.
    """
    print("\n--- Part 4.a.iii: Computing AGC Gains ---")
    
    num_frames = len(rms_db)
    gains_db = np.zeros(num_frames, dtype=np.float32)
    
    # Ring buffer for running statistics
    buffer = np.full(stats_window_frames, rms_db[0], dtype=np.float32)
    buffer_idx = 0
    
    # Initialize smoothed gain
    initial_rms_stat = np.mean(buffer)
    prev_gain_db = desired_rms_db - initial_rms_stat
    
    # Sequential gain computation (inherently sequential due to attack/release)
    for t in range(num_frames):
        # Update ring buffer
        buffer[buffer_idx] = rms_db[t]
        buffer_idx = (buffer_idx + 1) % stats_window_frames
        
        # Compute running RMS statistic (mean of buffer)
        rms_stat_db = np.mean(buffer)
        
        # Target gain to reach desired RMS
        target_gain_db = desired_rms_db - rms_stat_db
        
        # Noise floor gating: don't amplify frames below noise floor
        if rms_db[t] < noise_floor_db:
            target_gain_db = min(target_gain_db, 0.0)
        
        # Attack/release smoothing
        if target_gain_db < prev_gain_db:
            # Attack: gain is decreasing (need to attenuate) - faster response
            coef = attack_coef
        else:
            # Release: gain is increasing (amplifying) - slower response
            coef = release_coef
        
        # Exponential smoothing
        smoothed_gain_db = prev_gain_db + coef * (target_gain_db - prev_gain_db)
        
        gains_db[t] = smoothed_gain_db
        prev_gain_db = smoothed_gain_db
    
    print(f"-> Computed gains for {num_frames} frames")
    print(f"-> Gain range: {gains_db.min():.2f} dB to {gains_db.max():.2f} dB")
    
    return gains_db


def apply_agc(
    audio: np.ndarray,
    gains_db: np.ndarray,
    n_fft: int,
    hop_length: int
) -> np.ndarray:
    """
    Apply frame-wise gains to audio using overlap-add.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    gains_db : np.ndarray
        Gain per frame in dB.
    n_fft : int
        Window size in samples.
    hop_length : int
        Hop size in samples.
        
    Returns
    -------
    output : np.ndarray
        Gain-adjusted audio signal.
    """
    print("\n--- Part 4.a.iii: Applying AGC Gains ---")
    
    num_frames = len(gains_db)
    
    # Convert gains from dB to linear
    gains_linear = 10 ** (gains_db / 20.0)
    
    # Create Hann window for smooth overlap-add
    window = np.hanning(n_fft)
    
    # Vectorized framing
    indices = np.arange(n_fft)[None, :] + hop_length * np.arange(num_frames)[:, None]
    frames = audio[indices]
    
    # Apply gains to each frame (vectorized)
    gained_frames = frames * gains_linear[:, None]
    
    # Apply synthesis window
    windowed_frames = gained_frames * window[None, :]
    
    # Overlap-add reconstruction (vectorized using np.add.at)
    output_length = (num_frames - 1) * hop_length + n_fft
    output = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)
    
    frame_starts = np.arange(num_frames) * hop_length
    output_indices = frame_starts[:, None] + np.arange(n_fft)[None, :]
    
    flat_indices = output_indices.ravel()
    flat_audio = windowed_frames.ravel()
    flat_window = np.tile(window ** 2, num_frames)
    
    np.add.at(output, flat_indices, flat_audio)
    np.add.at(window_sum, flat_indices, flat_window)
    
    # Normalize by window sum
    window_sum = np.maximum(window_sum, 1e-8)
    output = output / window_sum
    
    # Trim to original length
    output = output[:len(audio)]
    
    # Apply fade-in/fade-out to avoid edge spikes
    fade_len = n_fft
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    output[:fade_len] *= fade_in
    output[-fade_len:] *= fade_out
    
    print(f"-> Applied AGC to {len(output)} samples")
    
    return output.astype(np.float32)


def soft_clip_sigmoid(audio: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """
    Apply soft clipping using tanh to avoid overflow.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (may have values > 1.0).
    drive : float
        Drive parameter (higher = harder clipping, default: 1.0 for gentle).
        
    Returns
    -------
    clipped : np.ndarray
        Soft-clipped audio signal in range (-1, 1).
    """
    print("\n--- Part 4.a.iv: Applying Soft Clipping ---")
    
    # Check if clipping is needed
    max_val = np.abs(audio).max()
    print(f"-> Max amplitude before clipping: {max_val:.4f}")
    
    if max_val <= 1.0:
        print("-> No clipping needed (signal already in range)")
        return audio.astype(np.float32)
    
    # Soft clip using tanh: y = tanh(drive * x) / tanh(drive)
    # This maps [-inf, inf] to (-1, 1) smoothly
    clipped = np.tanh(drive * audio) / np.tanh(drive)
    
    print(f"-> Max amplitude after clipping: {np.abs(clipped).max():.4f}")
    
    return clipped.astype(np.float32)


def plot_gain_curve(
    gains_db: np.ndarray,
    sr: int,
    hop_length: int,
    output_path: str = "part4_agc_gains.png"
):
    """
    Plot AGC scaling factors vs time.
    
    Parameters
    ----------
    gains_db : np.ndarray
        Gain per frame in dB.
    sr : int
        Sampling rate in Hz.
    hop_length : int
        Hop size in samples.
    output_path : str
        Output path for the plot.
    """
    print("\n--- Part 4.a.vi: Plotting AGC Gains ---")
    
    num_frames = len(gains_db)
    time = np.arange(num_frames) * hop_length / sr
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(time, gains_db, color='blue', linewidth=1.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='0 dB (unity gain)')
    
    ax.set_title("Part 4.a.vi: AGC Scaling Factors vs Time")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Gain [dB]")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"-> Saved plot to: {output_path}")
    plt.close()

