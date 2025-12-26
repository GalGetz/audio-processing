import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def compute_frame_energy(audio: np.ndarray, sr: int, win_ms: float = 20.0, hop_ms: float = 10.0) -> tuple[np.ndarray, int, int]:
    """
    Compute frame-wise energy using vectorized framing.
    
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
    energy : np.ndarray
        Energy per frame.
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
    
    # Compute energy per frame: E_t = sum(x_t[n]^2)
    energy = np.sum(frames**2, axis=1)
    
    return energy, n_fft, hop_length


def compute_vad_mask(energy: np.ndarray, threshold_percentile: float = 65.0) -> tuple[np.ndarray, float]:
    """
    Compute Voice Activity Detection mask based on energy threshold.
    
    Parameters
    ----------
    energy : np.ndarray
        Energy per frame.
    threshold_percentile : float
        Percentile of energy to use as threshold (default: 65).
        
    Returns
    -------
    is_speech : np.ndarray
        Boolean mask where True indicates speech frames.
    threshold : float
        The computed threshold value.
    """
    # Use percentile-based threshold for robustness
    threshold = np.percentile(energy, threshold_percentile)
    
    # Create VAD mask: frames with energy above threshold are speech
    is_speech = energy > threshold
    
    return is_speech, threshold


def plot_vad_threshold(energy: np.ndarray, threshold: float, sr: int, hop_length: int, is_speech: np.ndarray = None):
    """
    Plot energy contour with threshold line overlay.
    
    Parameters
    ----------
    energy : np.ndarray
        Energy per frame.
    threshold : float
        Threshold value.
    sr : int
        Sampling rate in Hz.
    hop_length : int
        Hop size in samples.
    is_speech : np.ndarray, optional
        VAD mask to highlight speech regions.
    """
    print("\n--- Part 3.a: Plotting VAD Threshold ---")
    
    # Time axis
    num_frames = len(energy)
    time = np.arange(num_frames) * hop_length / sr
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot energy contour
    ax.plot(time, energy, label='Energy', color='blue', alpha=0.8)
    
    # Plot threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    # Highlight speech regions if provided
    if is_speech is not None:
        speech_mask = is_speech.astype(float) * energy.max() * 0.1
        ax.fill_between(time, 0, speech_mask, alpha=0.3, color='green', label='Speech Regions')
    
    ax.set_title("Part 3.a: Voice Activity Detection (Energy Threshold)")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    output_filename = "part3_vad_threshold.png"
    plt.savefig(output_filename)
    print(f"-> Saved plot to: {output_filename}")
    plt.close()

