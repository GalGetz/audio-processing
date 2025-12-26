import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def stft(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform using vectorized framing.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    n_fft : int
        Window/FFT size in samples.
    hop_length : int
        Hop size in samples.
    window : np.ndarray
        Window function (e.g., Hann).
        
    Returns
    -------
    stft_complex : np.ndarray
        Complex STFT matrix, shape (num_frames, n_fft//2 + 1).
    """
    # Vectorized framing using advanced indexing
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    indices = np.arange(n_fft)[None, :] + hop_length * np.arange(num_frames)[:, None]
    frames = audio[indices]
    
    # Apply window to all frames (vectorized)
    windowed_frames = frames * window[None, :]
    
    # Compute rFFT for all frames at once
    stft_complex = np.fft.rfft(windowed_frames, axis=1)
    
    return stft_complex


def istft(
    stft_complex: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
    output_length: int = None
) -> np.ndarray:
    """
    Compute Inverse STFT using vectorized overlap-add.
    
    Parameters
    ----------
    stft_complex : np.ndarray
        Complex STFT matrix, shape (num_frames, n_fft//2 + 1).
    n_fft : int
        Window/FFT size in samples.
    hop_length : int
        Hop size in samples.
    window : np.ndarray
        Window function for synthesis.
    output_length : int, optional
        Desired output length (will be trimmed/padded).
        
    Returns
    -------
    audio : np.ndarray
        Reconstructed audio signal.
    """
    num_frames = stft_complex.shape[0]
    
    # Inverse rFFT for all frames (vectorized)
    frames = np.fft.irfft(stft_complex, n=n_fft, axis=1)
    
    # Apply synthesis window
    windowed_frames = frames * window[None, :]
    
    # Overlap-add reconstruction (vectorized using np.add.at)
    total_length = (num_frames - 1) * hop_length + n_fft
    output = np.zeros(total_length, dtype=np.float64)
    window_sum = np.zeros(total_length, dtype=np.float64)
    
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
    
    # Trim to desired length
    if output_length is not None:
        output = output[:output_length]
    
    return output.astype(np.float32)


def phase_vocoder(
    stft_complex: np.ndarray,
    rate: float,
    hop_length: int
) -> np.ndarray:
    """
    Apply phase vocoder for time-stretching (improved implementation).
    
    Uses proper instantaneous frequency estimation for better phase coherence.
    
    Parameters
    ----------
    stft_complex : np.ndarray
        Complex STFT matrix, shape (num_frames, num_bins).
    rate : float
        Time-stretch rate (1.5 = 1.5x speed, i.e. shorter output).
    hop_length : int
        Hop size in samples.
        
    Returns
    -------
    output_stft : np.ndarray
        Time-stretched complex STFT.
    """
    num_frames_in, num_bins = stft_complex.shape
    
    # Compute number of output frames
    num_frames_out = int(np.ceil(num_frames_in / rate))
    
    # Precompute magnitude and phase of input
    magnitude_in = np.abs(stft_complex)
    phase_in = np.angle(stft_complex)
    
    # Output STFT
    output_stft = np.zeros((num_frames_out, num_bins), dtype=np.complex128)
    
    # Expected phase advance per hop for each frequency bin
    # omega_k = 2 * pi * k / n_fft
    n_fft = (num_bins - 1) * 2
    freq_bins = np.arange(num_bins)
    omega = 2 * np.pi * freq_bins / n_fft
    expected_phase_advance = omega * hop_length
    
    # Compute instantaneous frequency for all input frames
    # Phase difference between consecutive frames
    phase_diff = np.diff(phase_in, axis=0)
    
    # Remove expected phase advance to get deviation
    phase_deviation = phase_diff - expected_phase_advance[None, :]
    
    # Wrap to [-pi, pi]
    phase_deviation = np.mod(phase_deviation + np.pi, 2 * np.pi) - np.pi
    
    # True instantaneous frequency = expected + deviation/hop
    inst_freq = omega[None, :] + phase_deviation / hop_length
    
    # Initialize output phase with first input frame
    phase_acc = phase_in[0].copy()
    
    # Output hop (in input frame units) 
    output_hop = rate
    
    # Sequential phase vocoder
    for t_out in range(num_frames_out):
        # Map output frame to input time (monotonic mapping)
        t_in = t_out * rate
        
        # Find adjacent input frames
        t0 = int(np.floor(t_in))
        t1 = min(t0 + 1, num_frames_in - 1)
        
        # Interpolation weight
        w = t_in - t0
        
        # Magnitude interpolation
        mag_out = (1 - w) * magnitude_in[t0] + w * magnitude_in[t1]
        
        if t_out == 0:
            # First frame: use input phase directly
            phase_out = phase_in[t0]
            phase_acc = phase_out.copy()
        else:
            # Get instantaneous frequency at this input position
            # Use the frame before t0 for the inst_freq (since inst_freq[i] is diff between frame i and i+1)
            if t0 > 0:
                inst_freq_interp = (1 - w) * inst_freq[t0 - 1] + w * inst_freq[min(t0, num_frames_in - 2)]
            else:
                inst_freq_interp = inst_freq[0]
            
            # Advance phase by output hop * instantaneous frequency
            phase_acc = phase_acc + hop_length * inst_freq_interp
            phase_out = phase_acc
        
        # Recompose complex STFT
        output_stft[t_out] = mag_out * np.exp(1j * phase_out)
    
    return output_stft


def time_stretch_phase_vocoder(
    audio: np.ndarray,
    sr: int,
    rate: float = 1.5,
    win_ms: float = 40.0,
    hop_ms: float = 10.0
) -> np.ndarray:
    """
    Time-stretch audio using phase vocoder while preserving pitch.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    sr : int
        Sampling rate in Hz.
    rate : float
        Time-stretch rate (1.5 = 1.5x speed, i.e. shorter output).
    win_ms : float
        Window size in milliseconds (default: 40ms for better freq resolution).
    hop_ms : float
        Hop size in milliseconds (default: 10ms = 75% overlap with 40ms window).
        
    Returns
    -------
    stretched : np.ndarray
        Time-stretched audio signal.
    """
    print(f"\n--- Part 5: Time-Stretching (x{rate}) with Phase Vocoder ---")
    
    n_fft = int(win_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    
    # Use Hann window (better for OLA reconstruction than Hamming)
    window = np.hanning(n_fft)
    
    print(f"-> Window: {win_ms}ms ({n_fft} samples), Hop: {hop_ms}ms ({hop_length} samples)")
    print(f"-> Overlap: {100 * (1 - hop_length/n_fft):.0f}%")
    
    # 5.a.ii: Apply STFT
    print(f"-> Applying STFT...")
    stft_complex = stft(audio, n_fft, hop_length, window)
    print(f"-> Input STFT shape: {stft_complex.shape}")
    
    # 5.a.i & 5.a.iii: Phase vocoder (mapping + magnitude/phase computation)
    print(f"-> Applying phase vocoder (rate={rate})...")
    output_stft = phase_vocoder(stft_complex, rate, hop_length)
    print(f"-> Output STFT shape: {output_stft.shape}")
    
    # 5.a.iv: Apply iSTFT
    print("-> Applying iSTFT...")
    expected_length = int(len(audio) / rate)
    stretched = istft(output_stft, n_fft, hop_length, window, output_length=expected_length)
    
    print(f"-> Original length: {len(audio)} samples ({len(audio)/sr:.2f}s)")
    print(f"-> Stretched length: {len(stretched)} samples ({len(stretched)/sr:.2f}s)")
    
    return stretched


def plot_time_stretch_comparison(
    original: np.ndarray,
    stretched: np.ndarray,
    sr: int,
    rate: float,
    output_filename: str = "part5_time_stretch.png"
):
    """
    Plot original vs time-stretched audio in time and spectral domains.
    
    Parameters
    ----------
    original : np.ndarray
        Original audio signal.
    stretched : np.ndarray
        Time-stretched audio signal.
    sr : int
        Sampling rate in Hz.
    rate : float
        Time-stretch rate used.
    output_filename : str
        Output filename for the plot.
    """
    print(f"\n--- Part 5.a.v: Plotting Time-Stretch Comparison ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Part 5: Time-Stretching (x{rate}) with Phase Vocoder", fontsize=14)
    
    # Time axes
    time_orig = np.arange(len(original)) / sr
    time_stretched = np.arange(len(stretched)) / sr
    
    # Top-left: Original waveform
    axes[0, 0].plot(time_orig, original, color='blue', linewidth=0.5)
    axes[0, 0].set_title(f"Original Waveform ({len(original)/sr:.2f}s)")
    axes[0, 0].set_xlabel("Time [sec]")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)
    
    # Top-right: Stretched waveform
    axes[0, 1].plot(time_stretched, stretched, color='green', linewidth=0.5)
    axes[0, 1].set_title(f"Stretched Waveform ({len(stretched)/sr:.2f}s)")
    axes[0, 1].set_xlabel("Time [sec]")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True)
    
    # Spectrogram parameters
    n_fft = int(0.04 * sr)  # 40ms
    hop_length = int(0.01 * sr)  # 10ms
    
    # Bottom-left: Original spectrogram
    f_orig, t_orig, Sxx_orig = signal.spectrogram(
        original, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, window='hann'
    )
    Sxx_orig_db = 10 * np.log10(Sxx_orig + 1e-10)
    axes[1, 0].pcolormesh(t_orig, f_orig, Sxx_orig_db, shading='gouraud', cmap='magma')
    axes[1, 0].set_title("Original Spectrogram")
    axes[1, 0].set_xlabel("Time [sec]")
    axes[1, 0].set_ylabel("Frequency [Hz]")
    
    # Bottom-right: Stretched spectrogram
    f_str, t_str, Sxx_str = signal.spectrogram(
        stretched, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, window='hann'
    )
    Sxx_str_db = 10 * np.log10(Sxx_str + 1e-10)
    axes[1, 1].pcolormesh(t_str, f_str, Sxx_str_db, shading='gouraud', cmap='magma')
    axes[1, 1].set_title("Stretched Spectrogram")
    axes[1, 1].set_xlabel("Time [sec]")
    axes[1, 1].set_ylabel("Frequency [Hz]")
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"-> Saved plot to: {output_filename}")
    plt.close()
