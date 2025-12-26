import numpy as np


def spectral_subtraction(
    noisy_audio: np.ndarray,
    is_speech: np.ndarray,
    sr: int,
    win_ms: float = 20.0,
    hop_ms: float = 10.0,
    alpha: float = 0.95,
    beta: float = 1.0,
    floor: float = 0.05
) -> np.ndarray:
    """
    Apply sequential spectral subtraction to enhance noisy audio.
    
    Parameters
    ----------
    noisy_audio : np.ndarray
        Input noisy audio signal.
    is_speech : np.ndarray
        VAD mask (True = speech, False = noise).
    sr : int
        Sampling rate in Hz.
    win_ms : float
        Window size in milliseconds (default: 20ms).
    hop_ms : float
        Hop size in milliseconds (default: 10ms).
    alpha : float
        Smoothing factor for noise estimation (default: 0.95).
    beta : float
        Subtraction factor (default: 1.0).
    floor : float
        Spectral floor to prevent negative values (default: 0.05).
        
    Returns
    -------
    enhanced_audio : np.ndarray
        Enhanced audio signal.
    """
    print("\n--- Part 3.b: Spectral Subtraction ---")
    
    n_fft = int(win_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    
    # Create Hann window
    window = np.hanning(n_fft)
    
    # Vectorized framing using advanced indexing
    num_frames = 1 + (len(noisy_audio) - n_fft) // hop_length
    indices = np.arange(n_fft)[None, :] + hop_length * np.arange(num_frames)[:, None]
    frames = noisy_audio[indices]
    
    # Apply window to all frames (vectorized)
    windowed_frames = frames * window[None, :]
    
    # Compute STFT - vectorized rFFT for all frames at once
    stft = np.fft.rfft(windowed_frames, axis=1)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    print(f"-> STFT computed: {num_frames} frames, {magnitude.shape[1]} frequency bins")
    
    # Initialize noise estimate from initial non-speech frames (vectorized)
    noise_frames_idx = np.where(~is_speech[:min(50, num_frames)])[0]
    noise_mag = np.mean(magnitude[noise_frames_idx], axis=0)
    print(f"-> Initialized noise estimate from {len(noise_frames_idx)} noise frames")
    
    # =========================================================================
    # Sequential noise estimation (inherently sequential - each frame depends
    # on the previous noise estimate, as required by the assignment)
    # All per-frame operations are vectorized (no inner loops)
    # =========================================================================
    noise_estimates = np.zeros_like(magnitude)
    
    for t in range(num_frames):
        if not is_speech[t]:
            noise_mag = alpha * noise_mag + (1 - alpha) * magnitude[t]
        noise_estimates[t] = noise_mag
    
    print(f"-> Sequential noise estimation complete")
    
    # Spectral subtraction - fully vectorized across all frames and frequency bins
    subtracted = magnitude - beta * noise_estimates
    floored = floor * noise_estimates
    enhanced_magnitude = np.maximum(subtracted, floored)
    
    print(f"-> Spectral subtraction applied to {num_frames} frames")
    
    # Reconstruct using original phase (vectorized)
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    
    # Inverse FFT - vectorized across all frames
    enhanced_frames = np.fft.irfft(enhanced_stft, n=n_fft, axis=1)
    
    # Apply synthesis window (vectorized)
    windowed_output = enhanced_frames * window[None, :]
    
    # =========================================================================
    # Overlap-add reconstruction - vectorized using np.add.at
    # =========================================================================
    output_length = (num_frames - 1) * hop_length + n_fft
    enhanced_audio = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)
    
    # Create index array for all frames (vectorized)
    frame_starts = np.arange(num_frames) * hop_length
    output_indices = frame_starts[:, None] + np.arange(n_fft)[None, :]
    
    # Flatten for np.add.at
    flat_indices = output_indices.ravel()
    flat_audio = windowed_output.ravel()
    flat_window = np.tile(window ** 2, num_frames)
    
    # Vectorized overlap-add using np.add.at
    np.add.at(enhanced_audio, flat_indices, flat_audio)
    np.add.at(window_sum, flat_indices, flat_window)
    
    # Normalize by window sum (vectorized)
    window_sum = np.maximum(window_sum, 1e-8)
    enhanced_audio = enhanced_audio / window_sum
    
    # Trim to original length
    enhanced_audio = enhanced_audio[:len(noisy_audio)]
    
    print(f"-> Overlap-add reconstruction complete: {len(enhanced_audio)} samples")
    
    return enhanced_audio.astype(np.float32)
