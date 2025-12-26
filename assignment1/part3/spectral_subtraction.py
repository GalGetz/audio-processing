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
    
    # Vectorized framing
    num_frames = 1 + (len(noisy_audio) - n_fft) // hop_length
    indices = np.arange(n_fft)[None, :] + hop_length * np.arange(num_frames)[:, None]
    frames = noisy_audio[indices]
    
    # Apply window to all frames
    windowed_frames = frames * window[None, :]
    
    # Compute STFT (rFFT for real signals)
    stft = np.fft.rfft(windowed_frames, axis=1)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    print(f"-> STFT computed: {num_frames} frames, {magnitude.shape[1]} frequency bins")
    
    # Initialize noise magnitude estimate using first few non-speech frames
    # Find initial noise frames (first N frames where is_speech is False)
    noise_frames_idx = np.where(~is_speech[:min(50, num_frames)])[0]
    
    if len(noise_frames_idx) > 0:
        noise_mag = np.mean(magnitude[noise_frames_idx], axis=0)
        print(f"-> Initialized noise estimate from {len(noise_frames_idx)} noise frames")
    else:
        # Fallback: use first frame or minimum magnitude
        noise_mag = magnitude[0].copy()
        print("-> Warning: No initial noise frames found, using first frame")
    
    # Sequential spectral subtraction
    enhanced_magnitude = np.zeros_like(magnitude)
    
    for t in range(num_frames):
        # Update noise estimate only during non-speech frames
        if not is_speech[t]:
            noise_mag = alpha * noise_mag + (1 - alpha) * magnitude[t]
        
        # Spectral subtraction with flooring
        # mag_enh = max(mag_t - beta * noise_mag, floor * noise_mag)
        subtracted = magnitude[t] - beta * noise_mag
        floored = floor * noise_mag
        enhanced_magnitude[t] = np.maximum(subtracted, floored)
    
    print(f"-> Spectral subtraction applied to {num_frames} frames")
    
    # Reconstruct using original phase
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    
    # Inverse FFT
    enhanced_frames = np.fft.irfft(enhanced_stft, n=n_fft, axis=1)
    
    # Overlap-add reconstruction
    output_length = (num_frames - 1) * hop_length + n_fft
    enhanced_audio = np.zeros(output_length, dtype=np.float32)
    window_sum = np.zeros(output_length, dtype=np.float32)
    
    for t in range(num_frames):
        start = t * hop_length
        end = start + n_fft
        enhanced_audio[start:end] += enhanced_frames[t] * window
        window_sum[start:end] += window ** 2
    
    # Normalize by window sum (avoid division by zero)
    window_sum = np.maximum(window_sum, 1e-8)
    enhanced_audio = enhanced_audio / window_sum
    
    # Trim to original length
    enhanced_audio = enhanced_audio[:len(noisy_audio)]
    
    print(f"-> Overlap-add reconstruction complete: {len(enhanced_audio)} samples")
    
    return enhanced_audio.astype(np.float32)

