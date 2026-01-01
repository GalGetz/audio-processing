import numpy as np


def spectral_subtraction(
    noisy_audio: np.ndarray,
    is_speech: np.ndarray,
    sr: int,
    win_ms: float = 20.0,
    hop_ms: float = 10.0,
    beta: float = 1.0,
    floor: float = 0.05,
    noise_buffer_frames: int = 50,
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
    beta : float
        Subtraction factor (default: 1.0).
    floor : float
        Spectral floor to prevent negative values (default: 0.05).
    noise_buffer_frames : int
        Noise footprint buffer size in frames (default: 50). Only non-speech frames
        are pushed into this buffer; the noise estimate is the average of the buffer.
        
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

    # -------------------------------------------------------------------------
    # IMPORTANT: pad + "centered" framing to avoid boundary artifacts.
    #
    # Without padding, the first/last few samples are reconstructed from just a
    # single Hann-windowed frame. Because Hann approaches 0 at the edges, the
    # overlap-add normalization divides by a very small window_sum and can
    # amplify a couple samples dramatically (the "spike at the end" seen in the
    # plots).
    #
    # By padding n_fft//2 on both sides (librosa-style centered STFT) and then
    # trimming back to the original length, the reconstructed boundary samples
    # correspond to the *middle* of analysis windows, where the window power is
    # well-behaved.
    # -------------------------------------------------------------------------
    pad = n_fft // 2
    padded_audio = np.pad(noisy_audio, (pad, pad), mode="reflect")
    
    # Vectorized framing using advanced indexing
    num_frames = 1 + (len(padded_audio) - n_fft) // hop_length
    indices = np.arange(n_fft)[None, :] + hop_length * np.arange(num_frames)[:, None]
    frames = padded_audio[indices]
    
    # Apply window to all frames (vectorized)
    windowed_frames = frames * window[None, :]
    
    # Compute STFT - vectorized rFFT for all frames at once
    stft = np.fft.rfft(windowed_frames, axis=1)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    print(f"-> STFT computed: {num_frames} frames, {magnitude.shape[1]} frequency bins")
    
    # Align the provided VAD mask to the padded framing.
    #
    # With pad=n_fft//2, the padded framing has exactly 2 extra frames (one at
    # the very start and one at the very end) compared to the unpadded framing
    # used by `compute_frame_energy`. The shift is +1 frame.
    is_speech_padded = np.zeros(num_frames, dtype=bool)
    if is_speech is not None and len(is_speech) > 0:
        start = 1
        end = min(start + len(is_speech), num_frames)
        is_speech_padded[start:end] = is_speech[: end - start]

    # Initialize noise estimate from initial non-speech frames (vectorized)
    noise_frames_idx = np.where(~is_speech_padded[:min(50, num_frames)])[0]
    if len(noise_frames_idx) > 0:
        noise_mag = np.mean(magnitude[noise_frames_idx], axis=0)
    else:
        noise_mag = np.zeros(magnitude.shape[1], dtype=magnitude.dtype)
    print(f"-> Initialized noise estimate from {len(noise_frames_idx)} noise frames")
    
    # =========================================================================
    # Sequential noise estimation (lecture-aligned):
    # - Aggregate non-speech frames into a fixed-size buffer
    # - Average the buffer to form the noise footprint
    # This is inherently sequential (buffer updates over time), but all per-frame
    # operations are vectorized (no inner loops over frequency bins).
    # =========================================================================
    noise_estimates = np.zeros_like(magnitude)

    n_buf = max(int(noise_buffer_frames), 1)
    # Ring buffer of noise magnitudes (only filled with non-speech frames)
    noise_buf = np.zeros((n_buf, magnitude.shape[1]), dtype=magnitude.dtype)
    buf_sum = np.zeros(magnitude.shape[1], dtype=magnitude.dtype)
    buf_count = 0
    buf_pos = 0

    for t in range(num_frames):
        if not is_speech_padded[t]:
            mag_t = magnitude[t]
            if buf_count < n_buf:
                noise_buf[buf_pos] = mag_t
                buf_sum += mag_t
                buf_count += 1
            else:
                # Replace oldest entry (ring buffer)
                buf_sum -= noise_buf[buf_pos]
                noise_buf[buf_pos] = mag_t
                buf_sum += mag_t

            buf_pos = (buf_pos + 1) % n_buf

            # Average buffer to estimate noise footprint
            noise_mag = buf_sum / float(buf_count)

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
    
    # Normalize by window sum (avoid boundary blow-ups)
    # With the centered-padding above, window_sum is well-behaved over the region
    # we keep after trimming. Still, guard against division by 0.
    window_sum = np.maximum(window_sum, 1e-8)
    enhanced_audio = enhanced_audio / window_sum
    
    # Trim back to original length (remove the padding)
    enhanced_audio = enhanced_audio[pad : pad + len(noisy_audio)]
    
    print(f"-> Overlap-add reconstruction complete: {len(enhanced_audio)} samples")
    
    return enhanced_audio.astype(np.float32)
