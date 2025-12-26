"""Part 1.a: Load the Audio File."""

import os
import numpy as np
import soundfile as sf


def load_audio(file_path: str, target_duration: float = 10.0) -> tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono if necessary, and trim to target duration.
    
    Parameters
    ----------
    file_path : str
        Path to the audio file.
    target_duration : float
        Desired duration in seconds (default: 10.0).
    
    Returns
    -------
    audio : np.ndarray
        Audio signal as a 1D numpy array.
    native_sr : int
        Native sampling frequency of the audio file in Hz.
    """
    print("--- Part 1.a: Loading Audio ---")
    
    # Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Audio file not found: {file_path}\n"
            "Please ensure your recording is in the project directory."
        )
    
    if not file_path.lower().endswith('.wav'):
        raise ValueError(f"Unsupported file format: {file_path}. Only .wav files are supported.")

    # Load audio with native sampling rate using soundfile
    # always_2d=True returns shape (num_samples, num_channels)
    audio_2d, native_sr = sf.read(file_path, dtype="float32", always_2d=True)
    
    # i. If stereo, keep only a single channel
    num_channels = audio_2d.shape[1]
    if num_channels > 1:
        print(f"-> Detected multi-channel audio with {num_channels} channels.")
        audio = audio_2d[:, 0]  # Keep the first channel
        print("-> Converted to mono (kept first channel).")
    else:
        print("-> Detected mono audio.")
        audio = audio_2d[:, 0]
    
    # ii. Report the sampling frequency
    print(f"-> Native Sampling Frequency: {native_sr} Hz")
    
    # Trim to target duration (10 seconds)
    max_samples = int(target_duration * native_sr)
    original_duration = len(audio) / native_sr
    
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"-> Trimmed audio from {original_duration:.2f}s to {target_duration:.2f}s.")
    else:
        print(f"-> Audio duration: {original_duration:.2f}s (no trimming needed).")
    
    return audio, native_sr

