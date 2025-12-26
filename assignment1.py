"""
Audio Processing - Assignment 1
Advanced Topics in Audio Processing using Deep Learning

This script implements the technical requirements for Assignment 1.
"""

import os
import numpy as np
import librosa


# =============================================================================
# Part 1.a: Load the Audio File
# =============================================================================

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
    
    # Load audio with original sampling rate (sr=None)
    # mono=False to detect if it was recorded in stereo
    audio, native_sr = librosa.load(file_path, sr=None, mono=False)
    
    # i. If stereo, keep only a single channel
    if audio.ndim > 1:
        num_channels = audio.shape[0]
        print(f"-> Detected stereo audio with {num_channels} channels.")
        audio = audio[0]  # Keep the first (left) channel
        print("-> Converted to mono (kept first channel).")
    else:
        print("-> Detected mono audio.")
    
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


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "10-sec-gal.m4a"
    
    # Part 1.a: Load the audio
    audio, fs = load_audio(AUDIO_FILE)
    
    print("\n--- Summary ---")
    print(f"Audio shape: {audio.shape}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Duration: {len(audio) / fs:.2f} seconds")

