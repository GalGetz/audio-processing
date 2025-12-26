"""Part 2: Noise Loading, Addition, and Visualization."""

import os
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def load_and_resample_noise(file_path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """
    Load a noise audio file and resample it to the target sampling rate.
    
    Parameters
    ----------
    file_path : str
        Path to the noise audio file.
    target_sr : int
        Target sampling rate in Hz.
        
    Returns
    -------
    noise : np.ndarray
        Resampled noise signal as np.float32.
    native_sr : int
        Original sampling rate of the noise file.
    """
    print("\n--- Part 2.a: Loading and Resampling Noise ---")
    
    # Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Noise file not found: {file_path}")
    
    # Load noise with native sampling rate
    noise_2d, native_sr = sf.read(file_path, dtype="float32", always_2d=True)
    
    # Convert to mono if stereo
    if noise_2d.shape[1] > 1:
        noise = noise_2d[:, 0]
        print(f"-> Converted noise to mono (from {noise_2d.shape[1]} channels).")
    else:
        noise = noise_2d[:, 0]
    
    print(f"-> Noise native sampling rate: {native_sr} Hz")
    print(f"-> Noise duration: {len(noise) / native_sr:.2f}s ({len(noise)} samples)")
    
    # Resample to target sampling rate
    if native_sr != target_sr:
        num_samples_target = int(len(noise) * target_sr / native_sr)
        noise_resampled = signal.resample(noise, num_samples_target).astype(np.float32)
        print(f"-> Resampled noise from {native_sr} Hz to {target_sr} Hz.")
        print(f"-> New noise length: {len(noise_resampled)} samples")
    else:
        noise_resampled = noise.astype(np.float32)
        print(f"-> Noise already at target rate ({target_sr} Hz).")
    
    return noise_resampled, native_sr


def add_noise_to_audio(audio: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add noise to audio signal, handling length mismatch by truncation.
    
    Parameters
    ----------
    audio : np.ndarray
        Clean audio signal.
    noise : np.ndarray
        Noise signal.
        
    Returns
    -------
    audio_truncated : np.ndarray
        Audio signal (possibly truncated to match noise length).
    noise_truncated : np.ndarray
        Noise signal (possibly truncated to match audio length).
    noisy_audio : np.ndarray
        Sum of audio and noise.
    """
    print("\n--- Part 2.b: Adding Noise to Audio ---")
    
    # Handle length mismatch
    audio_len = len(audio)
    noise_len = len(noise)
    
    if audio_len == noise_len:
        print(f"-> Audio and noise have same length ({audio_len} samples).")
        audio_truncated = audio
        noise_truncated = noise
    elif noise_len > audio_len:
        print(f"-> Noise ({noise_len}) longer than audio ({audio_len}). Truncating noise.")
        noise_truncated = noise[:audio_len]
        audio_truncated = audio
    else:
        print(f"-> Audio ({audio_len}) longer than noise ({noise_len}). Truncating audio.")
        audio_truncated = audio[:noise_len]
        noise_truncated = noise
    
    # Add noise using '+' operator
    noisy_audio = (audio_truncated + noise_truncated).astype(np.float32)
    
    print(f"-> Final length: {len(noisy_audio)} samples")
    
    return audio_truncated, noise_truncated, noisy_audio


def plot_noise_addition(audio: np.ndarray, noise: np.ndarray, noisy: np.ndarray, sr: int):
    """
    Plot clean audio, noise, and noisy audio signals.
    
    Parameters
    ----------
    audio : np.ndarray
        Clean audio signal.
    noise : np.ndarray
        Noise signal.
    noisy : np.ndarray
        Noisy audio signal (audio + noise).
    sr : int
        Sampling rate in Hz.
    """
    print("\n--- Part 2.c: Plotting Audio, Noise, and Noisy Signals ---")
    
    # Time axis (all signals have same length after truncation)
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Part 2: Noise Addition", fontsize=16)
    
    # Plot 1: Clean Audio
    axes[0].plot(time, audio, color='blue')
    axes[0].set_title("Clean Audio (from Q1.c.2)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)
    
    # Plot 2: Noise
    axes[1].plot(time, noise, color='red')
    axes[1].set_title("Stationary Noise (resampled to 16kHz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)
    
    # Plot 3: Noisy Audio
    axes[2].plot(time, noisy, color='purple')
    axes[2].set_title("Noisy Audio (Clean + Noise)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time [sec]")
    axes[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_filename = "part2_noise_addition.png"
    plt.savefig(output_filename)
    print(f"-> Saved plot to: {output_filename}")
    plt.close()

