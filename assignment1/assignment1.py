import os
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import parselmouth
import librosa
import librosa.display

matplotlib.use('Agg')

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


# =============================================================================
# Part 1.b: Resample to 32kHz
# =============================================================================

def resample_to_32k(audio: np.ndarray, original_sr: int) -> tuple[np.ndarray, int]:
    """
    Resample the audio to 32kHz using scipy.signal.resample.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    original_sr : int
        Original sampling rate in Hz.
        
    Returns
    -------
    resampled_audio : np.ndarray
        Audio signal resampled to 32kHz.
    target_sr : int
        The target sampling rate (32000).
    """
    print("\n--- Part 1.b: Resampling to 32kHz ---")
    
    target_sr = 32000
    audio_f32 = audio.astype(np.float32)
    
    # Calculate target number of samples
    num_samples_target = int(len(audio_f32) * target_sr / original_sr)
    
    # Resample using scipy.signal.resample
    resampled_audio = signal.resample(audio_f32, num_samples_target)
    
    print(f"-> Resampled from {original_sr} Hz to {target_sr} Hz.")
    print(f"-> New number of samples: {len(resampled_audio)}")
    
    return resampled_audio, target_sr


# =============================================================================
# Part 1.c: Downsample to 16kHz
# =============================================================================

def downsample_to_16k(audio_32k: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Downsample the 32kHz audio to 16kHz using two different methods.
    
    Parameters
    ----------
    audio_32k : np.ndarray
        Input audio signal at 32kHz.
        
    Returns
    -------
    audio_16k_naive : np.ndarray
        Audio downsampled by taking every even sample.
    audio_16k_resampled : np.ndarray
        Audio downsampled using scipy.signal.resample.
    target_sr : int
        The target sampling rate (16000).
    """
    print("\n--- Part 1.c: Downsampling to 16kHz ---")
    
    target_sr = 16000
    
    # Method 1: Take every even sample (naive decimation)
    # This takes samples at indices 0, 2, 4, ...
    audio_16k_naive = audio_32k[::2].astype(np.float32)
    
    # Method 2: Resample using scipy.signal.resample
    # Calculate target number of samples (should be exactly half for 32k -> 16k)
    num_samples_target = len(audio_32k) // 2
    audio_16k_resampled = signal.resample(audio_32k, num_samples_target).astype(np.float32)
    
    print(f"-> Method 1 (Naive): {len(audio_16k_naive)} samples.")
    print(f"-> Method 2 (Resample): {len(audio_16k_resampled)} samples.")
    
    return audio_16k_naive, audio_16k_resampled, target_sr


# =============================================================================
# Part 1.d: Visualization Function
# =============================================================================

def plot_audio_analysis(audio: np.ndarray, sr: int, title: str):
    """
    Plots a figure containing 4 subplots:
    i. Audio
    ii. Spectrogram with Pitch contour
    iii. Mel-Spectrogram
    iv. Energy and RMS
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    sr : int
        Sampling frequency in Hz.
    title : str
        Title for the figure.
    """
    print(f"\n--- Part 1.d: Generating plots for '{title}' ---")
    
    # Parameters for analysis
    win_ms = 20
    hop_ms = 10
    n_fft = int(win_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    
    # Time axis for audio plot
    time_audio = np.linspace(0, len(audio) / sr, num=len(audio))
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # i. Audio Plot
    axes[0].plot(time_audio, audio)
    axes[0].set_title("Audio Signal (Time Domain)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)
    
    # ii. Spectrogram + Pitch Contour
    # Calculate spectrogram using scipy
    frequencies, times, spec = signal.spectrogram(audio, fs=sr, 
                                                 nperseg=n_fft, 
                                                 noverlap=n_fft - hop_length)
    
    # Use log scale for better visualization and avoid log(0)
    spec_db = 10 * np.log10(spec + 1e-10)
    
    im = axes[1].pcolormesh(times, frequencies, spec_db, shading='gouraud', cmap='magma')
    axes[1].set_title(f"Spectrogram (Fmax = {sr/2} Hz)")
    axes[1].set_ylabel("Frequency [Hz]")
    
    # Pitch Contour using Parselmouth (Praat)
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    # Pitch analysis: time_step, floor (Hz), ceiling (Hz)
    pitch = snd.to_pitch(time_step=hop_ms/1000, pitch_floor=75, pitch_ceiling=600)
    pitch_values = pitch.selected_array['frequency']
    # Replace zeros with NaN so they don't plot
    pitch_values[pitch_values == 0] = np.nan
    pitch_times = pitch.xs()
    
    # Plot pitch on a secondary y-axis
    ax_pitch = axes[1].twinx()
    ax_pitch.plot(pitch_times, pitch_values, color='cyan', linewidth=2, label='Pitch Contour')
    ax_pitch.set_ylabel("Pitch [Hz]", color='cyan')
    ax_pitch.tick_params(axis='y', labelcolor='cyan')
    ax_pitch.set_ylim(0, 600)
    
    # iii. Mel-Spectrogram (using librosa)
    n_mels = 128
    
    # Calculate Mel Spectrogram using librosa
    S_mel = librosa.feature.melspectrogram(y=audio, sr=sr, 
                                           n_fft=n_fft, 
                                           hop_length=hop_length, 
                                           n_mels=n_mels)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    
    # Plot using librosa's display helper
    librosa.display.specshow(S_mel_db, sr=sr, hop_length=hop_length, 
                             x_axis='time', y_axis='mel', ax=axes[2], cmap='viridis')
    axes[2].set_title(f"Mel-Spectrogram (librosa, {n_mels} mels)")
    axes[2].set_ylabel("Frequency [Hz] (Mel Scale)")
    
    # iv. Energy and RMS (Vectorized - no for loops!)
    # Create frames using vectorized indexing
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    indices = np.arange(n_fft)[None, :] + hop_length * np.arange(num_frames)[:, None]
    frames = audio[indices]
    
    # Compute RMS and Energy for all frames at once
    rms = np.sqrt(np.mean(frames**2, axis=1))
    energy = np.sum(frames**2, axis=1)
    
    # Time axis for RMS/Energy
    time_rms = np.linspace(0, len(audio)/sr, num=len(rms))
    
    axes[3].plot(time_rms, energy, label='Energy', color='orange')
    axes[3].plot(time_rms, rms, label='RMS', color='green')
    axes[3].set_title("Energy and RMS")
    axes[3].set_ylabel("Magnitude")
    axes[3].set_xlabel("Time [sec]")
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_filename = f"analysis_{title.replace(' ', '_').lower()}.png"
    plt.savefig(output_filename)
    print(f"-> Saved plot to: {output_filename}")
    plt.close()


# =============================================================================
# Part 2.a: Load and Resample Noise
# =============================================================================

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


# =============================================================================
# Part 2.b: Add Noise to Audio
# =============================================================================

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


# =============================================================================
# Part 2.c: Plot Noise Addition
# =============================================================================

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


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "10-sec-gal.wav"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_PATH = os.path.join(SCRIPT_DIR, AUDIO_FILE)
    
    # Part 1.a: Load the audio
    audio, fs = load_audio(AUDIO_PATH)
    
    # Part 1.b: Resample to 32kHz
    audio_32k, fs_32k = resample_to_32k(audio, fs)
    
    # Part 1.c: Downsample to 16kHz
    audio_16k_naive, audio_16k_resampled, fs_16k = downsample_to_16k(audio_32k)
    
    # Part 1.d: Visualization
    plot_audio_analysis(audio_16k_naive, fs_16k, "Naive Downsampling (16kHz)")
    plot_audio_analysis(audio_16k_resampled, fs_16k, "Scipy Resample (16kHz)")
    
    # Save audio files for listening (Part 1.e)
    sf.write(os.path.join(SCRIPT_DIR, "audio_16k_naive.wav"), audio_16k_naive, fs_16k)
    sf.write(os.path.join(SCRIPT_DIR, "audio_16k_resampled.wav"), audio_16k_resampled, fs_16k)
    print("\n-> Saved 16kHz audio files for comparison.")
    
    print("\n--- Summary ---")
    print(f"Original Audio: {fs} Hz, {len(audio)} samples")
    print(f"Resampled Audio (1.b): {fs_32k} Hz, {len(audio_32k)} samples")
    print(f"Naive Downsampled (1.c.i.1): {fs_16k} Hz, {len(audio_16k_naive)} samples")
    print(f"Scipy Resampled (1.c.i.2): {fs_16k} Hz, {len(audio_16k_resampled)} samples")
    print(f"Duration: {len(audio_16k_resampled) / fs_16k:.2f} seconds")
    
    # =========================================================================
    # Part 2: Adding Noise
    # =========================================================================
    NOISE_FILE = "stationary_noise.wav"
    NOISE_PATH = os.path.join(SCRIPT_DIR, NOISE_FILE)
    
    # Part 2.a: Load and resample noise to 16kHz
    noise_16k, noise_native_sr = load_and_resample_noise(NOISE_PATH, fs_16k)
    
    # Part 2.b: Add noise to audio from Q1.c.2 (scipy resampled)
    audio_clean, noise_truncated, noisy_audio = add_noise_to_audio(audio_16k_resampled, noise_16k)
    
    # Part 2.c: Plot the signals
    plot_noise_addition(audio_clean, noise_truncated, noisy_audio, fs_16k)
    
    # Save the noisy audio
    sf.write(os.path.join(SCRIPT_DIR, "audio_noisy.wav"), noisy_audio, fs_16k)
    print("\n-> Saved noisy audio to: audio_noisy.wav")

