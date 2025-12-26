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

