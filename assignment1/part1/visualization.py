"""Part 1.d: Visualization Function."""

import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import parselmouth
import librosa
import librosa.display

matplotlib.use('Agg')


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
    
    axes[1].pcolormesh(times, frequencies, spec_db, shading='gouraud', cmap='magma')
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

