import numpy as np
from scipy import signal


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

