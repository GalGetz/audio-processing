#!/usr/bin/env python3
"""
Audio Processing - Assignment 1
Advanced Topics in Audio Processing using Deep Learning

Main runner script that executes all parts of the assignment.
"""

import os
import soundfile as sf

from part1 import (
    load_audio,
    resample_to_32k,
    downsample_to_16k,
    plot_audio_analysis,
)

from part2 import (
    load_and_resample_noise,
    add_noise_to_audio,
    plot_noise_addition,
)


def main():
    """Main execution function."""
    
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_FILE = "10-sec-gal.wav"
    NOISE_FILE = "stationary_noise.wav"
    AUDIO_PATH = os.path.join(SCRIPT_DIR, AUDIO_FILE)
    NOISE_PATH = os.path.join(SCRIPT_DIR, NOISE_FILE)
    
    print("=" * 60)
    print("ASSIGNMENT 1: Audio Processing")
    print("=" * 60)
    
    # =========================================================================
    # PART 1: Audio Loading, Resampling, and Visualization
    # =========================================================================
    
    # Part 1.a: Load the audio
    audio, fs = load_audio(AUDIO_PATH)
    
    # Part 1.b: Resample to 32kHz
    audio_32k, fs_32k = resample_to_32k(audio, fs)
    
    # Part 1.c: Downsample to 16kHz
    audio_16k_naive, audio_16k_resampled, fs_16k = downsample_to_16k(audio_32k)
    
    # Part 1.d: Visualization
    plot_audio_analysis(audio_16k_naive, fs_16k, "Naive Downsampling (16kHz)")
    plot_audio_analysis(audio_16k_resampled, fs_16k, "Scipy Resample (16kHz)")
    
    # Part 1.e: Save audio files for listening comparison
    sf.write(os.path.join(SCRIPT_DIR, "audio_16k_naive.wav"), audio_16k_naive, fs_16k)
    sf.write(os.path.join(SCRIPT_DIR, "audio_16k_resampled.wav"), audio_16k_resampled, fs_16k)
    print("\n-> Saved 16kHz audio files for comparison.")
    
    print("\n--- Part 1 Summary ---")
    print(f"Original Audio: {fs} Hz, {len(audio)} samples")
    print(f"Resampled Audio (1.b): {fs_32k} Hz, {len(audio_32k)} samples")
    print(f"Naive Downsampled (1.c.i.1): {fs_16k} Hz, {len(audio_16k_naive)} samples")
    print(f"Scipy Resampled (1.c.i.2): {fs_16k} Hz, {len(audio_16k_resampled)} samples")
    print(f"Duration: {len(audio_16k_resampled) / fs_16k:.2f} seconds")
    
    # =========================================================================
    # PART 2: Adding Noise
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 2: Adding Noise")
    print("=" * 60)
    
    # Part 2.a: Load and resample noise to 16kHz
    noise_16k, noise_native_sr = load_and_resample_noise(NOISE_PATH, fs_16k)
    
    # Part 2.b: Add noise to audio from Q1.c.2 (scipy resampled)
    audio_clean, noise_truncated, noisy_audio = add_noise_to_audio(audio_16k_resampled, noise_16k)
    
    # Part 2.c: Plot the signals
    plot_noise_addition(audio_clean, noise_truncated, noisy_audio, fs_16k)
    
    # Save the noisy audio
    sf.write(os.path.join(SCRIPT_DIR, "audio_noisy.wav"), noisy_audio, fs_16k)
    print("\n-> Saved noisy audio to: audio_noisy.wav")
    
    print("\n" + "=" * 60)
    print("Assignment 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

