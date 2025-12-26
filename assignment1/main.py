#!/usr/bin/env python3
"""
Audio Processing - Assignment 1
Advanced Topics in Audio Processing using Deep Learning

Main runner script that executes all parts of the assignment.
"""

import os
import soundfile as sf
import numpy as np

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

from part3 import (
    compute_frame_energy,
    compute_vad_mask,
    plot_vad_threshold,
    spectral_subtraction,
)

from part4 import (
    compute_frame_rms_db,
    estimate_agc_params,
    compute_agc_gains,
    apply_agc,
    soft_clip_sigmoid,
    plot_gain_curve,
)

from part5 import (
    time_stretch_phase_vocoder,
    plot_time_stretch_comparison,
)


def main():
    """Main execution function."""
    
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_FILE = "10-sec-gal.wav"
    NOISE_FILE = "stationary_noise.wav"
    AUDIO_PATH = os.path.join(SCRIPT_DIR, AUDIO_FILE)
    NOISE_PATH = os.path.join(SCRIPT_DIR, NOISE_FILE)
    
    # Output Directories
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    PART1_DIR = os.path.join(OUTPUT_DIR, "part1")
    PART2_DIR = os.path.join(OUTPUT_DIR, "part2")
    PART3_DIR = os.path.join(OUTPUT_DIR, "part3")
    PART4_DIR = os.path.join(OUTPUT_DIR, "part4")
    PART5_DIR = os.path.join(OUTPUT_DIR, "part5")
    
    for d in [PART1_DIR, PART2_DIR, PART3_DIR, PART4_DIR, PART5_DIR]:
        os.makedirs(d, exist_ok=True)
    
    print("=" * 60)
    print("ASSIGNMENT 1: Audio Processing")
    print("=" * 60)
    
    # =========================================================================
    # PART 1: Audio Loading, Resampling, and Visualization
    # =========================================================================
    
    print("\n" + "-" * 60)
    print("PART 1: Audio Loading, Resampling, and Visualization")
    print("-" * 60)
    
    # Part 1.a: Load the audio
    audio, fs = load_audio(AUDIO_PATH)
    
    # Part 1.b: Resample to 32kHz
    audio_32k, fs_32k = resample_to_32k(audio, fs)
    
    # Part 1.c: Downsample to 16kHz
    audio_16k_naive, audio_16k_resampled, fs_16k = downsample_to_16k(audio_32k)
    
    # Part 1.d: Visualization
    plot_audio_analysis(
        audio_16k_naive, 
        fs_16k, 
        "Naive Downsampling (16kHz)",
        output_path=os.path.join(PART1_DIR, "analysis_naive_downsampling.png")
    )
    plot_audio_analysis(
        audio_16k_resampled, 
        fs_16k, 
        "Scipy Resample (16kHz)",
        output_path=os.path.join(PART1_DIR, "analysis_scipy_resample.png")
    )
    
    # Part 1.e: Save audio files
    sf.write(os.path.join(PART1_DIR, "audio_16k_naive.wav"), audio_16k_naive, fs_16k)
    sf.write(os.path.join(PART1_DIR, "audio_16k_resampled.wav"), audio_16k_resampled, fs_16k)
    print("\n-> Saved 16kHz audio files to outputs/part1/")
    
    print(f"Original Audio: {fs} Hz, {len(audio)} samples")
    print(f"Resampled Audio (1.b): {fs_32k} Hz, {len(audio_32k)} samples")
    
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
    plot_noise_addition(
        audio_clean, 
        noise_truncated, 
        noisy_audio, 
        fs_16k,
        output_path=os.path.join(PART2_DIR, "part2_noise_addition.png")
    )
    
    # Save the noisy audio
    sf.write(os.path.join(PART2_DIR, "audio_noisy.wav"), noisy_audio, fs_16k)
    print("\n-> Saved noisy audio to: outputs/part2/audio_noisy.wav")
    
    # =========================================================================
    # PART 3: Spectral Subtraction
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 3: Spectral Subtraction")
    print("=" * 60)
    
    # Part 3.a: Voice Activity Detection using energy threshold
    energy, n_fft, hop_length = compute_frame_energy(noisy_audio, fs_16k)
    is_speech, threshold = compute_vad_mask(energy, threshold_percentile=40.0)
    
    print(f"-> VAD: {is_speech.sum()} speech frames, {(~is_speech).sum()} noise frames")
    
    # Plot VAD threshold over energy contour
    plot_vad_threshold(
        energy, 
        threshold, 
        fs_16k, 
        hop_length, 
        is_speech,
        output_path=os.path.join(PART3_DIR, "part3_vad_threshold.png")
    )
    
    # Part 3.b: Sequential spectral subtraction
    enhanced_audio = spectral_subtraction(
        noisy_audio, is_speech, fs_16k,
        beta=3.0,              # aggressive subtraction (3x noise magnitude)
        floor=0.001,           # very low floor for maximum suppression
        noise_buffer_frames=50 # lecture-style noise footprint buffer size
    )
    
    # Save enhanced audio
    sf.write(os.path.join(PART3_DIR, "audio_enhanced.wav"), enhanced_audio, fs_16k)
    print("\n-> Saved enhanced audio to: outputs/part3/audio_enhanced.wav")
    
    # Part 3.c: Plot enhanced audio
    plot_audio_analysis(
        enhanced_audio, 
        fs_16k, 
        "Enhanced (Spectral Subtraction)",
        output_path=os.path.join(PART3_DIR, "analysis_enhanced.png")
    )
    
    # =========================================================================
    # PART 4: Auto Gain Control (AGC)
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 4: Auto Gain Control (AGC)")
    print("=" * 60)
    
    # Part 4.a.i & 4.a.ii: Compute RMS and determine desired RMS & noise floor
    print("\n--- Part 4.a.i & 4.a.ii: Computing RMS and AGC Parameters ---")
    rms, rms_db, n_fft_agc, hop_length_agc = compute_frame_rms_db(audio_16k_resampled, fs_16k)
    
    # Estimate AGC parameters: desired RMS (75th percentile of speech) and noise floor (20th percentile)
    desired_rms_db, noise_floor_db = estimate_agc_params(rms_db)
    
    print(f"-> Desired RMS (target): {desired_rms_db:.2f} dB")
    print(f"-> Noise floor threshold: {noise_floor_db:.2f} dB")
    
    # Part 4.a.iii: Compute sequential AGC gains
    gains_db = compute_agc_gains(
        rms_db, 
        desired_rms_db, 
        noise_floor_db,
        stats_window_frames=100,  # ~1s at 10ms hop
        attack_coef=0.1,          # fast attack
        release_coef=0.01         # slow release
    )
    
    # Apply gains to audio
    agc_audio = apply_agc(audio_16k_resampled, gains_db, n_fft_agc, hop_length_agc)
    
    # Part 4.a.iv: Soft clip to avoid overflow
    agc_audio = soft_clip_sigmoid(agc_audio, drive=1.0)
    
    # Save AGC audio
    sf.write(os.path.join(PART4_DIR, "audio_agc.wav"), agc_audio, fs_16k)
    print("\n-> Saved AGC audio to: outputs/part4/audio_agc.wav")
    
    # Part 4.a.v: Plot AGC output
    plot_audio_analysis(
        agc_audio, 
        fs_16k, 
        "AGC Output (16kHz)",
        output_path=os.path.join(PART4_DIR, "analysis_agc_output.png")
    )
    
    # Part 4.a.vi: Plot scaling factors vs time
    plot_gain_curve(
        gains_db, 
        fs_16k, 
        hop_length_agc,
        output_path=os.path.join(PART4_DIR, "part4_agc_gains.png")
    )
    
    # =========================================================================
    # PART 5: Time-Stretching with Phase Vocoder
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 5: Time-Stretching (x1.5) with Phase Vocoder")
    print("=" * 60)
    
    # 5.a: Time-stretch using phase vocoder (1.5x speed, pitch preserved)
    TIME_STRETCH_RATE = 1.5
    stretched_audio = time_stretch_phase_vocoder(
        audio_16k_resampled, fs_16k, rate=TIME_STRETCH_RATE
    )
    
    # Save stretched audio
    sf.write(os.path.join(PART5_DIR, "audio_speedx1p5.wav"), stretched_audio, fs_16k)
    print(f"\n-> Saved time-stretched audio to: outputs/part5/audio_speedx1p5.wav")
    
    # 5.a.v: Plot time and spectral domain comparison
    plot_time_stretch_comparison(
        audio_16k_resampled, 
        stretched_audio, 
        fs_16k, 
        TIME_STRETCH_RATE,
        output_path=os.path.join(PART5_DIR, "part5_time_stretch.png")
    )
    
    print("\n" + "=" * 60)
    print("Assignment 1 Complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
