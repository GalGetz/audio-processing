# Audio Processing - Assignment 1

Advanced Topics in Audio Processing using Deep Learning

---

## Overview

This project implements fundamental audio processing tasks including:
- Audio loading and preprocessing
- Resampling and downsampling techniques
- Visualization (spectrograms, mel-spectrograms, pitch contours, energy/RMS)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your recording (`10-sec-gal.wav`) in the project root directory.
   - **File Format:** Only `.wav` files are supported.
   - **Recording requirements:** 10 seconds total
     - 0-5 seconds: Speak ~20cm from the microphone
     - 5-10 seconds: Speak ~3m from the microphone

3. Run the assignment:
   ```bash
   python assignment1.py
   ```

---

## Part 1.a: Load the Audio File

**Objective:** Load the recorded audio, handle stereo-to-mono conversion, and identify the native sampling frequency.

### Implementation Details

1. **Loading:** Use `soundfile.read()` with `dtype="float32"` to preserve the original sampling rate.
2. **Stereo Handling:** If the audio has multiple channels, keep only the first channel (left).
3. **Trimming:** Ensure the audio is exactly 10 seconds.
4. **Output:** Print the native sampling frequency (Hz).

### Expected Output

```
--- Part 1.a: Loading Audio ---
-> Detected mono audio.
-> Native Sampling Frequency: XXXXX Hz
```

---

## Part 1.b: Resample to 32kHz

**Objective:** Change the sampling rate of the signal to 32kHz.

### Implementation Details

1. **Method:** Use `scipy.signal.resample`.
2. **Type Casting:** Ensure the audio signal is cast to `np.float32`.
3. **Calculation:** Determine the target number of samples based on the ratio of the target rate (32kHz) to the native rate.

### Expected Output

```
--- Part 1.b: Resampling to 32kHz ---
-> Resampled from 44100 Hz to 32000 Hz.
-> New number of samples: 320000
```

---

## Part 1.c: Downsample to 16kHz

**Objective:** Downsample the 32kHz audio to 16kHz using two different methods and compare them.

### Implementation Details

1. **Method 1 (Naive Decimation):** Select every even sample (index 0, 2, 4...) from the 32kHz signal.
2. **Method 2 (Scipy Resample):** Use `scipy.signal.resample` to resample from 32kHz to 16kHz. This method typically applies an anti-aliasing low-pass filter before decimation.

### Expected Output

```
--- Part 1.c: Downsampling to 16kHz ---
-> Method 1 (Naive): 160000 samples.
-> Method 2 (Resample): 160000 samples.
```

---

## Part 1.d: Visualization Function

**Objective:** Plot audio characteristics including waveform, spectrogram, mel-spectrogram, and energy/RMS.

### Implementation Details

1. **Windowing:** Used a 20ms window size and 10ms hop size for all calculations.
2. **Subplots:**
   - **Audio:** Time-domain waveform.
   - **Spectrogram:** Frequency-domain representation (0 to $F_{max}$).
   - **Pitch Contour:** Overlaid on the spectrogram using Praat (Parselmouth).
   - **Mel-Spectrogram:** Perceptual frequency scale representation.
   - **Energy and RMS:** Temporal evolution of loudness and signal energy.

### Analysis: Missing Timeframes in Pitch Contour
In the pitch contour, some timeframes may be missing (shown as gaps in the line). This happens because:
- The signal in those frames is **unvoiced** (e.g., silence or fricatives like /s/, /f/).
- The pitch detection algorithm (autocorrelation/cross-correlation) did not find a strong enough periodic component above the defined threshold.

---

## Part 1.e: Analysis and Comparison

**Comparison of Downsampling Methods (16kHz):**

1. **Which one is better?** 
   The version resampled using `scipy.signal.resample` is superior to the naive decimation method.

2. **Why?**
   - **Aliasing:** Naive decimation (taking every second sample) violates the Nyquist-Shannon sampling theorem if the 32kHz signal contains frequencies above 8kHz. These higher frequencies "alias" or fold back into the audible lower frequency range, creating distortion and artifacts.
   - **Filter Application:** `scipy.signal.resample` uses a frequency-domain approach (FFT-based) which effectively prevents aliasing. In a standard multi-rate signal processing pipeline, this would be equivalent to applying an anti-aliasing low-pass filter (with a cutoff at 8kHz) before downsampling.

### Final Outputs
- `audio_16k_naive.wav`: Naive downsampled audio.
- `audio_16k_resampled.wav`: High-quality resampled audio.
- `analysis_naive_downsampling_(16khz).png`: Visualization of the naive version.
- `analysis_scipy_resample_(16khz).png`: Visualization of the resampled version.

