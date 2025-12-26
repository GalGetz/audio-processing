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

2. Place your recording (`10-sec-gal.m4a`) in the project root directory.
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

1. **Loading:** Use `librosa.load()` with `sr=None` to preserve the original sampling rate.
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

*(To be implemented)*

---

## Part 1.c: Downsample to 16kHz

*(To be implemented)*

---

## Part 1.d: Visualization Function

*(To be implemented)*

---

## Part 1.e: Analysis and Comparison

*(To be implemented)*

