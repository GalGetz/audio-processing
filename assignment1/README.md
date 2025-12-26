# Audio Processing - Assignment 1

Advanced Topics in Audio Processing using Deep Learning

---

## Project Structure

```
assignment1/
├── main.py                 # Main runner script
├── part1/
│   ├── __init__.py
│   ├── loading.py          # load_audio()
│   ├── resampling.py       # resample_to_32k(), downsample_to_16k()
│   └── visualization.py    # plot_audio_analysis()
├── part2/
│   ├── __init__.py
│   └── noise.py            # load_and_resample_noise(), add_noise_to_audio(), plot_noise_addition()
├── part3/
│   ├── __init__.py
│   ├── vad.py              # compute_frame_energy(), compute_vad_mask(), plot_vad_threshold()
│   └── spectral_subtraction.py  # spectral_subtraction()
├── part4/
│   ├── __init__.py
│   └── agc.py              # compute_frame_rms_db(), compute_agc_gains(), apply_agc(), soft_clip_sigmoid(), plot_gain_curve()
├── part5/
│   ├── __init__.py
│   └── phase_vocoder.py    # stft(), istft(), phase_vocoder(), time_stretch_phase_vocoder(), plot_time_stretch_comparison()
├── 10-sec-gal.wav          # Recording (10 seconds)
└── stationary_noise.wav    # Noise file
```

---

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your recording (`10-sec-gal.wav`) in the `assignment1/` directory.
   - **File Format:** Only `.wav` files are supported.
   - **Recording requirements:** 10 seconds total
     - 0-5 seconds: Speak ~20cm from the microphone
     - 5-10 seconds: Speak ~3m from the microphone

3. Run the assignment:
   ```bash
   cd assignment1
   python main.py
   ```

---

## Part 1: Audio Loading, Resampling, and Visualization

### Part 1.a: Load the Audio File

**Objective:** Load the recorded audio, handle stereo-to-mono conversion, and identify the native sampling frequency.

**Implementation:** `part1/loading.py`

1. **Loading:** Use `soundfile.read()` with `dtype="float32"` to preserve the original sampling rate.
2. **Stereo Handling:** If the audio has multiple channels, keep only the first channel (left).
3. **Trimming:** Ensure the audio is exactly 10 seconds.
4. **Output:** Print the native sampling frequency (Hz).

---

### Part 1.b: Resample to 32kHz

**Objective:** Change the sampling rate of the signal to 32kHz.

**Implementation:** `part1/resampling.py`

1. **Method:** Use `scipy.signal.resample`.
2. **Type Casting:** Ensure the audio signal is cast to `np.float32`.
3. **Calculation:** Determine the target number of samples based on the ratio of the target rate (32kHz) to the native rate.

---

### Part 1.c: Downsample to 16kHz

**Objective:** Downsample the 32kHz audio to 16kHz using two different methods and compare them.

**Implementation:** `part1/resampling.py`

1. **Method 1 (Naive Decimation):** Select every even sample (index 0, 2, 4...) from the 32kHz signal.
2. **Method 2 (Scipy Resample):** Use `scipy.signal.resample` to resample from 32kHz to 16kHz.

---

### Part 1.d: Visualization Function

**Objective:** Plot audio characteristics including waveform, spectrogram, mel-spectrogram, and energy/RMS.

**Implementation:** `part1/visualization.py`

1. **Windowing:** 20ms window size, 10ms hop size (50% overlap).
2. **Subplots:**
   - **Audio:** Time-domain waveform.
   - **Spectrogram:** Frequency-domain representation (0 to Fmax) with pitch contour overlay (Praat/Parselmouth).
   - **Mel-Spectrogram:** Perceptual frequency scale (librosa).
   - **Energy and RMS:** Temporal evolution of loudness (vectorized NumPy).

**Analysis: Missing Timeframes in Pitch Contour**

Some timeframes may be missing (gaps in the cyan line) because:
- The signal in those frames is **unvoiced** (silence or fricatives like /s/, /f/).
- The pitch detection algorithm did not find a strong enough periodic component.

---

### Part 1.e: Analysis and Comparison

**Which downsampling method is better?**

The `scipy.signal.resample` version is superior.

**Why?**
- **Aliasing:** Naive decimation (taking every 2nd sample) violates Nyquist if the signal has frequencies above 8kHz. These fold back as distortion.
- **Anti-aliasing:** `scipy.signal.resample` uses an FFT-based approach that effectively prevents aliasing.

**Lecture-aligned parameter choices:**
- **Spectrogram/STFT window:** Hamming
- **Mel filterbank size:** 80 mels (typical 40–80 @ 16kHz)

**Outputs:**
- `audio_16k_naive.wav` - Naive downsampled audio
- `audio_16k_resampled.wav` - Scipy resampled audio
- `analysis_naive_downsampling_(16khz).png` - Visualization (naive)
- `analysis_scipy_resample_(16khz).png` - Visualization (scipy)

---

## Part 2: Adding Noise

### Part 2.a: Load and Resample Noise

**Objective:** Load `stationary_noise.wav` and resample to 16kHz.

**Implementation:** `part2/noise.py`

---

### Part 2.b: Add Noise to Audio

**Objective:** Add noise to the clean audio from Part 1.c.2 using the `+` operator.

**Implementation:** `part2/noise.py`

- Handles length mismatch by truncation (shorter signal determines final length).

---

### Part 2.c: Plot Noise Addition

**Objective:** Visualize clean audio, noise, and noisy audio.

**Implementation:** `part2/noise.py`

**Outputs:**
- `part2_noise_addition.png` - 3-subplot visualization
- `audio_noisy.wav` - The noisy audio file

---

## Part 3: Spectral Subtraction

### Part 3.a: Voice Activity Detection (VAD)

**Objective:** Find speech parts using a threshold on the energy level.

**Implementation:** `part3/vad.py`

1. **Frame Energy:** Compute energy per frame using vectorized framing (20ms window, 10ms hop).
2. **Threshold:** Use percentile-based threshold (40th percentile) for robust speech/noise classification.
3. **VAD Mask:** Frames with energy above threshold are classified as speech.
4. **Plot:** Energy contour with threshold line overlay.

---

### Part 3.b: Sequential Spectral Subtraction

**Objective:** For every time-frame, estimate noise and subtract it from the signal sequentially.

**Implementation:** `part3/spectral_subtraction.py`

1. **STFT:** Compute Short-Time Fourier Transform using Hann window.
2. **Noise Estimation (lecture-aligned):** Collect non-speech frames into a **fixed-size buffer** and estimate the noise footprint as the **average of that buffer**.
3. **Spectral Subtraction:** 
   - `enhanced_mag = max(mag - beta * noise_mag, floor * noise_mag)`
4. **Reconstruction:** Overlap-add with original phase.

**Parameters:**
- `beta=3.0` - Subtraction factor (higher = more aggressive)
- `floor=0.001` - Spectral floor (lower = more suppression)
- `noise_buffer_frames=50` - Noise footprint buffer size (in frames)

---

### Part 3.c: Plot Enhanced Audio

**Objective:** Visualize enhanced audio using the Part 1 plotting function.

**Implementation:** Reuses `part1/visualization.py`

**Outputs:**
- `part3_vad_threshold.png` - Energy contour with VAD threshold
- `audio_enhanced.wav` - Enhanced audio after spectral subtraction
- `analysis_enhanced_(spectral_subtraction).png` - Full visualization of enhanced audio

---

## Part 4: Auto Gain Control (AGC)

### Part 4.a.i: Determine Desired RMS

**Objective:** Set the target RMS level in dB for AGC normalization.

**Implementation:** `part4/agc.py`

- Compute per-frame RMS in dB
- Identify speech frames (frames above noise floor)
- Set `desired_rms_db` as **75th percentile** of RMS over speech frames

---

### Part 4.a.ii: Determine Noise Floor Threshold

**Objective:** Set the noise floor to avoid amplifying noise.

**Implementation:** `part4/agc.py`

- Set `noise_floor_db` as **20th percentile** of RMS (global)
- Frames below noise floor will have gain limited to <= 0 dB (no amplification)

---

### Part 4.a.iii: Sequential Gain Computation

**Objective:** For every time-frame, compute gain using ~1s statistics window.

**Implementation:** `part4/agc.py`

1. **Running Statistics:** Maintain a ring buffer of ~100 frames (~1s at 10ms hop)
2. **Target Gain:** `target_gain_db = desired_rms_db - running_rms_stat_db`
3. **Noise Floor Gating:** If frame RMS < noise floor, limit gain to <= 0 dB
4. **Attack/Release Smoothing:** Sequential exponential smoothing
   - Fast attack (gain decreasing): `coef=0.1`
   - Slow release (gain increasing): `coef=0.01`

---

### Part 4.a.iv: Overflow Prevention

**Objective:** Avoid clipping after gain application.

**Implementation:** `part4/agc.py`

- Apply soft clipping using `tanh`: `y = tanh(drive * x) / tanh(drive)`
- Ensures output stays in (-1, 1) range without hard clipping artifacts

---

### Part 4.a.v: Plot AGC Output

**Objective:** Visualize AGC-processed audio using the Part 1 plotting function.

**Implementation:** Reuses `part1/visualization.py`

---

### Part 4.a.vi: Plot Scaling Factors

**Objective:** Plot the AGC gain curve over time.

**Implementation:** `part4/agc.py`

**Outputs:**
- `audio_agc.wav` - AGC-processed audio
- `part4_agc_gains.png` - Gain (dB) vs time plot
- `analysis_agc_output_(16khz).png` - Full visualization of AGC output

---

## Part 5: Time-Stretching with Phase Vocoder

### Part 5.a: Overview

**Objective:** Increase the speed of the audio from Q1.c.2 by a factor of x1.5 while **preserving pitch** using a **phase vocoder** algorithm.

**Implementation:** `part5/phase_vocoder.py`

---

### Part 5.a.i: Input/Output Mapping

**Objective:** Set the mapping between input and output times.

- Use **fixed monotonic mapping** for rate=1.5 (speed up)
- Output frame `t_out` maps to input time `t_in = t_out * rate`
- Find adjacent frames `t0 = floor(t_in)`, `t1 = min(t0+1, T-1)`
- Interpolation weight `w = t_in - t0`

---

### Part 5.a.ii: Apply STFT

**Objective:** Compute Short-Time Fourier Transform on the input audio.

- Window: **Hamming** (lecture default)
- Window size: 20ms
- Hop size: 10ms
- Vectorized framing using advanced indexing

---

### Part 5.a.iii: Magnitude and Phase Computation

**Objective:** Calculate magnitude and phase values for the output STFT.

**Lecture-aligned algorithm:**

1. **Magnitude Interpolation:**
   - `mag_out = (1-w) * |X[t0]| + w * |X[t1]|`

2. **Phase Update (accumulation):**
   - Compute phase difference: `phase_diff = angle(X[t1]) - angle(X[t0])`
   - Wrap to [-pi, pi]
   - Accumulate: `phase_out = phase_prev + expected_advance + phase_diff`

3. **Recompose:**
   - `Y[t_out] = mag_out * exp(1j * phase_out)`

---

### Part 5.a.iv: Apply iSTFT

**Objective:** Reconstruct time-stretched audio from the modified STFT.

- Inverse rFFT per frame
- Overlap-add reconstruction (vectorized using `np.add.at`)
- Output `np.float32`

---

### Part 5.a.v: Plot Signals

**Objective:** Plot original vs stretched audio in time and spectral domains.

**2x2 subplot figure:**
- Top-left: Original waveform
- Top-right: Stretched waveform
- Bottom-left: Original spectrogram
- Bottom-right: Stretched spectrogram

**Outputs:**
- `audio_speedx1p5.wav` - Time-stretched audio (1.5x speed, pitch preserved)
- `part5_time_stretch.png` - Time and spectral domain comparison
