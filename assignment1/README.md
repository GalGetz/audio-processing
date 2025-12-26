# Audio Processing - Assignment 1

Advanced Topics in Audio Processing using Deep Learning

---

## Project Structure

```
assignment1/
├── outputs/                # Generated outputs organized by part
│   ├── part1/
│   ├── part2/
│   ├── part3/
│   ├── part4/
│   └── part5/
├── main.py
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
2. **Stereo Handling:** If the audio has multiple channels, keep only the first channel.
3. **Output:** Print the native sampling frequency (Hz).
   - **Answer:** The sampling frequency of the original audio is **44100 Hz**.

---

### Part 1.b: Resample to 32kHz

**Objective:** Change the sampling rate of the signal to 32kHz.

---

### Part 1.c: Downsample to 16kHz

**Objective:** Downsample the 32kHz audio to 16kHz using two different methods and compare them.

1. **Method 1 (Naive):** Select every even sample (index 0, 2, 4...).
2. **Method 2 (Scipy Resample):** Using `scipy.signal.resample`.

---

### Part 1.d: Visualization Function

**Objective:** Plot audio characteristics including waveform, spectrogram, mel-spectrogram, and energy/RMS.

**Implementation:** `part1/visualization.py`

1. **Windowing:** 20ms window size, 10ms hop size (50% overlap).
2. **Subplots:**
   - **Audio:** Time-domain waveform.
   - **Spectrogram:** Frequency-domain representation (0 to Fmax) with pitch contour overlay (Praat/Parselmouth).
   - **Mel-Spectrogram:** Perceptual frequency scale (librosa).
   - **Energy and RMS:** Temporal evolution of loudness (NumPy).

**Missing Timeframes in Pitch Contour**

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

**Parameter choices:**
- **Spectrogram/STFT window:** Hamming
- **Mel filterbank size:** 80 mels (typical 40–80 @ 16kHz)

**Outputs:**
- `outputs/part1/audio_16k_naive.wav` - Naive downsampled audio
- `outputs/part1/audio_16k_resampled.wav` - Scipy resampled audio
- `outputs/part1/analysis_naive_downsampling.png` - Visualization (naive)
- `outputs/part1/analysis_scipy_resample.png` - Visualization (scipy)

---

## Part 2: Adding Noise

### Part 2.a: Load and Resample Noise

**Objective:** Load `stationary_noise.wav` and resample to 16kHz.

---

### Part 2.b: Add Noise to Audio

**Objective:** Add noise to the clean audio from Part 1.c.2 using the `+` operator.

---

### Part 2.c: Plot Noise Addition

**Objective:** Visualize clean audio, noise, and noisy audio.


**Outputs:**
- `outputs/part2/part2_noise_addition.png` - 3-subplot visualization
- `outputs/part2/audio_noisy.wav` - The noisy audio file

---

## Part 3: Spectral Subtraction

### Part 3.a: Voice Activity Detection (VAD)

**Objective:** Find speech parts using a threshold on the energy level.

1. **Frame Energy:** Compute energy per frame using vectorized framing (20ms window, 10ms hop).
2. **Threshold:** Using percentile-based threshold (40th percentile).
3. **VAD Mask:** Frames with energy above threshold are classified as speech.
4. **Plot:** Energy contour with threshold line overlay.

---

### Part 3.b: Sequential Spectral Subtraction

**Objective:** For every time-frame, estimate noise and subtract it from the signal sequentially.

1. **STFT:** Compute Short-Time Fourier Transform using Hann window.
2. **Noise Estimation:** Collect non-speech frames into a **fixed-size buffer** and estimate the noise footprint as the **average of that buffer**.
3. **Spectral Subtraction:** 
   - `enhanced_mag = max(mag - beta * noise_mag, floor * noise_mag)`
4. **Reconstruction:** Overlap-add with original phase.

---

### Part 3.c: Plot Enhanced Audio

**Objective:** Visualize enhanced audio using the Part 1 plotting function.

**Outputs:**
- `outputs/part3/part3_vad_threshold.png` - Energy contour with VAD threshold
- `outputs/part3/audio_enhanced.wav` - Enhanced audio after spectral subtraction
- `outputs/part3/analysis_enhanced.png` - Full visualization of enhanced audio

---

## Part 4: Auto Gain Control (AGC)

### Part 4.a.i: Determine Desired RMS

**Objective:** Set the target RMS level in dB for AGC normalization.

- Compute per-frame RMS in dB
- Identify speech frames (frames above noise floor)
- Set `desired_rms_db` as **75th percentile** of RMS over speech frames

---

### Part 4.a.ii: Determine Noise Floor Threshold

- Set `noise_floor_db` as **20th percentile** of RMS (global)
- Frames below noise floor will have gain limited to <= 0 dB (no amplification)

---

### Part 4.a.iii: Sequential Gain Computation

**Objective:** For every time-frame, compute gain using ~1s statistics window.

1. **Running Statistics:** Maintain a ring buffer of ~100 frames (~1s at 10ms hop)
2. **Target Gain:** `target_gain_db = desired_rms_db - running_rms_stat_db`
3. **Noise Floor Gating:** If frame RMS < noise floor, limit gain to <= 0 dB
4. **Attack/Release Smoothing:** Sequential exponential smoothing

---

### Part 4.a.iv: Overflow Prevention

**Objective:** Avoid clipping after gain application.

- Apply soft clipping using `tanh`: `y = tanh(drive * x) / tanh(drive)`
- Ensures output stays in (-1, 1) range without hard clipping artifacts

---

### Part 4.a.v: Plot AGC Output

**Objective:** Visualize AGC-processed audio using the Part 1 plotting function.

---

### Part 4.a.vi: Plot Scaling Factors

**Objective:** Plot the AGC gain curve over time.

**Outputs:**
- `outputs/part4/audio_agc.wav` - AGC-processed audio
- `outputs/part4/part4_agc_gains.png` - Gain (dB) vs time plot
- `outputs/part4/analysis_agc_output.png` - Full visualization of AGC output

---

## Part 5: Time-Stretching with Phase Vocoder

### Part 5.a: Overview

**Objective:** Increase the speed of the audio from Q1.c.2 by a factor of x1.5 while **preserving pitch** using a **phase vocoder** algorithm.

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

- Window: **Hann** (optimal for OLA reconstruction)
- Window size: **40ms** (better frequency resolution)
- Hop size: **10ms** (75% overlap)

---

### Part 5.a.iii: Magnitude and Phase Computation

**Objective:** Calculate magnitude and phase values for the output STFT.

---

### Part 5.a.iv: Apply iSTFT

**Objective:** Reconstruct time-stretched audio from the modified STFT.

- Inverse rFFT per frame
- Overlap-add reconstruction
- Normalized by squared window sum 
- Output `np.float32`

---

### Part 5.a.v: Plot Signals

**Objective:** Plot original vs stretched audio in time and spectral domains.

**2x2 subplot figure:**
- Top-left: Original waveform
- Top-right: Stretched waveform
- Bottom-left: Original spectrogram
- Bottom-right: Stretched spectrogram

---

### Quality Improvements

Phase vocoders can produce "metallic" or "phasey" artifacts. The following improvements were applied:

| Parameter | Initial | Improved | Reason |
|-----------|---------|----------|--------|
| Window size | 20ms | **40ms** | Better frequency resolution reduces smearing |
| Overlap | 50% | **75%** | Smoother transitions between frames |
| Window type | Hamming | **Hann** | Better sidelobe suppression for OLA |
| Phase logic | Simple diff | **Instantaneous frequency** | Proper phase coherence |
| Volume | Unnormalized | **RMS Normalized** | Fixes low volume after time-stretching |

**Outputs:**
- `outputs/part5/audio_speedx1p5.wav` - Time-stretched audio
- `outputs/part5/part5_time_stretch.png` - Time and spectral domain comparison
