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
2. **Noise Estimation:** Initialize from non-speech frames, update sequentially with exponential smoothing:
   - `noise_mag = alpha * noise_mag + (1-alpha) * current_mag` (only during non-speech)
3. **Spectral Subtraction:** 
   - `enhanced_mag = max(mag - beta * noise_mag, floor * noise_mag)`
4. **Reconstruction:** Overlap-add with original phase.

**Parameters:**
- `alpha=0.98` - Noise smoothing (higher = more stable estimate)
- `beta=3.0` - Subtraction factor (higher = more aggressive)
- `floor=0.001` - Spectral floor (lower = more suppression)

---

### Part 3.c: Plot Enhanced Audio

**Objective:** Visualize enhanced audio using the Part 1 plotting function.

**Implementation:** Reuses `part1/visualization.py`

**Outputs:**
- `part3_vad_threshold.png` - Energy contour with VAD threshold
- `audio_enhanced.wav` - Enhanced audio after spectral subtraction
- `analysis_enhanced_(spectral_subtraction).png` - Full visualization of enhanced audio
