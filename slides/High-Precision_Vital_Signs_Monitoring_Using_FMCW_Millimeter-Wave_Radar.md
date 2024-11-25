---
marp: true
math: latex
size: 4:3
--- 

# High-Precision Vital Signs Monitoring Using FMCW Millimeter-Wave Radar

###### Digital Signal Processing (DSP) techniques for non-contact detection of respiration and heart rate.

---
## Why This Research?
- **Non-contact vital sign monitoring** is crucial for:
  - Burn patients, newborns, elderly, and infectious disease monitoring.
  - Applications like home healthcare and driver fatigue detection.
- **Challenges**:
  - Hardware noise → **Low SNR** (signal-to-noise ratio)
  - Extracting small motions like breathing/heartbeat accurately.

---

# Proposed System Overview

---

## FMCW Radar-Based Monitoring System
- **Radar Type**: 77 GHz FMCW millimeter-wave radar.
- **Scenario**: Monitor a person sitting in an office environment.
- **Data Acquired**: 
  - **Respiration rate**
  - **Heart rate**
- **DSP Chain**: Key innovation of the paper.

---

# Signal Processing Chain

---

## Steps in DSP Workflow

1. **Signal Preprocessing**:
   - Static clutter removal (background interference).
   - DC offset compensation (correct hardware bias).
2. **Phase Extraction In Range Window**:
   - Use extended DACM algorithm to extract phase.
3. **Noise Reduction**:
   - Iterative VMD Wavelet-Interval-Thresholding.
4. **Feature Extraction**:
   - Respiration: FFT-CZT hybrid algorithm.
   - Heartbeat: Time-domain peak-seeking & FFT.
5. **Final Output**:
   - Accurate respiration and heart rate with low relative errors.

---

![FMCW](img/FMCW_DSP/img1.png)

---

# Range FFT and Static Clutter Removal

---
# Range FFT
- Converts radar raw data (in time domain) into frequency domain.
- Identifies the **distance (range)** of targets by analyzing signal frequencies.

# Static Signal Clutter Removal
- Eliminates interference from stationary objects (e.g., walls or furniture).
- Focuses only on signals reflected from moving targets (e.g., human breathing or heartbeat).

---

# Range FFT
### Key Concepts:
- **Radar Signal:** Reflected signals contain time delays proportional to the distance of objects.
- **FFT (Fast Fourier Transform):** Converts the time-delay signal into frequencies to calculate distances.

### Steps:
1. **Raw Data:** Radar captures signals over multiple fast-time intervals.
2. **Apply FFT:** Transforms time-domain data into frequency domain.
3. **Identify Range:** The frequency peaks correspond to the distances (ranges) of objects.

---

# Static Signal Clutter Removal
### Key Concepts:
- Stationary objects (e.g., walls, furniture) cause "clutter" in radar data.
- Moving targets (e.g., human chest movements) create dynamic signals.

##### Method:
$$
y[m,n] = y_0[m,n]-\frac{1}{N_{\text{frames}}}\sum^{N_{\text{frames}}}_{n=1}y_0[m,n]
$$
where $m=1,2,3,...,N_{\text{samples}}; n=1,2,3,...,N_{\text{frames}}$.
$N_{\text{samples}}$ means the number of sampling points of each chirp; and $N_{\text{frames}}$ means the number of frames.

---

# Why Are These Steps Important?
## Range FFT:
- Determines **where** the targets are located (distance).
- Creates the **range bins** for further processing.

## Static Signal Clutter Removal:
- Focuses on **dynamic movements** of interest.
- Improves signal-to-noise ratio (SNR) for vital signs detection.

---

![](img/FMCW_DSP/img2.png)

---

# Experiments and Results

---
## Experimental Setup
- **Radar Used**: TI AWR1642 with DCA1000 acquisition board.
- **Subjects**: 11 people, 2 groups.
  - Group 1: Normal respiration & heart rates.
  - Group 2: Accelerated breathing/heart rates.
- **Test Scenarios**: Distances of 0.8m, 1m, 1.3m, 1.5m.

---

## Results
- **SNR Improvement**:
  - Respiration: +1.89 dB.
  - Heartbeat: +1.44 dB.
- **Accuracy**:
  - Respiration: Avg. error = 1.33%.
  - Heartbeat: Avg. error = 1.96%.

---

# Contributions of the Paper

---
## Why This Paper Matters
- Combines FMCW radar with advanced DSP techniques.
- Demonstrates non-contact vital sign monitoring with high accuracy.
- Addresses challenges like:
  - Low SNR.
  - Hardware imperfections.
  - Small motion extraction.
- **Applications**:
  - Healthcare monitoring.
  - Emergency alerts.
  - Home-based elderly care.

---

# Summary and Conclusion

---
## Key Takeaways
- FMCW radar + DSP = High-precision monitoring.
- **DSP Techniques**:
  - Iterative VMD Wavelet-Interval-Thresholding.
  - FFT-CZT hybrid algorithm.
  - DC offset compensation.
- **Results**:
  - Accurate respiration and heart rate monitoring.
  - Applicable in noisy, real-world environments.
- **Future Scope**:
  - Real-time processing.
  - Broader healthcare applications.

