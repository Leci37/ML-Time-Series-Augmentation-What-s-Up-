# ML Time Series Augmentation: What's Up? 

In time series ML models, data augmentation is a common tool. How can we do this augmentation today?

Let's explore three different augmentation strategies:

---

## 1. Custom Augmentations (DIY Style)

**What it is:**  
A set of custom functions, essentially adding random "noise" to the time series. Techniques include:

- **Time warping/stretching:** Speeds up or slows down the sequence.
- **Jitter:** Adds random noise.
- **Window shearing:** Extracts and interpolates a portion of the series.
- **Masking/inverting/shifting:** Alters the temporal order or hides portions of the signal.
- **Frequency and trend noise injection:** Modifies the spectral or trend components of the signal.

🔗 **The code:**  
[Custom Augmentations - GitHub](https://github.com/Leci37/ML-Time-Series-Augmentation-What-s-Up-/blob/main/time_series.py)

---

## 2. Tsaug Library

**Tsaug** is ideal for quick, scalable augmentation pipelines with clean, well-documented APIs. Key methods:

- **Time warp:** Non-linear temporal distortions
- **Crop:** Keep only part of the signal
- **Drift:** Gradual shift over time
- **Quantize:** Reduce signal precision
- **Reverse & pool:** Flip sequences or downsample

🔗 **The code:**  
[Tsaug Example - GitHub](https://github.com/Leci37/ML-Time-Series-Augmentation-What-s-Up-/blob/main/time_series_tsaug.py)

---

There you have the visual results of each method — use the one that comes closest to what you want!
