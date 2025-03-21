Great question! While tsfresh is a well-known library for feature extraction from time series, there are other powerful tools—some focus on augmentation, others on transformation, representation, or synthetic generation.

Here are strong alternatives depending on your goals:

🔁 Time Series Augmentation Libraries
These are built specifically for creating new data to improve model robustness.

Library	Highlights
numenta/nupic	Focus on anomaly detection, supports encoders and streaming data augmentation.
nni (Neural Network Intelligence)	Has a TimeSeriesAug module offering jittering, scaling, permutation, etc.
tsaug ⭐	Easy-to-use; supports random time warping, permutation, noise injection, cropping, drift, trend, etc.
GluonTS	Amazon's time series lib; great for probabilistic forecasting + synthetic generation.
TFT (Temporal Fusion Transformer)	Built on PyTorch, with attention-based understanding and support for synthetic data generation.
Datawig	For imputing missing time series (a type of augmentation via filling gaps).
🔍 Feature Extraction / Representation Alternatives to tsfresh
These convert time series into descriptive features for ML models:

Library	Highlights
Kats (by Meta)	Forecasting + anomaly detection + feature extraction.
sktime	Classical + deep learning pipelines, includes transformers and feature tools.
tsfel	Extracts features for classification — energy, entropy, FFT, etc.
catch22	Extracts 22 carefully-selected features from time series, very fast.
pyts	Focuses on time series classification (includes shapelet transform, bag-of-words, etc.)
🧪 Synthetic Time Series Generation
For creating entirely new sequences from learned distributions:

TimeGAN (TensorFlow): Combines autoencoder + GAN to generate realistic multivariate time series.
CTGAN / TVAE: Can be adapted for sequential data from tabular synthetic generators.
Darts (by Unit8): Time series forecasting and generation, with deep learning and statistical models.
✅ Recommendation
For your use case (augmentation for lathe manufacturing signals), I’d suggest:

✅ tsaug — super flexible and NumPy-friendly.
✅ tsfel or catch22 — if you want explainable, lightweight feature sets.
✅ TimeGAN — if you ever want to generate new (not just augmented) sequences for simulation or bootstrapping.
Would you like to integrate one of these (e.g., tsaug) into your current code for visualization next to your custom methods?