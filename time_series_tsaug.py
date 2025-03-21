import numpy as np
import os
import matplotlib.pyplot as plt
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Pool

# Example Usage:
if __name__ == "__main__":
    # Simulated choppy sinusoidal time series data (478 timesteps, 13 features)
    np.random.seed(42)
    timesteps = 478
    features = 13
    t = np.linspace(0, 4 * np.pi, timesteps)
    base_wave = np.sin(t)
    noise = np.random.normal(scale=0.2, size=(timesteps, features))
    time_series_data = (base_wave[:, None] + noise) * 0.5 + 0.5  # Normalize to [0,1] range
    time_series_data = time_series_data[:int(timesteps * 0.7)]  # Cut the series by 30%

    feature_index = 0
    time_axis = np.arange(len(time_series_data))  # Adjust time axis to new length
    original_feature = time_series_data[:, feature_index]

    # Define transformations explicitly
    tsaug_time_warp = TimeWarp(n_speed_change=8, max_speed_ratio=5)
    tsaug_drift = Drift(max_drift=(0.5, 0.5))
    tsaug_quantize = Quantize(n_levels=4)
    tsaug_crop = Crop(size=max(2, len(time_series_data) - 20))
    tsaug_reverse = Reverse()
    tsaug_pool = Pool(size=2)
    tsaug_crop_small = Crop(size=max(2, int(len(time_series_data) * 0.5)))
    tsaug_pool_3 = Pool(size=3)
    tsaug_drift_small = Drift(max_drift=(0.03, 0.03))
    tsaug_quantize_coarse = Quantize(n_levels=5)

    # Define transformations with stronger effects
    tsaug_time_warp = TimeWarp(n_speed_change=8, max_speed_ratio=5)
    tsaug_drift = Drift(max_drift=(0.5, 0.5))  # previously 0.1
    tsaug_quantize = Quantize(n_levels=4)  # previously 20 (too subtle)

    # Then use these in the dictionary
    augmentations = {
        "tsaug_time_warp": tsaug_time_warp.augment,
        "tsaug_drift": tsaug_drift.augment,
        "tsaug_quantize": tsaug_quantize.augment,
        "tsaug_crop": tsaug_crop.augment,
        "tsaug_reverse": tsaug_reverse.augment,
        "tsaug_pool": tsaug_pool.augment,
        "tsaug_crop_small": tsaug_crop_small.augment,
        "tsaug_pool_3": tsaug_pool_3.augment,
        "tsaug_drift_small": tsaug_drift_small.augment,
        "tsaug_quantize_coarse": tsaug_quantize_coarse.augment,
        "tsaug_add_noise": lambda x: x + np.random.normal(0, 0.05, x.shape)  # keep lambda where needed
    }

    # Create one big plot with all augmentations
    n_cols = 3
    n_rows = int(np.ceil(len(augmentations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * (500 / 96)),
                             constrained_layout=True)  # 500 pixels height per chart
    axes = axes.flatten()

    valid_idx = 0
    for name, func in augmentations.items():
        try:
            augmented = func(time_series_data)
            if augmented.shape[0] != len(time_series_data):
                continue  # Skip if output has different time length
            feature = augmented[:, feature_index]
            axes[valid_idx].plot(time_axis, original_feature, label="Original", alpha=0.5)
            axes[valid_idx].plot(time_axis, feature, label=name.capitalize(), alpha=0.4)
            axes[valid_idx].fill_between(time_axis, original_feature, feature, color='#2ca02c', alpha=0.2,
                                         label='Deviation')
            axes[valid_idx].set_title(name)
            axes[valid_idx].set_xlabel("Time Step")
            axes[valid_idx].set_ylabel("Value")
            axes[valid_idx].legend(fontsize=8)
            valid_idx += 1
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Remove any unused subplots
    for i in range(valid_idx, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Time Series Augmentations", fontsize=16)
    plt.savefig("plots/tsaug_augmentations_combined.png")
    plt.show()

    print("Combined plot saved to: plots/tsaug_augmentations_combined.png")
