import numpy as np
import os
import matplotlib.pyplot as plt
from Augmentation_Methods import *

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

    augmentations = {
        "jittered": jittering,
        "scaled": scaling,
        "warped": time_warping,
        "sliced": window_slicing,
        "masked": time_masking,
        "shifted": rolling_shift,
        "reversed": reverse_augmentation,
        "stretched": time_stretching,
        "permuted": permutation,
        "freq_noised": frequency_noise,
        "mean_added": mean_value_addition,
        "blurred": gaussian_blur,
        "amplitude_scaled": amplitude_scaling,
        "sine_wave": sine_wave_perturbation,
        "warped_interp": time_warping_interpolation,
        "erased": random_erasing,
        "spiked": spike_injection,
        "cumsum": cumulative_sum,
        "diff": differencing,
        "poly_trend": polynomial_trend_addition,
        "reverse_window": lambda x: np.flip(x.copy(), axis=0),
        "flip_features": lambda x: x[:, ::-1],
        "add_white_noise": lambda x: x + np.random.normal(0, 0.01, size=x.shape),
        "zero_padding": lambda x: np.pad(x[:460], ((0, 18), (0, 0)), mode='constant') if x.shape[0] >= 460 else x
    }

    # Create one big plot with all augmentations
    n_cols = 3
    n_rows = int(np.ceil(len(augmentations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * (1770 / 96)),
                             constrained_layout=True)  # 70 pixels height per chart
    axes = axes.flatten()

    valid_idx = 0
    for name, func in augmentations.items():
        try:
            augmented = func(time_series_data)
            if augmented.shape[0] != len(time_series_data):
                continue  # Skip if output has different time length
            feature = augmented[:, feature_index]
            axes[valid_idx].plot(time_axis, original_feature, alpha=0.3)
            axes[valid_idx].plot(time_axis, feature, label=name.capitalize(), alpha=0.6)
            axes[valid_idx].fill_between(time_axis, original_feature, feature, color='#2ca02c', alpha=0.08,
                                         label='Deviation')
            axes[valid_idx].set_title(f"{name} (Original vs {name})")
            axes[valid_idx].set_xlabel("Time Step")
            axes[valid_idx].set_ylabel("Value")
            axes[valid_idx].legend(fontsize=8)

            axes[valid_idx].set_ylim(0, 1)  # Zoom into a region to better see differences
            valid_idx += 1
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Remove any unused subplots
    for i in range(valid_idx, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Time Series Augmentations", fontsize=16)
    plt.savefig("plots/DIY_all_augmentations_combined.png")
    plt.show()

    print("Combined plot saved to: plots/all_augmentations_combined.png")
