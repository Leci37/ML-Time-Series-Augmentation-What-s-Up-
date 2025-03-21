import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Simulated choppy sinusoidal time series data (478 timesteps, 13 features)
    np.random.seed(42)
    timesteps = 478
    features = 13
    t = np.linspace(0, 4 * np.pi, timesteps)
    base_wave = np.sin(t)
    noise = np.random.normal(scale=0.2, size=(timesteps, features))
    time_series_data = (base_wave[:, None] + noise) * 0.5 + 0.5
    time_series_data = time_series_data[:int(timesteps * 0.7)]  # 30% cut

    feature_index = 0
    time_axis = np.arange(len(time_series_data))
    original_feature = time_series_data[:, feature_index]

    # Augmentations
    def jitter(x, sigma=0.02):
        return x + np.random.normal(0, sigma, x.shape)

    def drift(x, magnitude=0.005):
        drift_values = np.cumsum(np.random.normal(0, magnitude, x.shape))
        return x + drift_values

    def dropout(x, drop_prob=0.1):
        mask = np.random.binomial(1, 1 - drop_prob, size=x.shape)
        return x * mask

    def time_mask(x, mask_size=30):
        x = x.copy()
        start = np.random.randint(0, len(x) - mask_size)
        x[start:start + mask_size] = np.mean(x)
        return x

    def window_slice(x, slice_ratio=0.6):
        size = int(len(x) * slice_ratio)
        start = np.random.randint(0, len(x) - size)
        window = x[start:start + size]
        return np.interp(np.linspace(0, size, num=len(x)), np.arange(size), window)

    # Define all augmentations
    augmentations = {
        "add_noise": lambda x: x + np.random.normal(0, 0.05, x.shape),
        "scaling": lambda x: x * 1.2,
        "smoothing": lambda x: np.convolve(x, np.ones(5) / 5, mode='same'),
        "shift_up": lambda x: x + 0.2,
        "flip": lambda x: np.flip(x, axis=0),
        "jitter": jitter,
        "drift": drift,
        "dropout": dropout,
        "time_mask": time_mask,
        "window_slice": window_slice,
        "reverse": lambda x: x[::-1],
    }

    # Create plot folder
    os.makedirs("../plots", exist_ok=True)

    # Plotting
    n_cols = 3
    n_rows = int(np.ceil(len(augmentations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4), constrained_layout=True)
    axes = axes.flatten()

    for idx, (name, func) in enumerate(augmentations.items()):
        try:
            augmented = func(original_feature)
            axes[idx].plot(time_axis, original_feature, label="Original", alpha=0.6)
            axes[idx].plot(time_axis, augmented, label=name, alpha=0.5)
            axes[idx].fill_between(time_axis, original_feature, augmented, color='orange', alpha=0.3, label='Deviation')
            axes[idx].set_title(name)
            axes[idx].set_xlabel("Time Step")
            axes[idx].set_ylabel("Value")
            axes[idx].legend(fontsize=8)
        except Exception as e:
            print(f"Error with {name}: {e}")

    # Clean up extra axes
    for i in range(len(augmentations), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Manual Time Series Augmentations vs Original", fontsize=16)
    plt.savefig("plots/manual_augmentations_expanded.png")
    plt.show()

    print("Saved to: plots/manual_augmentations_expanded.png")
