import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Augmentation Methods
def jittering(data, sigma=0.02):
    """Adds Gaussian noise to the time-series data"""
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise

def scaling(data, sigma=0.1):
    """Scales the time-series data by a random factor"""
    scale = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[0], 1))
    return data * scale

def time_warping(data, alpha=1.5):
    """Randomly stretches or compresses the time series"""
    factor = np.random.uniform(low=1/alpha, high=alpha)
    indices = np.round(np.linspace(0, len(data) - 1, int(len(data) * factor))).astype(int)
    indices = np.clip(indices, 0, len(data) - 1)
    return data[indices]

def window_slicing(data, slice_ratio=0.9):
    """Randomly slices a portion of the time series"""
    slice_size = int(len(data) * slice_ratio)
    start = np.random.randint(0, len(data) - slice_size)
    return data[start:start + slice_size]

def time_masking(data, mask_ratio=0.1):
    """Randomly masks values in the time series"""
    mask = np.random.choice([0, 1], size=data.shape, p=[mask_ratio, 1 - mask_ratio])
    return data * mask

def rolling_shift(data, shift_max=10):
    """Randomly shifts the time-series data circularly"""
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift, axis=0)

def amplitude_scaling(data, factor_range=(0.8, 1.2)):
    """Scales the amplitude of the signal randomly"""
    factor = np.random.uniform(*factor_range)
    return data * factor

def sine_wave_perturbation(data, freq=0.1, magnitude=0.05):
    """Adds sinusoidal variation to the signal"""
    t = np.linspace(0, 2*np.pi, data.shape[0])
    sine_wave = np.sin(freq * t)[:, None] * magnitude
    return data + sine_wave

def time_warping_interpolation(data, num_knots=4, sigma=0.2):
    """Non-linear time warping using interpolation"""
    orig_time = np.arange(data.shape[0])
    knots = np.linspace(0, data.shape[0]-1, num_knots)
    warp_factors = np.random.normal(loc=1.0, scale=sigma, size=num_knots)
    warped_knots = np.clip(knots * warp_factors, 0, data.shape[0]-1)
    interpolator = interp1d(warped_knots, data[np.round(knots).astype(int)], axis=0, kind='linear', fill_value="extrapolate")
    return interpolator(orig_time)

def random_erasing(data, erase_ratio=0.1):
    """Randomly erases parts of the time series with noise"""
    erase_indices = np.random.choice([0, 1], size=data.shape, p=[erase_ratio, 1 - erase_ratio])
    noise = np.random.normal(loc=np.mean(data), scale=np.std(data), size=data.shape)
    return np.where(erase_indices == 0, noise, data)

def spike_injection(data, num_spikes=5, magnitude=0.2):
    """Injects random spikes into the time-series"""
    spike_indices = np.random.choice(data.shape[0], num_spikes, replace=False)
    data[spike_indices] += np.random.uniform(-magnitude, magnitude, size=(num_spikes, data.shape[1]))
    return data

def cumulative_sum(data):
    """Transforms data into cumulative sum form"""
    return np.cumsum(data, axis=0)

def differencing(data):
    """Applies first-order differencing"""
    return np.diff(data, axis=0, prepend=data[0:1])

def polynomial_trend_addition(data, degree=2, magnitude=0.05):
    """Adds a random polynomial trend to the time-series"""
    x = np.linspace(-1, 1, data.shape[0])  # Time index for polynomial evaluation
    coeffs = np.random.uniform(-magnitude, magnitude, size=(degree + 1, data.shape[1]))  # Generate coefficients
    trend = np.polynomial.polynomial.polyval(x[:, None], coeffs)  # Ensure trend has shape (478, 13)
    return data + trend

def reverse_augmentation(data):
    """Reverses the time-series data"""
    return data[::-1]

def time_stretching(data, stretch_factor=1.2):
    """Alters the speed of the time-series"""
    indices = np.round(np.linspace(0, len(data) - 1, int(len(data) * stretch_factor))).astype(int)
    indices = np.clip(indices, 0, len(data) - 1)
    return data[indices]

def permutation(data, segment_length=10):
    """Randomly permutes segments of the time series"""
    num_segments = len(data) // segment_length
    segment_indices = np.random.permutation(num_segments)
    permuted_data = np.concatenate([data[i * segment_length:(i + 1) * segment_length] for i in segment_indices], axis=0)
    return permuted_data

def frequency_noise(data, sigma=0.01):
    """Adds high-frequency noise to the signal"""
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + np.sin(2 * np.pi * noise)

def mean_value_addition(data, alpha=0.1):
    """Adds a portion of the mean value of the series to itself"""
    mean_val = np.mean(data, axis=0)
    return data + alpha * mean_val

def gaussian_blur(data, sigma=1):
    """Applies Gaussian blur for signal smoothing"""
    return gaussian_filter1d(data, sigma=sigma, axis=0)
