import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers.timeseries import TimeGAN

# Create dummy time series data: 100 samples, 24 time steps, 3 features
seq_len = 24
n_samples = 100
n_features = 3

# Generate random data
raw_data = np.random.rand(n_samples, seq_len, n_features)

# Normalize data to [0,1] using a global scaler
scaler = MinMaxScaler()
data_scaled = raw_data.reshape(-1, n_features)
data_scaled = scaler.fit_transform(data_scaled)
data_scaled = data_scaled.reshape(n_samples, seq_len, n_features)

# Define TimeGAN arguments
gan_args = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'noise_dim': 32,
    'layers_dim': 64,
    'seq_len': seq_len,
    'epochs': 50
}

# Initialize and train TimeGAN
synthesizer = TimeGAN(model_parameters=gan_args)
synthesizer.train(data_scaled)

# Generate synthetic data
synthetic_data = synthesizer.sample(n_samples)

print("Original shape:", raw_data.shape)
print("Synthetic shape:", synthetic_data.shape)
