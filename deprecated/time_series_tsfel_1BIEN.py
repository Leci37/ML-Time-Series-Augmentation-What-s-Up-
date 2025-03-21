import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tsfel
import pandas as pd

if __name__ == "__main__":
    # Simulate data
    np.random.seed(42)
    timesteps = 478
    features = 13
    t = np.linspace(0, 4 * np.pi, timesteps)
    base_wave = np.sin(t)
    noise = np.random.normal(scale=0.2, size=(timesteps, features))
    time_series_data = (base_wave[:, None] + noise) * 0.5 + 0.5
    time_series_data = time_series_data[:int(timesteps * 0.7)]

    feature_index = 0
    signal = time_series_data[:, feature_index]

    # TSFEL combined config
    cfg = tsfel.get_features_by_domain()

    # Extract features from single signal
    features_df = tsfel.time_series_features_extractor(cfg, signal, verbose=0)
    features = features_df.iloc[0]  # 1D Series

    # Ensure plots folder
    os.makedirs("../plots", exist_ok=True)

    # --- Histogram of Feature Values ---
    plt.figure(figsize=(14, 5))
    features.plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Extracted Features")
    plt.xlabel("Feature Value")
    plt.grid(True)
    plt.savefig("plots/tsfel_histogram.png")
    plt.show()

    # --- Boxplot (single signal) ---
    plt.figure(figsize=(16, 6))
    sns.boxplot(data=features_df)
    plt.xticks(rotation=90)
    plt.title("Boxplot of TSFEL Features")
    plt.tight_layout()
    plt.savefig("plots/tsfel_boxplot.png")
    plt.show()

    # --- Extract features across multiple noisy signals ---
    num_signals = 20
    signals = []
    for _ in range(num_signals):
        noisy = (base_wave + np.random.normal(scale=0.2, size=timesteps)) * 0.5 + 0.5
        signals.append(noisy)

    feature_matrix = []
    for sig in signals:
        feats = tsfel.time_series_features_extractor(cfg, sig, verbose=0).iloc[0]
        feature_matrix.append(feats.values)

    df_all = pd.DataFrame(feature_matrix, columns=features_df.columns)

    # Save feature matrix to CSV
    os.makedirs("../data", exist_ok=True)
    csv_path = "data/tsfel_feature_matrix.csv"
    df_all.to_csv(csv_path, index=False)

    # --- Boxplot (multiple signals) ---
    plt.figure(figsize=(16, 6))
    sns.boxplot(data=df_all)
    plt.xticks(rotation=90)
    plt.title("Boxplot of TSFEL Features Across Multiple Signals")
    plt.tight_layout()
    plt.savefig("plots/tsfel_boxplot_multiple_signals.png")
    plt.show()

    # --- Correlation Heatmap ---
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_all.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/tsfel_correlation_heatmap.png")
    plt.show()

    print("Files saved to:")
    print(" - plots/tsfel_histogram.png")
    print(" - plots/tsfel_boxplot.png")
    print(" - plots/tsfel_boxplot_multiple_signals.png")
    print(" - plots/tsfel_correlation_heatmap.png")
    print(f" - {csv_path}")
