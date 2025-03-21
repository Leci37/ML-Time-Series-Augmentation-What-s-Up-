import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.preprocessing import StandardScaler

# Step 1: Simulate time series data
np.random.seed(42)
timesteps = 478
features = 13
t = np.linspace(0, 4 * np.pi, timesteps)
base_wave = np.sin(t)
noise = np.random.normal(scale=0.2, size=(timesteps, features))
time_series_data = (base_wave[:, None] + noise) * 0.5 + 0.5
time_series_data = time_series_data[:int(timesteps * 0.7)]

# Step 2: Prepare data in long format for tsfresh
df_list = []
for feature_index in range(features):
    df = pd.DataFrame({
        'id': feature_index,
        'time': np.arange(len(time_series_data)),
        'value': time_series_data[:, feature_index]
    })
    df_list.append(df)
long_format_df = pd.concat(df_list, ignore_index=True)

# Step 3: Extract features using tsfresh
extracted_features = extract_features(long_format_df, column_id="id", column_sort="time")
impute(extracted_features)

# Step 4: Normalize features for visualization
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(extracted_features),
                               index=extracted_features.index,
                               columns=extracted_features.columns)

# Step 5: Plot as heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(features_scaled, cmap="viridis", annot=False, cbar_kws={"label": "Normalized Feature Value"})
plt.title("📊 TSFresh Extracted Features per Time Series (One Row = One Feature Column)", fontsize=14)
plt.xlabel("Extracted Feature")
plt.ylabel("Original Feature Index")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("plots/tsfresh_features_heatmap.png")
plt.show()

print("✅ Feature heatmap saved to: plots/tsfresh_features_heatmap.png")
