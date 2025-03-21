import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tsfel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

if __name__ == "__main__":
    # ---------------- Config ----------------
    np.random.seed(42)
    timesteps = 478
    t = np.linspace(0, 4 * np.pi, timesteps)
    base_wave = np.sin(t)

    os.makedirs("../plots", exist_ok=True)
    os.makedirs("../data", exist_ok=True)

    # ---------------- Get ALL TSFEL features ----------------
    cfg = tsfel.get_features_by_domain("all")

    # ---------------- Generate signals & labels ----------------
    feature_matrix = []
    labels = []
    raw_signals = []
    label_vectors = []

    num_signals = 100

    for _ in range(num_signals):
        noise = np.random.normal(scale=0.2, size=timesteps)
        signal = (base_wave + noise) * 0.5 + 0.5
        raw_signals.append(signal)

        label_vector = np.zeros_like(signal)  # just filler for now
        label_vectors.append(label_vector)

        # Extract features
        feats = tsfel.time_series_features_extractor(cfg, signal, verbose=0).iloc[0]
        feature_matrix.append(feats.values)

    # Create 20% positive labels randomly
    labels = np.zeros(num_signals, dtype=int)
    positive_indices = np.random.choice(num_signals, size=int(num_signals * 0.2), replace=False)
    labels[positive_indices] = 1


    # ---------------- Save raw labels and signals ----------------
    pd.DataFrame(label_vectors).to_csv("../data/label_vectors.csv", index=False)
    pd.DataFrame(raw_signals).to_csv("../data/raw_signals.csv", index=False)

    # ---------------- Create DataFrame ----------------
    example_cols = tsfel.time_series_features_extractor(cfg, base_wave, verbose=0).columns
    df_features = pd.DataFrame(feature_matrix, columns=example_cols)
    df_features["label"] = labels
    df_features.to_csv("data/tsfel_features_with_labels.csv", index=False)

    # ---------------- Plot TSFEL Feature Statistics ----------------
    # Histogram
    plt.figure(figsize=(14, 5))
    df_features.drop("label", axis=1).iloc[0].plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Extracted Features (1st sample)")
    plt.xlabel("Feature Value")
    plt.grid(True)
    plt.savefig("plots/tsfel_histogram.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(16, 6))
    sns.boxplot(data=df_features.drop("label", axis=1))
    plt.xticks(rotation=90)
    plt.title("Boxplot of TSFEL Features")
    plt.tight_layout()
    plt.savefig("plots/tsfel_boxplot.png")
    plt.close()

    # Correlation Heatmap
    corr = df_features.drop("label", axis=1).corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/tsfel_correlation_heatmap.png")
    plt.close()

    # ---------------- Train Random Forest ----------------
    X = df_features.drop(columns=["label"])
    y = df_features["label"]
    print("\n🔢 Label Distribution:")
    print(y.value_counts().rename_axis('Class').reset_index(name='Count'))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    print("\n📊 Classification Report:\n", classification_report(y, y_pred))
    print("📉 Confusion Matrix:\n", confusion_matrix(y, y_pred))

    # ---------------- SHAP Interpretation ----------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values_to_use = shap_values[1]
    else:
        shap_values_to_use = shap_values

    # SHAP bar plot
    shap.summary_plot(shap_values_to_use, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Class 1)")
    plt.tight_layout()
    plt.savefig("plots/shap_feature_importance_bar.png")
    plt.close()

    # SHAP beeswarm
    shap.summary_plot(shap_values_to_use, X, show=False)
    plt.tight_layout()
    plt.savefig("plots/shap_feature_importance_beeswarm.png")
    plt.close()

    # ---------------- Summary ----------------
    print("\n✅ Done!")
    print("📁 Data:")
    print(" - data/tsfel_features_with_labels.csv")
    print(" - data/label_vectors.csv")
    print(" - data/raw_signals.csv")
    print("📊 Plots:")
    print(" - plots/tsfel_histogram.png")
    print(" - plots/tsfel_boxplot.png")
    print(" - plots/tsfel_correlation_heatmap.png")
    print(" - plots/shap_feature_importance_bar.png")
    print(" - plots/shap_feature_importance_beeswarm.png")
