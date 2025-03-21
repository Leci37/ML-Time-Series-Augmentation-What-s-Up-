import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def main():
    # Simulación de datos
    np.random.seed(42)
    timesteps = 478
    features = 13
    t = np.linspace(0, 4 * np.pi, timesteps)
    base_wave = np.sin(t)
    noise = np.random.normal(scale=0.2, size=(timesteps, features))
    time_series_data = (base_wave[:, None] + noise) * 0.5 + 0.5
    time_series_data = time_series_data[:int(timesteps * 0.7)]

    feature_index = 0
    original_feature = time_series_data[:, feature_index].astype(float)
    time_axis = np.arange(len(original_feature)).astype(int)

    # Asegurar que 'value' es un Series con nombre
    value_series = pd.Series(original_feature, name="value")

    # Preparar datos para TSFRESH
    df_shift, y = make_forecasting_frame(value_series, kind="simulated", max_timeshift=10, rolling_direction=1)

    # Extraer y filtrar features
    X = extract_features(df_shift, column_id="id", column_sort="time", disable_progressbar=True)
    X_filtered = select_features(X, y)

    # Dividir y entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse:.4f}")

    # Gráfico serie original
    plt.figure(figsize=(14, 6))
    plt.plot(time_axis, original_feature, label='Simulated Feature')
    plt.title('Simulated Time Series Feature')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Gráfico predicción vs real
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(f"LightGBM: Predicción vs Realidad (RMSE: {rmse:.4f})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
