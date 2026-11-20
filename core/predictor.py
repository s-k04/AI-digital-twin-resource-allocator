"""
ML Predictor for CPU and RAM usage.
Uses scikit-learn models: Random Forest, Gradient Boosting, Linear Regression.
Performs feature engineering on time-series data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime, timedelta
from collections import deque


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


AVAILABLE_MODELS = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Ridge Regression": Ridge(alpha=1.0),
    "Linear Regression": LinearRegression(),
}


class ResourcePredictor:
    """
    Predicts future CPU % and RAM % using scikit-learn models.
    Maintains a rolling data buffer for online learning.
    """

    def __init__(self, buffer_size: int = 500, horizon: int = 5):
        self.buffer_size = buffer_size
        self.horizon = horizon  # predict N steps ahead
        self.data_buffer: deque = deque(maxlen=buffer_size)
        self.cpu_model = None
        self.ram_model = None
        self.cpu_model_name = None
        self.ram_model_name = None
        self.scaler_cpu = StandardScaler()
        self.scaler_ram = StandardScaler()
        self.trained = False
        self.metrics = {}
        self.feature_names = []

    @property
    def sample_count(self) -> int:
        """Return the number of samples in the rolling buffer."""
        return len(self.data_buffer)

    def generate_synthetic_samples(self, n: int = 120, base_cpu: float = 30.0, base_ram: float = 50.0):
        """Pre-populate the buffer with realistic synthetic time-series data."""
        now = datetime.now()
        t = np.linspace(0, 4 * np.pi, n)
        cpu_series = base_cpu + 15 * np.sin(t) + 5 * np.cos(2.5 * t) + np.random.normal(0, 3, n)
        ram_series = base_ram + 8 * np.sin(0.5 * t) + np.random.normal(0, 1.5, n)
        proc_series = np.random.randint(150, 350, n)

        for i in range(n):
            dt = now - timedelta(seconds=(n - i) * 2)
            self.data_buffer.append({
                "timestamp": dt,
                "cpu": float(np.clip(cpu_series[i], 1.0, 99.0)),
                "ram": float(np.clip(ram_series[i], 5.0, 95.0)),
                "num_processes": int(proc_series[i]),
            })

    def add_sample(self, cpu_pct: float, ram_pct: float, num_processes: int = 0):
        """Add a new observation to the rolling buffer."""
        self.data_buffer.append({
            "timestamp": datetime.now(),
            "cpu": cpu_pct,
            "ram": ram_pct,
            "num_processes": num_processes,
        })

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering: lags, rolling stats, time features."""
        df = df.copy().reset_index(drop=True)
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["second"] = df["timestamp"].dt.second
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        for lag in [1, 2, 3, 5, 10]:
            df[f"cpu_lag_{lag}"] = df["cpu"].shift(lag)
            df[f"ram_lag_{lag}"] = df["ram"].shift(lag)

        for window in [5, 10, 20]:
            df[f"cpu_roll_mean_{window}"] = df["cpu"].rolling(window).mean()
            df[f"cpu_roll_std_{window}"] = df["cpu"].rolling(window).std()
            df[f"ram_roll_mean_{window}"] = df["ram"].rolling(window).mean()
            df[f"ram_roll_std_{window}"] = df["ram"].rolling(window).std()

        df["cpu_delta"] = df["cpu"].diff()
        df["ram_delta"] = df["ram"].diff()
        df["cpu_delta2"] = df["cpu_delta"].diff()

        # Target: value N steps ahead
        df[f"cpu_target"] = df["cpu"].shift(-self.horizon)
        df[f"ram_target"] = df["ram"].shift(-self.horizon)

        df.dropna(inplace=True)
        return df

    def build_dataset(self):
        """Convert buffer to feature-engineered dataset."""
        if len(self.data_buffer) < 50:
            return None, None, None, None

        df = pd.DataFrame(list(self.data_buffer))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = self._build_features(df)

        exclude = {"timestamp", "cpu_target", "ram_target"}
        self.feature_names = [c for c in df.columns if c not in exclude and c not in {"cpu", "ram"}]

        X = df[self.feature_names].values
        y_cpu = df["cpu_target"].values
        y_ram = df["ram_target"].values

        return X, y_cpu, y_ram, df

    def train(self, model_name_cpu: str = "Random Forest", model_name_ram: str = "Random Forest"):
        """Train CPU and RAM models on current buffer data."""
        X, y_cpu, y_ram, _ = self.build_dataset()
        if X is None or len(X) < 30:
            return {"error": "Not enough data. Need at least 50 samples."}

        X_train, X_test, yc_train, yc_test = train_test_split(X, y_cpu, test_size=0.2, random_state=42)
        _, _, yr_train, yr_test = train_test_split(X, y_ram, test_size=0.2, random_state=42)

        # CPU model
        cpu_estimator = AVAILABLE_MODELS[model_name_cpu].__class__(**AVAILABLE_MODELS[model_name_cpu].get_params())
        self.cpu_model = Pipeline([("scaler", StandardScaler()), ("model", cpu_estimator)])
        self.cpu_model.fit(X_train, yc_train)
        self.cpu_model_name = model_name_cpu

        # RAM model
        ram_estimator = AVAILABLE_MODELS[model_name_ram].__class__(**AVAILABLE_MODELS[model_name_ram].get_params())
        self.ram_model = Pipeline([("scaler", StandardScaler()), ("model", ram_estimator)])
        self.ram_model.fit(X_train, yr_train)
        self.ram_model_name = model_name_ram

        # Metrics
        cpu_pred = self.cpu_model.predict(X_test)
        ram_pred = self.ram_model.predict(X_test)

        self.metrics = {
            "cpu": {
                "MAE": round(mean_absolute_error(yc_test, cpu_pred), 3),
                "RMSE": round(np.sqrt(mean_squared_error(yc_test, cpu_pred)), 3),
                "R2": round(r2_score(yc_test, cpu_pred), 3),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            },
            "ram": {
                "MAE": round(mean_absolute_error(yr_test, ram_pred), 3),
                "RMSE": round(np.sqrt(mean_squared_error(yr_test, ram_pred)), 3),
                "R2": round(r2_score(yr_test, ram_pred), 3),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            },
        }

        self.trained = True
        self._save_models()
        return self.metrics

    def predict(self):
        """Return predicted CPU % and RAM % for the next horizon steps."""
        if not self.trained or self.cpu_model is None:
            return None

        _, _, _, df = self.build_dataset()
        if df is None or len(df) == 0:
            return None

        last_row = df.iloc[[-1]][self.feature_names]
        X_pred = last_row.values

        cpu_pred = float(np.clip(self.cpu_model.predict(X_pred)[0], 0, 100))
        ram_pred = float(np.clip(self.ram_model.predict(X_pred)[0], 0, 100))

        # Simple confidence estimate via prediction variance (RF only)
        cpu_ci = self._get_ci(self.cpu_model, X_pred)
        ram_ci = self._get_ci(self.ram_model, X_pred)

        return {
            "cpu_predicted": round(cpu_pred, 2),
            "ram_predicted": round(ram_pred, 2),
            "cpu_ci_low": round(max(0, cpu_pred - cpu_ci), 2),
            "cpu_ci_high": round(min(100, cpu_pred + cpu_ci), 2),
            "ram_ci_low": round(max(0, ram_pred - ram_ci), 2),
            "ram_ci_high": round(min(100, ram_pred + ram_ci), 2),
            "horizon_s": self.horizon,
        }

    def _get_ci(self, pipeline, X, z=1.96):
        """Estimate confidence interval width for a pipeline."""
        model = pipeline.named_steps.get("model")
        if hasattr(model, "estimators_"):
            X_scaled = pipeline.named_steps["scaler"].transform(X)
            preds = np.array([e.predict(X_scaled)[0] for e in model.estimators_])
            return z * np.std(preds)
        return 5.0  # default ±5% CI for linear models

    def get_prediction_series(self):
        """Return full actual vs predicted series for plotting."""
        _, y_cpu, y_ram, df = self.build_dataset()
        if df is None or not self.trained:
            return None
        X = df[self.feature_names].values
        cpu_preds = np.clip(self.cpu_model.predict(X), 0, 100)
        ram_preds = np.clip(self.ram_model.predict(X), 0, 100)
        return {
            "timestamps": df["timestamp"].tolist(),
            "actual_cpu": y_cpu.tolist(),
            "predicted_cpu": cpu_preds.tolist(),
            "actual_ram": y_ram.tolist(),
            "predicted_ram": ram_preds.tolist(),
        }

    def get_feature_importance(self, target="cpu"):
        """Return feature importances if model supports it."""
        pipeline = self.cpu_model if target == "cpu" else self.ram_model
        if pipeline is None:
            return {}
        model = pipeline.named_steps.get("model")
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(self.feature_names, model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
        elif hasattr(model, "coef_"):
            importance = dict(zip(self.feature_names, np.abs(model.coef_)))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
        return {}

    def _save_models(self):
        if self.cpu_model:
            joblib.dump(self.cpu_model, os.path.join(MODEL_DIR, "cpu_model.pkl"))
        if self.ram_model:
            joblib.dump(self.ram_model, os.path.join(MODEL_DIR, "ram_model.pkl"))

    def load_models(self):
        cpu_path = os.path.join(MODEL_DIR, "cpu_model.pkl")
        ram_path = os.path.join(MODEL_DIR, "ram_model.pkl")
        if os.path.exists(cpu_path) and os.path.exists(ram_path):
            self.cpu_model = joblib.load(cpu_path)
            self.ram_model = joblib.load(ram_path)
            self.trained = True
            return True
        return False
