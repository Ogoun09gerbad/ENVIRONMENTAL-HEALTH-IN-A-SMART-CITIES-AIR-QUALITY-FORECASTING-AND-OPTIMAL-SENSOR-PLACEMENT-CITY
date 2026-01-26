"""
Prédiction Spatio-Temporelle PM2.5 avec Placement Optimal de Capteurs
======================================================================

Ce code implémente:
1. ARIMA pour prédiction temporelle par capteur
2. Prophet pour tendances et saisonnalité
3. XGBoost avec features spatio-temporelles
4. LSTM pour séquences temporelles
5. Démonstration de l'impact du placement optimal

Installation requise:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
pip install statsmodels prophet tensorflow keras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Imports ML/Stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import cdist

# Tentative d'importation des modèles (optionnels)
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️ ARIMA non disponible. Installez: pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet non disponible. Installez: pip install prophet")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost non disponible. Installez: pip install xgboost")

try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️ LSTM non disponible. Installez: pip install tensorflow keras")


# ============================================================================
# PARTIE 1: CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """Charge les données depuis CSV."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    return df


def prepare_spatiotemporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features spatio-temporelles complètes.
    """
    df = df.copy()
    df = df.sort_values(['sensor_id', 'timestamp'])
    
    # ===== FEATURES TEMPORELLES =====
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    
    # Features cycliques (capture la périodicité)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ===== FEATURES DE LAG (valeurs passées) =====
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'PM2_5_lag_{lag}'] = df.groupby('sensor_id')['PM2_5'].shift(lag)
        df[f'PM10_lag_{lag}'] = df.groupby('sensor_id')['PM10'].shift(lag)
    
    # ===== ROLLING STATISTICS =====
    for window in [3, 6, 12, 24]:
        df[f'PM2_5_rolling_mean_{window}'] = (
            df.groupby('sensor_id')['PM2_5']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df[f'PM2_5_rolling_std_{window}'] = (
            df.groupby('sensor_id')['PM2_5']
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
    
    # ===== FEATURES SPATIALES =====
    # Distance au centre de la ville (moyenne des positions)
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    df['dist_to_center'] = np.sqrt(
        (df['lat'] - center_lat)**2 + (df['lon'] - center_lon)**2
    )
    
    # Encodage de la position
    df['lat_norm'] = (df['lat'] - df['lat'].min()) / (df['lat'].max() - df['lat'].min())
    df['lon_norm'] = (df['lon'] - df['lon'].min()) / (df['lon'].max() - df['lon'].min())
    
    return df


def get_spatial_neighbors(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Ajoute les valeurs des k capteurs voisins les plus proches.
    """
    df = df.copy()
    
    # Grouper par timestamp
    for timestamp in df['timestamp'].unique():
        mask = df['timestamp'] == timestamp
        df_t = df[mask].copy()
        
        if len(df_t) < 2:
            continue
        
        # Calculer distances entre capteurs
        positions = df_t[['lat', 'lon']].values
        distances = cdist(positions, positions)
        
        # Pour chaque capteur, trouver les k voisins
        for i, idx in enumerate(df_t.index):
            # Trier par distance (exclure soi-même)
            neighbors_idx = np.argsort(distances[i])[1:k+1]
            
            # Moyenne PM2.5 des voisins
            neighbor_values = df_t.iloc[neighbors_idx]['PM2_5'].values
            df.loc[idx, f'neighbors_PM2_5_mean'] = np.nanmean(neighbor_values)
            df.loc[idx, f'neighbors_PM2_5_std'] = np.nanstd(neighbor_values)
    
    return df


# ============================================================================
# PARTIE 2: MODÈLE 1 - ARIMA (Temporel Simple)
# ============================================================================

class ARIMAPredictor:
    """Modèle ARIMA pour prédiction temporelle par capteur."""
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        self.order = order
        self.models = {}  # Un modèle par capteur
    
    def fit(self, df: pd.DataFrame, sensor_id: int):
        """Entraîne ARIMA pour un capteur spécifique."""
        if not ARIMA_AVAILABLE:
            raise ImportError("ARIMA non disponible")
        
        df_sensor = df[df['sensor_id'] == sensor_id].copy()
        df_sensor = df_sensor.sort_values('timestamp')
        
        # Prendre seulement les valeurs non-nulles
        y = df_sensor['PM2_5'].dropna()
        
        if len(y) < 50:
            print(f"⚠️ Pas assez de données pour capteur {sensor_id}")
            return None
        
        # Entraîner ARIMA
        model = ARIMA(y, order=self.order)
        fitted_model = model.fit()
        
        self.models[sensor_id] = fitted_model
        return fitted_model
    
    def predict(self, sensor_id: int, steps: int = 24) -> np.ndarray:
        """Prédit les prochains 'steps' points."""
        if sensor_id not in self.models:
            return None
        
        forecast = self.models[sensor_id].forecast(steps=steps)
        return forecast
    
    def evaluate(self, df: pd.DataFrame, sensor_id: int, 
                 test_size: int = 100) -> Dict[str, float]:
        """Évalue les performances sur les derniers points."""
        df_sensor = df[df['sensor_id'] == sensor_id].copy()
        df_sensor = df_sensor.sort_values('timestamp')
        y = df_sensor['PM2_5'].dropna()
        
        # Split train/test
        y_train = y[:-test_size]
        y_test = y[-test_size:]
        
        # Réentraîner sur train
        model = ARIMA(y_train, order=self.order).fit()
        
        # Prédire test
        y_pred = model.forecast(steps=test_size)
        
        # Métriques
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        return metrics


# ============================================================================
# PARTIE 3: MODÈLE 2 - PROPHET (Tendances + Saisonnalité)
# ============================================================================

class ProphetPredictor:
    """Modèle Prophet pour tendances et saisonnalité."""
    
    def __init__(self):
        self.models = {}
    
    def fit(self, df: pd.DataFrame, sensor_id: int):
        """Entraîne Prophet pour un capteur."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet non disponible")
        
        df_sensor = df[df['sensor_id'] == sensor_id].copy()
        df_sensor = df_sensor[['timestamp', 'PM2_5']].dropna()
        df_sensor.columns = ['ds', 'y']
        
        if len(df_sensor) < 50:
            return None
        
        # Créer et entraîner Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        model.fit(df_sensor)
        
        self.models[sensor_id] = model
        return model
    
    def predict(self, sensor_id: int, periods: int = 24) -> pd.DataFrame:
        """Prédit les prochains 'periods' points."""
        if sensor_id not in self.models:
            return None
        
        # Créer future dataframe
        future = self.models[sensor_id].make_future_dataframe(
            periods=periods, freq='H'
        )
        forecast = self.models[sensor_id].predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# ============================================================================
# PARTIE 4: MODÈLE 3 - XGBOOST SPATIO-TEMPOREL
# ============================================================================

class XGBoostSpatioTemporalPredictor:
    """XGBoost avec features spatio-temporelles complètes."""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.scaler = StandardScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prépare X et y."""
        df = prepare_spatiotemporal_features(df)
        
        # Sélectionner les features de base
        self.feature_cols = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'lat_norm', 'lon_norm', 'dist_to_center'
        ]
        
        # Ajouter features disponibles
        optional_features = ['temperature', 'humidity', 'PM10']
        for feat in optional_features:
            if feat in df.columns:
                self.feature_cols.append(feat)
        
        # Ajouter lags et rolling seulement si disponibles
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        self.feature_cols.extend(lag_cols)
        
        # Filtrer colonnes existantes
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]
        
        print(f"  - Features disponibles: {len(self.feature_cols)}")
        
        # Remplir les valeurs manquantes
        df_filled = df.copy()
        for col in self.feature_cols:
            if df_filled[col].dtype in ['float64', 'int64']:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            else:
                df_filled[col] = df_filled[col].fillna(0)
        
        # Supprimer les lignes où PM2.5 est NaN
        df_clean = df_filled.dropna(subset=['PM2_5'])
        
        print(f"  - Échantillons valides: {len(df_clean)}")
        
        if len(df_clean) == 0:
            print("⚠️ Pas de données valides après préparation")
            return pd.DataFrame(), np.array([])
        
        X = df_clean[self.feature_cols]
        y = df_clean['PM2_5'].values
        
        return X, y
    
    def fit(self, df: pd.DataFrame):
        """Entraîne XGBoost."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost non disponible")
        
        X, y = self.prepare_features(df)
        
        if len(X) == 0 or len(y) == 0:
            print("⚠️ Pas assez de données pour entraîner XGBoost")
            return {
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'R2': 0.0,
                'y_test': np.array([]),
                'y_pred': np.array([])
            }
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Pas de shuffle pour séries temporelles!
        )
        
        # Normaliser
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entraîner XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Évaluer
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Prédit PM2.5 pour nouvelles données."""
        df = prepare_spatiotemporal_features(df)
        df = get_spatial_neighbors(df, k=3)
        
        X = df[self.feature_cols].fillna(df[self.feature_cols].mean())
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des features."""
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


# ============================================================================
# PARTIE 5: MODÈLE 4 - LSTM (Deep Learning)
# ============================================================================

class LSTMPredictor:
    """LSTM pour prédiction de séquences temporelles."""
    
    def __init__(self, lookback: int = 24, forecast_horizon: int = 6):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def create_sequences(self, df: pd.DataFrame, sensor_id: int):
        """Crée des séquences pour LSTM."""
        df_sensor = df[df['sensor_id'] == sensor_id].copy()
        df_sensor = df_sensor.sort_values('timestamp')
        
        # Features: PM2.5, température, humidité
        feature_cols = ['PM2_5', 'temperature', 'humidity']
        data = df_sensor[feature_cols].dropna().values
        
        if len(data) < self.lookback + self.forecast_horizon + 50:
            return None, None
        
        # Normaliser
        data_scaled = self.scaler_X.fit_transform(data)
        
        # Créer séquences
        X, y = [], []
        for i in range(len(data_scaled) - self.lookback - self.forecast_horizon):
            X.append(data_scaled[i:i+self.lookback])
            y.append(data_scaled[i+self.lookback:i+self.lookback+self.forecast_horizon, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Construit l'architecture LSTM."""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, 
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, df: pd.DataFrame, sensor_id: int):
        """Entraîne LSTM."""
        if not LSTM_AVAILABLE:
            raise ImportError("LSTM non disponible")
        
        X, y = self.create_sequences(df, sensor_id)
        
        if X is None:
            print(f"⚠️ Pas assez de données pour LSTM (capteur {sensor_id})")
            return None
        
        # Split train/test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Construire modèle
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Entraîner
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Évaluer
        y_pred = self.model.predict(X_test)
        
        # Dénormaliser pour calcul métriques
        y_test_orig = self.scaler_X.inverse_transform(
            np.concatenate([y_test, np.zeros((y_test.shape[0], 2))], axis=1)
        )[:, 0]
        y_pred_orig = self.scaler_X.inverse_transform(
            np.concatenate([y_pred, np.zeros((y_pred.shape[0], 2))], axis=1)
        )[:, 0]
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'MAE': mean_absolute_error(y_test_orig, y_pred_orig),
            'R2': r2_score(y_test_orig, y_pred_orig),
            'history': history.history
        }
        
        return metrics


# ============================================================================
# PARTIE 6: DÉMONSTRATION IMPACT PLACEMENT OPTIMAL
# ============================================================================

def demonstrate_sensor_placement_impact(df: pd.DataFrame):
    """
    Démontre comment le placement optimal améliore les prédictions.
    """
    print("\n" + "="*70)
    print("DÉMONSTRATION: IMPACT DU PLACEMENT OPTIMAL DES CAPTEURS")
    print("="*70 + "\n")
    
    # Scénario 1: Tous les capteurs (baseline)
    print("📊 Scénario 1: Tous les capteurs disponibles")
    model_all = XGBoostSpatioTemporalPredictor()
    metrics_all = model_all.fit(df)
    
    print(f"  RMSE: {metrics_all['RMSE']:.2f} μg/m³")
    print(f"  MAE:  {metrics_all['MAE']:.2f} μg/m³")
    print(f"  R²:   {metrics_all['R2']:.4f}")
    print()
    
    # Scénario 2: Placement aléatoire (50% des capteurs)
    print("🎲 Scénario 2: Placement aléatoire (50% des capteurs)")
    sensor_ids = df['sensor_id'].unique()
    n_sensors = len(sensor_ids) // 2
    random_sensors = np.random.choice(sensor_ids, n_sensors, replace=False)
    df_random = df[df['sensor_id'].isin(random_sensors)]
    
    model_random = XGBoostSpatioTemporalPredictor()
    metrics_random = model_random.fit(df_random)
    
    print(f"  RMSE: {metrics_random['RMSE']:.2f} μg/m³")
    print(f"  MAE:  {metrics_random['MAE']:.2f} μg/m³")
    print(f"  R²:   {metrics_random['R2']:.4f}")
    print()
    
    # Scénario 3: Placement optimal (capteurs avec variance élevée + bien espacés)
    print("🎯 Scénario 3: Placement optimal (50% des capteurs)")
    print("  Critères: Variance élevée + Diversité spatiale")
    
    # Calculer variance par capteur
    sensor_variance = df.groupby('sensor_id')['PM2_5'].std().reset_index()
    sensor_variance.columns = ['sensor_id', 'variance']
    
    # Obtenir positions
    sensor_positions = df.groupby('sensor_id')[['lat', 'lon']].mean().reset_index()
    sensor_variance = sensor_variance.merge(sensor_positions, on='sensor_id')
    
    # Sélection greedy avec diversité
    optimal_sensors = greedy_diverse_selection(
        sensor_variance, n_sensors, d_min=0.02
    )
    
    df_optimal = df[df['sensor_id'].isin(optimal_sensors)]
    
    model_optimal = XGBoostSpatioTemporalPredictor()
    metrics_optimal = model_optimal.fit(df_optimal)
    
    print(f"  RMSE: {metrics_optimal['RMSE']:.2f} μg/m³")
    print(f"  MAE:  {metrics_optimal['MAE']:.2f} μg/m³")
    print(f"  R²:   {metrics_optimal['R2']:.4f}")
    print()
    
    # Comparaison
    print("📈 AMÉLIORATION DU PLACEMENT OPTIMAL:")
    improvement_rmse = ((metrics_random['RMSE'] - metrics_optimal['RMSE']) / 
                       metrics_random['RMSE'] * 100)
    improvement_r2 = ((metrics_optimal['R2'] - metrics_random['R2']) / 
                     metrics_random['R2'] * 100)
    
    print(f"  • Réduction RMSE: {improvement_rmse:.1f}%")
    print(f"  • Amélioration R²: {improvement_r2:.1f}%")
    print()
    
    return {
        'all': metrics_all,
        'random': metrics_random,
        'optimal': metrics_optimal
    }


def greedy_diverse_selection(sensor_df: pd.DataFrame, k: int, d_min: float):
    """Sélection greedy avec contrainte de diversité."""
    selected = []
    positions = sensor_df[['lat', 'lon']].values
    variances = sensor_df['variance'].values
    sensor_ids = sensor_df['sensor_id'].values
    
    # Premier capteur: variance maximale
    first_idx = np.argmax(variances)
    selected.append(sensor_ids[first_idx])
    
    # Sélection itérative
    for _ in range(k - 1):
        best_score = -np.inf
        best_sensor = None
        
        for i, sensor_id in enumerate(sensor_ids):
            if sensor_id in selected:
                continue
            
            # Vérifier distance minimale
            selected_positions = positions[[np.where(sensor_ids == s)[0][0] 
                                          for s in selected]]
            distances = np.sqrt(np.sum((selected_positions - positions[i])**2, axis=1))
            
            if np.all(distances >= d_min):
                # Score = variance
                if variances[i] > best_score:
                    best_score = variances[i]
                    best_sensor = sensor_id
        
        if best_sensor is not None:
            selected.append(best_sensor)
    
    return selected


# ============================================================================
# PARTIE 7: VISUALISATIONS
# ============================================================================

def plot_comparison_results(results: Dict):
    """Visualise la comparaison des 3 scénarios."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    scenarios = ['all', 'random', 'optimal']
    titles = ['Tous les capteurs', 'Placement Aléatoire', 'Placement Optimal']
    
    for idx, (scenario, title) in enumerate(zip(scenarios, titles)):
        metrics = results[scenario]
        y_test = metrics['y_test']
        y_pred = metrics['y_pred']
        
        # Plot 1: Série temporelle
        ax1 = axes[0, idx]
        time_idx = np.arange(len(y_test))
        ax1.plot(time_idx, y_test, 'b-', label='Réel', alpha=0.7, linewidth=1.5)
        ax1.plot(time_idx, y_pred, 'r--', label='Prédit', alpha=0.7, linewidth=1.5)
        ax1.set_title(f'{title}\nRMSE={metrics["RMSE"]:.2f}')
        ax1.set_xlabel('Index temporel')
        ax1.set_ylabel('PM2.5 (μg/m³)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter
        ax2 = axes[1, idx]
        ax2.scatter(y_test, y_pred, alpha=0.5, s=30)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax2.set_title(f'R² = {metrics["R2"]:.4f}')
        ax2.set_xlabel('PM2.5 Réel')
        ax2.set_ylabel('PM2.5 Prédit')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensor_placement_impact.png', dpi=300, bbox_inches='tight')
    print("📊 Graphique sauvegardé: sensor_placement_impact.png")
    plt.show()


def plot_model_comparison(df: pd.DataFrame, sensor_id: int):
    """Compare tous les modèles pour un capteur."""
    print(f"\n🔬 Comparaison des modèles pour capteur {sensor_id}\n")
    
    results = {}
    
    # ARIMA
    if ARIMA_AVAILABLE:
        print("Testing ARIMA...")
        arima = ARIMAPredictor()
        arima.fit(df, sensor_id)
        results['ARIMA'] = arima.evaluate(df, sensor_id)
    
    # Prophet
    if PROPHET_AVAILABLE:
        print("Testing Prophet...")
        # Prophet ne retourne pas de métriques facilement comparables
        pass
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("Testing XGBoost...")
        df_sensor = df[df['sensor_id'] == sensor_id]
        xgb_model = XGBoostSpatioTemporalPredictor()
        results['XGBoost'] = xgb_model.fit(df_sensor)
    
    # LSTM
    if LSTM_AVAILABLE:
        print("Testing LSTM...")
        lstm = LSTMPredictor()
        results['LSTM'] = lstm.fit(df, sensor_id)
    
    # Visualiser comparaison
    if len(results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(results.keys())
        rmse_values = [results[m]['RMSE'] for m in models]
        mae_values = [results[m]['MAE'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
        ax.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
        
        ax.set_xlabel('Modèle')
        ax.set_ylabel('Erreur (μg/m³)')
        ax.set_title(f'Comparaison des Modèles - Capteur {sensor_id}')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'model_comparison_sensor_{sensor_id}.png', dpi=300, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé: model_comparison_sensor_{sensor_id}.png")
        plt.show()


# ============================================================================
# PARTIE 8: FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale."""
    print("="*70)
    print("OPTIMISATION DE PLACEMENT DE CAPTEURS DE QUALITÉ DE L'AIR")
    print("="*70)
    
    # 1. Chargement des données
    print("1. Chargement des données...")
    df = load_data('data_encoded.csv')
    print(f"   Nombre de capteurs: {df['sensor_id'].nunique()}")
    print(f"   Période: {df['timestamp'].min()} à {df['timestamp'].max()}")
    
    # 2. Démonstration de l'impact du placement optimal
    results = demonstrate_sensor_placement_impact(df)
    
    # 3. Visualisation des résultats
    plot_comparison_results(results)
    
    # 4. Comparaison des modèles pour un capteur représentatif
    sensor_id = df['sensor_id'].value_counts().index[0]  # Capteur avec le plus de données
    plot_model_comparison(df, sensor_id)
    
    print("\n" + "="*70)
    print("ANALYSE COMPLÈTE TERMINÉE")
    print("="*70)


if __name__ == "__main__":
    main()