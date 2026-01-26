import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import List, Tuple, Set
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARTIE 1: CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

def load_and_prepare_data(csv_path: str = None) -> pd.DataFrame:
    """
    Charge et prépare les données de capteurs.
    Si csv_path est None, utilise les données d'exemple.
    """
    df = pd.read_csv('data_encoded.csv')
    
    # Conversion timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    return df

def aggregate_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données par capteur pour calculer variance et position moyenne.
    """
    # Grouper par sensor_id et location
    agg_dict = {
        'lat': 'mean',
        'lon': 'mean',
        'PM2_5': ['mean', 'std', 'count'],
        'PM10': ['mean', 'std', 'count'],
        'sensor_type': 'first'
    }
    
    sensor_stats = df.groupby('sensor_id').agg(agg_dict).reset_index()
    sensor_stats.columns = ['sensor_id', 'lat', 'lon', 
                           'PM2_5_mean', 'PM2_5_std', 'PM2_5_count',
                           'PM10_mean', 'PM10_std', 'PM10_count',
                           'sensor_type']
    
    # Remplacer NaN variance par 0
    sensor_stats['PM2_5_std'] = sensor_stats['PM2_5_std'].fillna(0)
    sensor_stats['PM10_std'] = sensor_stats['PM10_std'].fillna(0)
    
    return sensor_stats


# ============================================================================
# PARTIE 2: FONCTIONS OBJECTIF
# ============================================================================

class SensorPlacementObjective:
    """
    Classe pour calculer la fonction objectif de placement de capteurs.
    """
    
    def __init__(self, 
                 candidate_locations: np.ndarray,
                 grid_locations: np.ndarray,
                 sensor_stats: pd.DataFrame,
                 alpha: float = 0.6,
                 beta: float = 0.3,
                 gamma: float = 0.1,
                 sigma: float = 0.01,
                 lambda_pm10: float = 0.3):
        """
        Args:
            candidate_locations: Array (N, 2) des positions candidates [lat, lon]
            grid_locations: Array (M, 2) des positions de la grille [lat, lon]
            sensor_stats: DataFrame avec variance PM2.5/PM10 par capteur
            alpha: Poids couverture spatiale
            beta: Poids détection hotspots
            gamma: Poids coût
            sigma: Paramètre de corrélation spatiale
            lambda_pm10: Poids PM10 relatif à PM2.5
        """
        self.candidate_locs = candidate_locations
        self.grid_locs = grid_locations
        self.sensor_stats = sensor_stats
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.lambda_pm10 = lambda_pm10
        
        # Calculer les distances une seule fois (optimisation)
        self.dist_matrix = cdist(grid_locations, candidate_locations, 
                                metric='euclidean')
    
    def spatial_coverage(self, selected_indices: Set[int]) -> float:
        """
        Calcul de la couverture spatiale I(S).
        I(S) = sum_i max_s exp(-d(i,s)^2 / 2*sigma^2)
        """
        if len(selected_indices) == 0:
            return 0.0
        
        selected_list = list(selected_indices)
        distances = self.dist_matrix[:, selected_list]
        
        # Pour chaque point de grille, prendre la distance minimale
        min_distances = np.min(distances, axis=1)
        
        # Kernel gaussien
        coverage = np.sum(np.exp(-min_distances**2 / (2 * self.sigma**2)))
        
        return coverage
    
    def hotspot_detection(self, selected_indices: Set[int]) -> float:
        """
        Calcul du score de détection de hotspots H(S).
        H(S) = sum_s [Var(PM2.5_s) + lambda * Var(PM10_s)]
        """
        if len(selected_indices) == 0:
            return 0.0
        
        selected_list = list(selected_indices)
        
        # Variance PM2.5
        pm25_var = self.sensor_stats.iloc[selected_list]['PM2_5_std'].values**2
        pm25_var = np.nan_to_num(pm25_var, 0)
        
        # Variance PM10
        pm10_var = self.sensor_stats.iloc[selected_list]['PM10_std'].values**2
        pm10_var = np.nan_to_num(pm10_var, 0)
        
        hotspot_score = np.sum(pm25_var + self.lambda_pm10 * pm10_var)
        
        return hotspot_score
    
    def cost(self, selected_indices: Set[int]) -> float:
        """
        Calcul du coût (simplifié: nombre de capteurs).
        """
        return len(selected_indices)
    
    def objective(self, selected_indices: Set[int]) -> float:
        """
        Fonction objectif complète (à MAXIMISER).
        f(S) = alpha * I(S) + beta * H(S) - gamma * Cost(S)
        """
        coverage = self.spatial_coverage(selected_indices)
        hotspot = self.hotspot_detection(selected_indices)
        cost_val = self.cost(selected_indices)
        
        obj_value = (self.alpha * coverage + 
                    self.beta * hotspot - 
                    self.gamma * cost_val)
        
        return obj_value
    
    def marginal_gain(self, 
                     current_set: Set[int], 
                     candidate_idx: int) -> float:
        """
        Calcul du gain marginal d'ajouter candidate_idx à current_set.
        Δf(s|S) = f(S ∪ {s}) - f(S)
        """
        new_set = current_set.copy()
        new_set.add(candidate_idx)
        
        gain = self.objective(new_set) - self.objective(current_set)
        return gain


# ============================================================================
# PARTIE 3: ALGORITHME GREEDY
# ============================================================================

def greedy_sensor_placement(objective: SensorPlacementObjective,
                            k: int,
                            d_min: float = 0.02,
                            verbose: bool = True) -> Tuple[List[int], List[float]]:
    """
    Algorithme Greedy pour placement de capteurs avec contrainte de diversité.
    
    Args:
        objective: Instance de SensorPlacementObjective
        k: Nombre maximum de capteurs
        d_min: Distance minimale entre capteurs (contrainte anti-clustering)
        verbose: Afficher la progression
    
    Returns:
        selected_sensors: Liste des indices de capteurs sélectionnés
        objective_values: Valeurs objectives à chaque itération
    """
    N = len(objective.candidate_locs)
    selected = set()
    objective_values = [0.0]
    
    if verbose:
        print(f"{'Iter':<6} {'Sensor':<8} {'Marginal Gain':<15} {'Objective':<12}")
        print("-" * 50)
    
    for iteration in range(k):
        best_gain = -np.inf
        best_sensor = None
        
        # Essayer tous les capteurs candidats
        for sensor_idx in range(N):
            if sensor_idx in selected:
                continue
            
            # Vérifier contrainte de diversité spatiale
            if not check_diversity_constraint(sensor_idx, selected, 
                                             objective.candidate_locs, d_min):
                continue
            
            # Calculer gain marginal
            gain = objective.marginal_gain(selected, sensor_idx)
            
            if gain > best_gain:
                best_gain = gain
                best_sensor = sensor_idx
        
        if best_sensor is None:
            if verbose:
                print(f"Aucun capteur valide trouvé à l'itération {iteration+1}")
            break
        
        # Ajouter le meilleur capteur
        selected.add(best_sensor)
        current_obj = objective.objective(selected)
        objective_values.append(current_obj)
        
        if verbose:
            print(f"{iteration+1:<6} {best_sensor:<8} {best_gain:<15.4f} {current_obj:<12.4f}")
    
    return list(selected), objective_values


def check_diversity_constraint(candidate_idx: int,
                               selected: Set[int],
                               locations: np.ndarray,
                               d_min: float) -> bool:
    """
    Vérifie si le candidat respecte la contrainte de distance minimale.
    """
    if len(selected) == 0:
        return True
    
    candidate_pos = locations[candidate_idx]
    selected_locs = locations[list(selected)]
    
    distances = np.sqrt(np.sum((selected_locs - candidate_pos)**2, axis=1))
    
    return np.all(distances >= d_min)


# ============================================================================
# PARTIE 4: OPTIMISATION AVEC GAUSSIAN PROCESS (AVANCÉ)
# ============================================================================

class GPBasedPlacement:
    """
    Placement de capteurs basé sur Gaussian Process pour variance prédictive.
    """
    
    def __init__(self, 
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 length_scale: float = 0.01):
        """
        Args:
            X_train: Positions d'entraînement (N, 2)
            y_train: Mesures PM2.5 (N,)
            length_scale: Échelle de longueur du kernel RBF
        """
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp.fit(X_train, y_train)
    
    def predict_variance(self, X_test: np.ndarray) -> np.ndarray:
        """
        Calcule la variance prédictive à chaque point de test.
        
        Returns:
            std: Écart-type prédictif à chaque point
        """
        _, std = self.gp.predict(X_test, return_std=True)
        return std
    
    def uncertainty_reduction(self, 
                             selected_locs: np.ndarray,
                             grid_locs: np.ndarray) -> float:
        """
        Calcule la réduction d'incertitude sur la grille.
        J_uncertainty = sum log(sigma^2(s|x))
        """
        if len(selected_locs) == 0:
            return 0.0
        
        # Récupérer les mesures aux positions sélectionnées
        y_selected, _ = self.gp.predict(selected_locs, return_std=True)
        
        # Entraîner nouveau GP avec ces positions
        kernel = self.gp.kernel_
        gp_new = GaussianProcessRegressor(kernel=kernel)
        gp_new.fit(selected_locs, y_selected)
        
        # Variance sur la grille
        std_grid = gp_new.predict(grid_locs, return_std=True)[1]
        
        # Log variance (à minimiser)
        uncertainty = np.sum(np.log(std_grid**2 + 1e-6))
        
        return -uncertainty  # Négatif car on veut maximiser


# ============================================================================
# PARTIE 5: VISUALISATION
# ============================================================================

def plot_sensor_placement(candidate_locs: np.ndarray,
                         selected_indices: List[int],
                         grid_locs: np.ndarray,
                         objective_values: List[float],
                         sensor_stats: pd.DataFrame):
    """
    Visualise le placement optimal des capteurs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Carte des capteurs
    ax1 = axes[0]
    
    # Tous les candidats
    ax1.scatter(candidate_locs[:, 1], candidate_locs[:, 0], 
               c='lightgray', s=100, alpha=0.5, label='Candidats')
    
    # Capteurs sélectionnés
    selected_locs = candidate_locs[selected_indices]
    ax1.scatter(selected_locs[:, 1], selected_locs[:, 0],
               c='red', s=200, marker='*', edgecolors='black',
               linewidths=1.5, label='Sélectionnés', zorder=5)
    
    # Annotations
    for idx in selected_indices:
        sensor_id = sensor_stats.iloc[idx]['sensor_id']
        ax1.annotate(f'{sensor_id}', 
                    (candidate_locs[idx, 1], candidate_locs[idx, 0]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='darkred')
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Placement Optimal des Capteurs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Évolution de l'objectif
    ax2 = axes[1]
    iterations = range(len(objective_values))
    ax2.plot(iterations, objective_values, 'b-o', linewidth=2)
    ax2.set_xlabel('Nombre de capteurs')
    ax2.set_ylabel('Valeur objective f(S)')
    ax2.set_title('Convergence de l\'algorithme Greedy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensor_placement.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# PARTIE 6: FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale pour exécuter l'optimisation.
    """
    print("="*70)
    print("OPTIMISATION DE PLACEMENT DE CAPTEURS DE QUALITÉ DE L'AIR")
    print("="*70)
    print()
    
    # 1. Charger les données
    print("1. Chargement des données...")
    df = load_and_prepare_data()
    sensor_stats = aggregate_sensor_data(df)
    print(f"   Nombre de capteurs: {len(sensor_stats)}")
    print(f"   Période: {df['timestamp'].min()} à {df['timestamp'].max()}")
    print()
    
    # 2. Préparer les positions
    print("2. Préparation des positions...")
    candidate_locs = sensor_stats[['lat', 'lon']].values
    
    # Créer une grille pour évaluation
    lat_min, lat_max = candidate_locs[:, 0].min(), candidate_locs[:, 0].max()
    lon_min, lon_max = candidate_locs[:, 1].min(), candidate_locs[:, 1].max()
    
    lat_grid = np.linspace(lat_min, lat_max, 20)
    lon_grid = np.linspace(lon_min, lon_max, 20)
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
    grid_locs = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    
    print(f"   Candidats: {len(candidate_locs)}")
    print(f"   Points de grille: {len(grid_locs)}")
    print()
    
    # 3. Créer l'objectif
    print("3. Configuration de la fonction objectif...")
    objective = SensorPlacementObjective(
        candidate_locations=candidate_locs,
        grid_locations=grid_locs,
        sensor_stats=sensor_stats,
        alpha=0.6,
        beta=0.3,
        gamma=0.1,
        sigma=0.01,
        lambda_pm10=0.3
    )
    print(f"   α (couverture) = {objective.alpha}")
    print(f"   β (hotspots)   = {objective.beta}")
    print(f"   γ (coût)       = {objective.gamma}")
    print()
    
    # 4. Optimisation Greedy
    print("4. Optimisation Greedy...")
    print()
    k_max = min(20, len(candidate_locs))  # Placer au max 5 capteurs
    selected, obj_values = greedy_sensor_placement(
        objective=objective,
        k=k_max,
        d_min=0.02,  # 2 km minimum entre capteurs
        verbose=True
    )
    print()
    
    # 5. Résultats
    print("="*70)
    print("RÉSULTATS FINAUX")
    print("="*70)
    print(f"Capteurs sélectionnés: {selected}")
    print(f"Valeur objective finale: {obj_values[-1]:.4f}")
    print()
    
    print("Détails des capteurs sélectionnés:")
    print("-" * 70)
    for idx in selected:
        row = sensor_stats.iloc[idx]
        print(f"  Sensor {int(row['sensor_id'])} @ ({row['lat']:.4f}, {row['lon']:.4f})")
        print(f"    Type: {row['sensor_type']}")
        print(f"    PM2.5: μ={row['PM2_5_mean']:.2f}, σ={row['PM2_5_std']:.2f}")
        print(f"    PM10:  μ={row['PM10_mean']:.2f}, σ={row['PM10_std']:.2f}")
        print()
    
    # 6. Visualisation
    print("6. Génération des visualisations...")
    plot_sensor_placement(candidate_locs, selected, grid_locs, 
                         obj_values, sensor_stats)
    print("   Graphique sauvegardé: sensor_placement.png")
    print()
    
    print("="*70)
    print("OPTIMISATION TERMINÉE")
    print("="*70)


if __name__ == "__main__":
    main()