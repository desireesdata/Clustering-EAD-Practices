from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_k(
    affinity_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    plot: bool = True
) -> tuple[int, list[float]]:
    """
    Calcule les scores de silhouette pour différents k et retourne le meilleur.

    Args:
        affinity_matrix: Matrice de similarité [n_docs x n_docs]
        distance_matrix: Matrice de distance [n_docs x n_docs]
        k_min: Valeur minimale de k testée
        k_max: Valeur maximale de k testée
        plot: Affiche ou non un graphique

    Returns:
        - k optimal
        - liste des scores silhouette pour chaque k
    """
    scores = []
    ks = range(k_min, min(k_max + 1, affinity_matrix.shape[0]))

    print(f"[INFO] Recherche du meilleur k entre {k_min} et {k_max}...")
    for k in ks:
        try:
            sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42, n_init=10)
            labels = sc.fit_predict(affinity_matrix)

            if len(set(labels)) > 1:
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
            else:
                score = -1
        except Exception as e:
            print(f"[WARN] Clustering échoué pour k={k} : {e}")
            score = -1

        scores.append(score)

    best_k = ks[np.argmax(scores)]

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(ks, scores, marker='o')
        plt.title("Silhouette score selon le nombre de clusters")
        plt.xlabel("k (nombre de clusters)")
        plt.ylabel("Score de silhouette")
        plt.grid(True)
        plt.xticks(ks)
        plt.axvline(best_k, color='r', linestyle='--', label=f"k optimal = {best_k}")
        plt.legend()
        plt.show()

    return best_k, scores

def run_clustering(
    affinity_matrix: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Applique Spectral Clustering avec la matrice d'affinité et retourne les labels.

    Args:
        affinity_matrix: Matrice de similarité [n_docs x n_docs]
        k: nombre de clusters

    Returns:
        - np.ndarray: labels des clusters
    """
    print(f"[INFO] Clustering spectral en cours (k = {k})...")
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42, n_init=10)
    labels = sc.fit_predict(affinity_matrix)
    return labels
