from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh  # pour l'eigengap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

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

def determine_optimal_k(affinity_matrix, distance_matrix, max_k=15):
    """
    Affiche deux graphiques : valeurs propres du Laplacien (eigengap)
    et scores de silhouette pour t'aider à choisir un nombre de clusters optimal.
    """
    print("[INFO] Calcul des valeurs propres (eigengap)...")
    laplacian_matrix = laplacian(affinity_matrix, normed=True)
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix)
    eigenvalues_sorted = np.sort(eigenvalues)

    plt.figure(figsize=(10, 4))
    plt.plot(eigenvalues_sorted[:30], marker='o')
    plt.title("Valeurs propres du Laplacien (eigengap)")
    plt.xlabel("Index")
    plt.ylabel("Valeur propre")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("[INFO] Calcul des scores de silhouette...")
    silhouette_scores = []
    k_range = range(2, min(max_k, affinity_matrix.shape[0] - 1))

    for k in k_range:
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42, n_init=10)
        labels = sc.fit_predict(affinity_matrix)
        if len(set(labels)) > 1:
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)

    plt.figure(figsize=(10, 4))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title("Score de silhouette en fonction de k")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Silhouette score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def describe_clusters(labels, filepaths, feature_matrix, vocabulary, top_n=10):
    """
    Affiche les chemins structurels les plus fréquents pour chaque cluster.
    """
    import pandas as pd
    from collections import defaultdict

    print("\n[INFO] Description des clusters structurels")
    df = pd.DataFrame({
        'filepath': [str(p) for p in filepaths],
        'label': labels
    })

    index_to_path = {index: path for path, index in vocabulary.items()}
    n_clusters = len(set(labels))

    for cluster_id in range(n_clusters):
        docs_in_cluster = df[df['label'] == cluster_id].index
        print(f"\n--- Cluster {cluster_id} ({len(docs_in_cluster)} documents) ---")

        for doc_path in df[df['label'] == cluster_id]['filepath'].tolist():
            print(f"  - {doc_path}")

        if len(docs_in_cluster) == 0:
            print("Aucun document.")
            continue

        cluster_matrix = feature_matrix[docs_in_cluster]
        summed = cluster_matrix.sum(axis=0).A1
        top_indices = summed.argsort()[-top_n:][::-1]

        print("Chemins structurels les plus fréquents :")
        for idx in top_indices:
            print(f"  - {index_to_path[idx]} : {int(summed[idx])}")
