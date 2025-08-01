import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

def compute_similarity_and_distance(matrix: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la matrice de similarité et de distance à partir d'une matrice sparse.

    Args:
        matrix: Matrice de caractéristiques (CSR), shape [n_documents, n_features]

    Returns:
        - Matrice de similarité (cosine)
        - Matrice de distance (cosine)
    """
    print("[INFO] Calcul des matrices de similarité et de distance...")
    similarity = cosine_similarity(matrix)
    distance = pairwise_distances(matrix, metric="cosine")

    return similarity, distance
