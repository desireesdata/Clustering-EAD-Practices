import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path
from collections import Counter
from typing import List, Dict

def summarize_clusters(
    labels: np.ndarray,
    feature_matrix: csr_matrix,
    index_to_path: Dict[int, str],
    xml_files: List[Path],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Résume les clusters structurels en listant les fichiers et chemins dominants.

    Args:
        labels: Tableau des labels de clusters [n_documents]
        feature_matrix: Matrice CSR des features [n_documents x n_features]
        index_to_path: Dictionnaire index → chemin XML
        xml_files: Liste des fichiers XML, même ordre que les lignes de la matrice
        top_n: Nombre de chemins les plus fréquents à afficher par cluster

    Returns:
        DataFrame avec infos sur chaque document et son cluster
    """
    df = pd.DataFrame({
        'filepath': [str(p) for p in xml_files],
        'cluster': labels
    })

    n_clusters = len(set(labels))
    print(f"[INFO] Résumé de {n_clusters} clusters...\n")

    for cluster_id in sorted(set(labels)):
        indices = df[df['cluster'] == cluster_id].index
        n_docs = len(indices)

        print(f"\n--- Cluster {cluster_id} ({n_docs} documents) ---")
        print("Fichiers :")
        for path in df[df['cluster'] == cluster_id]['filepath'].head(5):
            print(f"  - {path}")

        if n_docs == 0:
            print("Aucun document.")
            continue

        cluster_matrix = feature_matrix[indices]
        summed = np.array(cluster_matrix.sum(axis=0)).flatten()
        top_indices = summed.argsort()[-top_n:][::-1]

        print("\nBalises les plus fréquentes :")
        for i in top_indices:
            chemin = index_to_path.get(i, f"<inconnu {i}>")
            count = int(summed[i])
            print(f"  - {chemin} : {count}")

    return df
