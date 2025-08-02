from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from .xml_parser import extract_xml_paths

def build_feature_matrix(ead_folder_path):
    """
    Lit tous les fichiers XML dans le dossier donné, extrait les chemins XML,
    construit la matrice structurelle et les matrices de similarité/distance.
    """
    xml_files = list(Path(ead_folder_path).rglob("*.xml"))
    if not xml_files:
        raise ValueError(f"Aucun fichier XML trouvé dans le dossier : {ead_folder_path}")

    print(f"[INFO] Nombre de fichiers XML trouvés : {len(xml_files)}")

    path_counts = []
    vocabulary = {}

    for i, xml_file in enumerate(xml_files):
        paths = extract_xml_paths(xml_file)
        path_count = defaultdict(int)
        for path in paths:
            if path not in vocabulary:
                vocabulary[path] = len(vocabulary)
            path_count[vocabulary[path]] += 1
        path_counts.append(path_count)

    n_docs = len(xml_files)
    n_features = len(vocabulary)
    matrix = lil_matrix((n_docs, n_features), dtype=np.float64)

    for i, counts in enumerate(path_counts):
        for index, count in counts.items():
            matrix[i, index] = count

    csr_matrix = matrix.tocsr()
    affinity = cosine_similarity(csr_matrix)
    distance = pairwise_distances(csr_matrix, metric='cosine')

    return csr_matrix, affinity, distance, vocabulary, xml_files