from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from .xml_parser import extract_xml_paths

def build_feature_matrix(xml_files: List[Path]) -> Tuple[csr_matrix, Dict[int, str], List[Path]]:
    """
    Construit une matrice creuse représentant la fréquence des chemins XML par document.

    Args:
        xml_files: Liste de fichiers XML à traiter.

    Returns:
        - csr_matrix: Matrice [n_documents x n_features]
        - Dict[int, str]: index → nom du chemin
        - List[Path]: fichiers traités dans l’ordre
    """
    paths_vocabulary = {}
    path_counts = []

    for i, xml_file in enumerate(xml_files):
        paths = extract_xml_paths(xml_file)
        counts = defaultdict(int)

        for path in paths:
            if path not in paths_vocabulary:
                paths_vocabulary[path] = len(paths_vocabulary)
            counts[paths_vocabulary[path]] += 1

        path_counts.append(counts)

    n_documents = len(xml_files)
    n_features = len(paths_vocabulary)
    matrix = lil_matrix((n_documents, n_features), dtype=np.float64)

    for i, counts in enumerate(path_counts):
        for path_index, count in counts.items():
            matrix[i, path_index] = count

    index_to_path = {index: path for path, index in paths_vocabulary.items()}
    return matrix.tocsr(), index_to_path, xml_files
