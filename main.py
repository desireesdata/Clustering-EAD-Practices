# main.py (version intégrée avec détection d'anomalies)

import argparse
from pathlib import Path
from ead_clustering.features import build_feature_matrix
from ead_clustering.clustering import determine_optimal_k, run_clustering, describe_clusters
from ead_clustering.anomaly import detect_anomalies, plot_anomalies, export_anomalies_csv


def main():
    parser = argparse.ArgumentParser(description="Analyse structurelle de fichiers EAD avec clustering spectral.")
    parser.add_argument("ead_folder", help="Chemin vers le dossier contenant les fichiers XML EAD")
    parser.add_argument("--k", type=int, help="Nombre de clusters à utiliser")
    parser.add_argument("--detect-anomalies", action="store_true", help="Activer la détection d'anomalies")
    args = parser.parse_args()

    # Étape 1 : Construction des matrices
    structural_matrix, affinity_matrix, distance_matrix, paths_vocab, filepaths = build_feature_matrix(Path(args.ead_folder))

    # Étape 2 : Clustering
    if args.k is None:
        determine_optimal_k(affinity_matrix, distance_matrix)
        print("[INFO] Veuillez relancer avec un nombre de clusters via --k.")
        return

    cluster_labels = run_clustering(affinity_matrix, args.k)
    describe_clusters(cluster_labels, filepaths, structural_matrix, paths_vocab)

    # Étape 3 : Anomalies (optionnel)
    if args.detect_anomalies:
        print("\n[INFO] Détection des anomalies activée...")
        df_anomalies = detect_anomalies(affinity_matrix, distance_matrix, cluster_labels, filepaths)
        plot_anomalies(df_anomalies)
        export_anomalies_csv(df_anomalies)


if __name__ == "__main__":
    main()
