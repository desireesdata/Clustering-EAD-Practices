from pathlib import Path
import argparse

from ead_clustering.features import build_feature_matrix
from ead_clustering.similarity import compute_similarity_and_distance
from ead_clustering.clustering import find_optimal_k, run_clustering
from ead_clustering.analysis import summarize_clusters

def main(ead_folder: str, k_manual: int | None = None):
    # 1. Récupération des fichiers
    xml_files = sorted(Path(ead_folder).rglob("*.xml"))
    if not xml_files:
        print(f"[ERREUR] Aucun fichier XML trouvé dans le dossier : {ead_folder}")
        return

    print(f"[INFO] {len(xml_files)} fichiers XML détectés.")

    # 2. Construction de la matrice structurelle
    feature_matrix, index_to_path, ordered_files = build_feature_matrix(xml_files)

    # 3. Calcul des similarités
    affinity, distance = compute_similarity_and_distance(feature_matrix)

    # 4. Choix du k
    if k_manual is not None:
        k = k_manual
        print(f"[INFO] Nombre de clusters forcé à {k}")
    else:
        k, _ = find_optimal_k(affinity, distance)

    # 5. Clustering
    labels = run_clustering(affinity, k)

    # 6. Résumé et affichage
    df_results = summarize_clusters(labels, feature_matrix, index_to_path, ordered_files)

    # 7. Export CSV (optionnel)
    from ead_clustering.utils import export_results_dataframe
    export_results_dataframe(df_results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse structurelle de fichiers XML (EAD)")
    parser.add_argument("ead_folder", help="Chemin vers le dossier contenant les fichiers XML")
    parser.add_argument("--k", type=int, default=None, help="Nombre de clusters à forcer (optionnel)")

    args = parser.parse_args()
    main(ead_folder=args.ead_folder, k_manual=args.k)
