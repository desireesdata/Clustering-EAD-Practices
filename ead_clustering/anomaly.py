# anomaly.py

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns


def detect_anomalies(affinity_matrix, distance_matrix, cluster_labels, filepaths):
    """
    Calcule des indicateurs d'anomalie pour chaque document :
    - score de silhouette
    - affinité moyenne avec les autres
    - cluster d'appartenance
    - suspicion binaire (score faible ou affinité très faible)
    """
    if len(set(cluster_labels)) < 2:
        print("[INFO] Seulement un cluster détecté, la détection d'anomalies par silhouette n'est pas possible.")
        silhouette_scores = np.zeros(len(cluster_labels))
    else:
        silhouette_scores = silhouette_samples(distance_matrix, cluster_labels, metric='precomputed')

    mean_affinity = affinity_matrix.mean(axis=1)

    df = pd.DataFrame({
        'filepath': filepaths,
        'cluster': cluster_labels,
        'silhouette_score': silhouette_scores,
        'mean_affinity': mean_affinity,
    })

    # Heuristique de détection
    low_silhouette = df['silhouette_score'] < 0.1
    low_affinity = df['mean_affinity'] < 0.1
    df['suspect'] = low_silhouette | low_affinity

    return df.sort_values(by='suspect', ascending=False)


def plot_anomalies(df_anomalies):
    """
    Affiche un nuage de points : score de silhouette vs affinité moyenne.
    Les points rouges sont considérés comme suspects.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_anomalies,
        x='silhouette_score',
        y='mean_affinity',
        hue='suspect',
        palette={True: 'red', False: 'green'},
        alpha=0.8
    )
    plt.title("Détection d'anomalies dans les structures EAD")
    plt.xlabel("Score de silhouette")
    plt.ylabel("Affinité moyenne avec les autres")
    plt.grid(True)
    plt.legend(title='Suspect')
    plt.tight_layout()
    plt.show()


def export_anomalies_csv(df_anomalies, output_path="anomalies.csv"):
    """Export les anomalies dans un fichier CSV lisible."""
    df_anomalies.to_csv(output_path, index=False)
    print(f"[INFO] Fichier exporté : {output_path}")
