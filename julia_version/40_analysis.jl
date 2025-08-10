using DataFrames
using SparseArrays
using LinearAlgebra
using Statistics

"""
    summarize_clusters(labels::Vector{Int}, feature_matrix::SparseMatrixCSC,
                        index_to_path::Dict{Int, String}, xml_files::Vector{String};
                        top_n::Int = 10)::DataFrame

Résume les clusters structurels en listant les fichiers et chemins dominants.

Args:
    labels: Un vecteur d'entiers représentant le label de cluster de chaque document.
    feature_matrix: Une matrice creuse (SparseMatrixCSC) des features.
    index_to_path: Un dictionnaire mappant l'index de la feature à son chemin XML.
    xml_files: Un vecteur de chaînes de caractères avec les chemins de fichiers XML.
    top_n: Le nombre de chemins les plus fréquents à afficher par cluster.

Returns:
    Un DataFrame avec les informations de chaque document et son cluster.
"""
function summarize_clusters(
    labels::Vector{Int},
    feature_matrix::SparseMatrixCSC,
    index_to_path::Dict{Int, String},
    xml_files::Vector{String};
    top_n::Int = 10
)::DataFrame
    # Création d'un DataFrame à partir des labels et des noms de fichiers
    df = DataFrame(filepath = xml_files, cluster = labels)

    n_clusters = length(unique(labels))
    println("[INFO] Résumé de $n_clusters clusters...\n")

    # Itération sur chaque cluster pour en faire la synthèse
    for cluster_id in sort(unique(labels))
        # Sélection des indices des documents appartenant au cluster actuel
        indices = findall(x -> x == cluster_id, df.cluster)
        n_docs = length(indices)

        println("\n--- Cluster $cluster_id ($n_docs documents) ---")
        println("Fichiers :")
        
        # Affichage des 5 premiers fichiers du cluster
        for path in first(df[indices, :].filepath, 5)
            println("  - $path")
        end

        if n_docs == 0
            println("Aucun document.")
            continue
        end

        # Extraction de la sous-matrice correspondant au cluster
        cluster_matrix = feature_matrix[indices, :]
        
        # Somme des colonnes pour obtenir la fréquence de chaque balise
        summed_counts = vec(sum(cluster_matrix, dims=1))
        
        # Tri pour trouver les indices des balises les plus fréquentes
        sorted_indices = sortperm(summed_counts, rev=true)
        top_indices = sorted_indices[1:min(top_n, length(sorted_indices))]

        println("\nBalises les plus fréquentes :")
        for i in top_indices
            chemin = get(index_to_path, i, "<inconnu $i>")
            count = Int(summed_counts[i])
            println("  - $chemin : $count")
        end
    end

    return df
end