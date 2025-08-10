using Pkg

include("10_sparse_matrix.jl")
include("20_similarity.jl")
include("30_spectral_clustering.jl")
include("40_analysis.jl")
include("50_utils.jl") # <-- Ajout de cette ligne
    
function main()
    folder_path = "../ead_exemples/"
    all_files = [joinpath(folder_path, name) for name in readdir(folder_path) if endswith(name, ".xml")]

    println("Fichiers à traiter : ", all_files)

    matrix, vocab, files = build_sparse_matrix(all_files)
    println("Matrice construite avec ", length(all_files), " documents")

    similarity = build_sparse_similarity_matrix(matrix)
    final_labels, k_found = cluster_with_eigengap(similarity)
    println("Clustering terminé. Nombre de clusters trouvé : ", k_found)


    index_to_path = Dict(v => k for (k, v) in vocab)

    df_results = summarize_clusters(final_labels, matrix, vocab, files)
    println("\n--- Résultats du clustering sous forme de DataFrame ---")
    # println(df_results)
    
    # Ajout de l'exportation des résultats
    export_results_dataframe(df_results)
end

# Exécuter la fonction principale
main()