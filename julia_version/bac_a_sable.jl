include("10_sparse_matrix.jl")
include("20_similarity.jl")
include("30_spectral_clustering.jl")
include("40_analysis.jl")

# Wouah ! Bibliothèque intéressante de julia ! pas besoin de libs ! Idée : Commencer tuto avec readdir
folder_path = "ead_files/"
all_files = [joinpath(folder_path, name) for name in readdir(folder_path) if endswith(name, ".xml")]
println(all_files)

matrix, vocab, files = build_sparse_matrix(all_files)
println("Matrice construite avec ", length(all_files), " dcouments")

# -----------------
# Étape de clustering
# -----------------

similarity = build_sparse_similarity_matrix(matrix)
D = pairwise(Euclidean(), matrix', dims=2)
best_k, scores = find_optimal_k(similarity, D)
println("Le nombre de clusters optimal est : ", best_k)

# Exécute le clustering final avec le k optimal
final_labels = run_clustering(similarity, best_k)

# -----------------
# Étape d'analyse
# -----------------

# 'vocab' est de la forme 'chemin_balise' => index
# On a besoin de l'inverse : 'index' => 'chemin_balise'
index_to_path = Dict(v => k for (k, v) in vocab)

# Utilisation de la nouvelle fonction pour résumer les clusters
df_results = summarize_clusters(
    final_labels,
    matrix,
    vocab, # <-- Cet argument doit être un Dict{Int, String}
    files
)

# Affiche le DataFrame final
println("\n--- Résultats du clustering sous forme de DataFrame ---")
println(df_results)