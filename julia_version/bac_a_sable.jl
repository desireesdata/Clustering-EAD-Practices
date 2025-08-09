include("10_sparse_matrix.jl")
include("20_similarity.jl")
include("30_spectral_clustering.jl")

# Wouah ! Bibliothèque intéressante de julia ! pas besoin de libs ! Idée : Commencer tuto avec readdir
folder_path = "ead_files/"
all_files = [joinpath(folder_path, name) for name in readdir(folder_path) if endswith(name, ".xml")]
println(all_files)

matrix, vocab, files = build_sparse_matrix(all_files)
println("Matrice construite avec ", length(all_files), " dcouments")
# println(matrix, vocab, files)

similarity, distance = compute_similarity_and_distance(matrix)
# print(similarity, distance)
print(spectral_clustering_dense(similarity, 3))
