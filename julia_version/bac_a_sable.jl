include("10_sparse_matrix.jl")

# Wouah ! Bibliothèque intéressante de julia ! pas besoin de libs ! Idée : Commencer tuto avec readdir
folder_path = "ead_files/"
all_files = [joinpath(folder_path, name) for name in readdir(folder_path) if endswith(name, ".xml")]
println(all_files)

matrix, vocab, files = build_sparse_matrix(all_files)
println("Matrice construite avec ", length(all_files), " dcouments")

println(matrix, vocab, files)