using Distances
using SparseArrays

function compute_similarity_and_distance(matrix::SparseMatrixCSC)::Tuple{Matrix, Matrix}
    print("INFO : calcul de la matrice de similarité DENSE (!) et de distance")
    distance_matrix = pairwise(CosineDist(), matrix)
    similarity_matrix = 1.0 .- distance_matrix
    return similarity_matrix, distance_matrix
end

function build_sparse_similarity_matrix(sparse_features::SparseMatrixCSC)::SparseMatrixCSC
    println("INFO : Calcul de la matrice de similarité creuse...")
    
    # Normalisation de chaque ligne de la matrice creuse.
    # On calcule la norme L2 de chaque vecteur-document.
    norms = sqrt.(sum(sparse_features .^ 2, dims=2))
    
    # On crée une matrice diagonale avec l'inverse des normes.
    # Ceci évite de diviser directement, ce qui est plus performant.
    inv_norms = sparse(Diagonal(vec(1 ./ norms)))
    
    # La matrice normalisée a des vecteurs-lignes de norme 1.
    # On fait le produit matriciel de la matrice normalisée et de sa transposée.
    # Le résultat est la matrice de similarité cosinus creuse.
    sparse_normalized = inv_norms * sparse_features
    sparse_similarity = sparse_normalized * sparse_normalized'
    
    return sparse_similarity
end