using Distances
using SparseArrays

function compute_similarity_and_distance(matrix::SparseMatrixCSC)::Tuple{Matrix, Matrix}
    print("INFO : calcul des matrics de similarité et de distance")
    distance_matrix = pairwise(CosineDist(), matrix)
    similarity_matrix = 1.0 .- distance_matrix
    return similarity_matrix, distance_matrix
end

