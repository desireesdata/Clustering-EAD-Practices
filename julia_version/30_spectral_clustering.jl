# Fichier : 30_spectral_clustering.jl

using Clustering
using LinearAlgebra
using SparseArrays
using Arpack
using Distances
using Statistics
using Plots

"""
    find_optimal_k(S::SparseMatrixCSC, D::Matrix; 
                   k_min::Int = 2, k_max::Int = 15, plot_results::Bool = true)

Teste différents k pour le clustering spectral et retourne celui qui maximise le score de silhouette.

Args:
    S: Matrice de similarité creuse [n_docs, n_docs].
    D: Matrice de distance dense [n_docs, n_docs].
    k_min: La valeur minimale de k à tester.
    k_max: La valeur maximale de k à tester.
    plot_results: Booléen pour afficher ou non un graphique.

Returns:
    Tuple{Int, Vector{Float64}}: Un tuple contenant le k optimal et la liste des scores de silhouette.
"""
# Fichier : 30_spectral_clustering.jl
# ... (Les using restent les mêmes)

function find_optimal_k(
    S::SparseMatrixCSC, 
    D::Matrix;
    k_min::Int = 2, 
    k_max::Int = 15, 
    plot_results::Bool = true
)::Tuple{Int, Vector{Float64}}
    
    ks = k_min:min(k_max, size(S, 1))
    scores = Float64[]

    println("[INFO] Recherche du meilleur k entre $k_min et $k_max pour le clustering...")

    if size(S, 1) < k_min
        @warn "Pas assez de documents pour le clustering."
        return -1, scores
    end
    
    local L, λs, X
    
    try
        D_diag = vec(sum(S, dims=2))
        D_inv_sqrt = spdiagm(0 => 1 ./ sqrt.(D_diag .+ eps()))
        L = I - D_inv_sqrt * S * D_inv_sqrt
        
        n_evecs = min(k_max, size(S, 1) - 1)
        if n_evecs < k_min
            @warn "Le nombre de vecteurs propres disponibles ($n_evecs) est inférieur à k_min ($k_min)."
            return -1, scores
        end
        λs, X = eigs(L, nev=n_evecs, which=:SR) 
        X = real(X) 
    catch e
        @warn "Erreur lors du calcul du Laplacien ou des vecteurs propres: $e"
        return -1, scores
    end
    
    for k in ks
        try
            if k > size(X, 2)
                 println("Stop : le nombre de clusters (k=$k) dépasse le nombre de vecteurs propres disponibles.")
                 break
            end
            
            embedding = X[:, 1:k]
            
            res_kmeans = kmeans(embedding', k; init=:kmpp)
            labels = res_kmeans.assignments
            
            if length(unique(labels)) > 1
                # Ligne corrigée : utilisation de la matrice d'embedding pour le score de silhouette
                # L'embedding est une matrice (n_points x k), et la fonction l'attend en premier.
                sil_scores = silhouettes(embedding', labels)
                push!(scores, mean(sil_scores))
            else
                push!(scores, -1.0)
            end
        catch e
            println("Erreur dans la boucle pour k=$k: ", e)
            push!(scores, -1.0)
        end
    end
    
    if isempty(scores)
        @warn "Aucun score de silhouette n'a pu être calculé."
        return -1, scores
    end
    
    best_k = ks[argmax(scores)]

    if plot_results
        p = plot(ks[1:length(scores)], scores, marker=:circle, xlabel="k", ylabel="Score de silhouette",
                 title="Silhouette score selon k", legend=false)
        vline!([best_k], color=:red, linestyle=:dash, label="k optimal = $best_k")
        display(p)
    end

    return best_k, scores
end

"""
    run_clustering(S::SparseMatrixCSC, k::Int) -> Vector{Int}

Applique le clustering spectral final avec les paramètres optimaux.
"""
function run_clustering(S::SparseMatrixCSC, k::Int)::Vector{Int}
    if k < 2
        error("Le nombre de clusters doit être > 1")
    end
    
    # Calcul du Laplacien
    D_diag = vec(sum(S, dims=2))
    D_inv_sqrt = spdiagm(0 => 1 ./ sqrt.(D_diag .+ eps()))
    L = I - D_inv_sqrt * S * D_inv_sqrt
    
    # Calcul de l'embedding avec `eigs` d'Arpack.jl
    λs, X = eigs(L, nev=k, which=:SR)
    embedding = real(X)
    
    # K-means sur l'embedding
    res_kmeans = kmeans(embedding', k; init=:kmpp)
    
    return res_kmeans.assignments
end