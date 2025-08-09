using LinearAlgebra
using Clustering
using Distances
using Statistics
using Plots


function spectral_clustering_dense(S::Matrix{Float64}, k::Int)
    n = size(S, 1)
    # Laplacien normalisé symétrique : L = I - D^{-1/2} S D^{-1/2}
    D = diagm(0 => sum(S, dims=2)[:])
    D_inv_sqrt = Diagonal(1 ./ sqrt.(diag(D) .+ eps()))
    L = I - D_inv_sqrt * S * D_inv_sqrt

    # valeurs propres et vecteurs propres
    evals, evecs = eigen(Symmetric(L))
    embedding = evecs[:, 1:k]

    # clustering kmeans sur les vecteurs propres (chaque ligne = point)
    res = kmeans(embedding', k; init=:kmpp)
    return res.assignments
end

function find_optimal_k(
    S::Matrix{Float64},
    D::Matrix{Float64};
    k_min::Int = 2,
    k_max::Int = 15,
    plot_results::Bool = true
)::Tuple{Int, Vector{Float64}}

    ks = k_min:min(k_max, size(S, 1))
    scores = Float64[]

    println("[INFO] Recherche du meilleur k entre $k_min et $k_max...")

    for k in ks
        try
            labels = spectral_clustering_dense(S, k)
            if length(unique(labels)) > 1
                sil_scores = silhouettes(labels, [count(==(i), labels) for i in 1:k], D)
                push!(scores, mean(sil_scores))
            else
                push!(scores, -1.0)
            end
        catch e
            @warn "Clustering échoué pour k=$k : $e"
            push!(scores, -1.0)
        end
    end

    best_k = ks[argmax(scores)]

    if plot_results
        plot(ks, scores, marker=:circle, xlabel="k", ylabel="Score de silhouette",
             title="Silhouette score selon k", legend=false)
        vline!([best_k], color=:red, linestyle=:dash, label="k optimal = $best_k")
    end

    return best_k, scores
end
