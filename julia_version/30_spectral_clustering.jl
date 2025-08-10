using Clustering
using LinearAlgebra
using SparseArrays
using Arpack
using Distances
using Statistics
using Plots

"""
    find_eigengap_k(λs::Vector) -> Int

Trouve le k optimal en utilisant l'Eigengap Heuristic.
"""
function find_eigengap_k(λs::Vector)::Int
    # On calcule les "sauts" (gaps) entre les valeurs propres consécutives
    gaps = diff(real.(λs))
    
    # Le k optimal est l'index du plus grand saut
    k = argmax(gaps)
    
    println("[INFO] Eigengap Heuristic suggère k = $k (basé sur le plus grand saut entre valeurs propres)")
    return k
end

"""
    find_optimal_k(S::SparseMatrixCSC, D::Matrix; 
                   k_min::Int = 2, k_max::Int = 15, plot_results::Bool = true)

Teste différents k pour le clustering spectral et retourne celui qui maximise le score de silhouette.
"""
function find_optimal_k(
    S::SparseMatrixCSC, 
    D::Matrix;
    k_min::Int = 2, 
    k_max::Int = 30, 
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

        # --- VISUALISATION EIGENGAP ---
        println("[INFO] Affichage du graphique des valeurs propres pour l'Eigengap Heuristic...")
        eig_plot = plot(
            1:length(λs),
            real.(λs),
            marker=:circle,
            title="Eigengap Heuristic",
            xlabel="Index de la valeur propre",
            ylabel="Valeur propre (partie réelle)",
            legend=false,
            grid=true
        )
        display(eig_plot)
        savefig(eig_plot, "eigengap_plot.png")
        println("[INFO] Graphique Eigengap sauvegardé dans eigengap_plot.png")
        # --- FIN VISUALISATION ---

        # --- DÉTECTION AUTOMATIQUE EIGENGAP ---
        find_eigengap_k(λs)
        # --- FIN DÉTECTION --- 
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

            # k-means sur l'embedding
            res_kmeans = kmeans(embedding', k; init=:kmpp)
            labels = res_kmeans.assignments

            if length(unique(labels)) > 1
                # --- MODIFICATION POUR TEST ---
                # On utilise la matrice de distance d'origine (D) pour le calcul de la silhouette,
                # afin de mimer le comportement de la version Python.
                sil_scores = silhouettes(labels, D)
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

        # Exporter le graphique en PNG
        savefig(p, "silhouette_plot.png")
        println("[INFO] Graphique des scores de silhouette sauvegardé dans silhouette_plot.png")
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

"""
    cluster_with_eigengap(S::SparseMatrixCSC; k_max::Int = 30) -> Tuple{Vector{Int}, Int}

Effectue le clustering spectral en déterminant k via l'Eigengap Heuristic.
C'est la méthode recommandée, cohérente avec la version Python.

Args:
    S: Matrice de similarité creuse [n_docs, n_docs].
    k_max: Le nombre maximum de valeurs propres à calculer.

Returns:
    Tuple{Vector{Int}, Int}: Un tuple contenant les labels des clusters et le k utilisé.
"""
function cluster_with_eigengap(S::SparseMatrixCSC; k_max::Int = 30)::Tuple{Vector{Int}, Int}
    println("[INFO] Détermination de k par Eigengap Heuristic...")

    if size(S, 1) < 2
        @warn "Pas assez de documents pour le clustering."
        return Int[], -1
    end

    local L, λs
    try
        D_diag = vec(sum(S, dims=2))
        D_inv_sqrt = spdiagm(0 => 1 ./ sqrt.(D_diag .+ eps()))
        L = I - D_inv_sqrt * S * D_inv_sqrt
        
        n_evecs = min(k_max, size(S, 1) - 1)
        if n_evecs < 2
             @warn "Pas assez de documents pour le clustering (n_evecs < 2)."
             return Int[], -1
        end
        λs, _ = eigs(L, nev=n_evecs, which=:SR)
    catch e
        @warn "Erreur lors du calcul du Laplacien ou des vecteurs propres: $e"
        return Int[], -1
    end

    # Visualiser et trouver k
    println("[INFO] Affichage du graphique des valeurs propres pour l'Eigengap Heuristic...")
    eig_plot = plot(
        1:length(λs),
        real.(λs),
        marker=:circle,
        title="Eigengap Heuristic",
        xlabel="Index de la valeur propre",
        ylabel="Valeur propre (partie réelle)",
        legend=false,
        grid=true
    )
    display(eig_plot)
    savefig(eig_plot, "eigengap_plot.png")
    println("[INFO] Graphique Eigengap sauvegardé dans eigengap_plot.png")
    
    # --- DÉTECTION AUTOMATIQUE EIGENGAP ---
    k = find_eigengap_k(λs)

    # Lancer le clustering final avec le k trouvé
    println("[INFO] Lancement du clustering final avec k = $k...")
    labels = run_clustering(S, k)
    
    return labels, k
end
