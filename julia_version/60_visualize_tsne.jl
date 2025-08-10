# 60_visualize_tsne.jl
# Script pour visualiser les clusters avec t-SNE

# --- Installation des dépendances ---
using Pkg
# --- Inclusions des scripts précédents ---
# Permet de réutiliser les fonctions déjà définies
include("10_sparse_matrix.jl")
include("20_similarity.jl")
include("30_spectral_clustering.jl")

# --- Importations ---
using TSne
using Plots
gr()
using SparseArrays
using Random

"""
    visualize_with_tsne(feature_matrix, labels; 
                        output_filename="tsne_clustering_visualization.png",
                        perplexity=30.0,
                        max_docs=1000)

Réduit la dimension de la matrice de features avec t-SNE et génère une visualisation des clusters.

Args:
    feature_matrix: Matrice creuse des features (documents x features).
    labels: Vecteur des labels de cluster pour chaque document.
    output_filename: Nom du fichier image de sortie.
    perplexity: Paramètre de perplexité pour t-SNE. Typiquement entre 5 et 50.
    max_docs: Nombre maximum de documents à utiliser pour t-SNE pour garder un temps de calcul raisonnable.
"""
function visualize_with_tsne(
    feature_matrix::SparseMatrixCSC,
    labels::Vector{Int};
    output_filename::String = "tsne_clustering_visualization.png",
    perplexity::Float64 = 30.0,
    max_docs::Int = 1000
)
    println("[INFO] Lancement de la visualisation t-SNE...")
    n_docs = size(feature_matrix, 1)

    local matrix_for_tsne, labels_for_tsne
    if n_docs > max_docs
        @warn "Le nombre de documents ($n_docs) est élevé. t-SNE sera exécuté sur un sous-ensemble aléatoire de $max_docs documents."
        Random.seed!(42) # Pour la reproductibilité
        subset_indices = sort(sample(1:n_docs, max_docs, replace=false))

        matrix_for_tsne = Matrix(feature_matrix[subset_indices, :])'
        labels_for_tsne = labels[subset_indices]
    else
        matrix_for_tsne = Matrix(feature_matrix)'
        labels_for_tsne = labels
    end

    # Assurez-vous que matrix_for_tsne est bien de type Float64
    matrix_for_tsne = Float64.(matrix_for_tsne)

    println("[INFO] Calcul de la réduction de dimension t-SNE (cela peut prendre un moment)...")
    
    # Appel de tsne avec les paramètres stables (reduce_dims=50, perplexity en Float64)
    tsne_result = tsne(matrix_for_tsne, 2, 50, 2000, perplexity)

    println("[INFO] Création du graphique de visualisation...")
    println("[DEBUG] Valeurs uniques dans labels_for_tsne : ", unique(labels_for_tsne))

    # Création du nuage de points
    unique_labels = sort(unique(labels_for_tsne))
    p = scatter(
        tsne_result[:, 1],
        tsne_result[:, 2],
        group = string.(labels_for_tsne), # Conversion en texte pour forcer la coloration
        marker = :circle,
        title = "Visualisation t-SNE des clusters EAD",
        xlabel = "Dimension t-SNE 1",
        ylabel = "Dimension t-SNE 2",
        legend = :outertopright,
        palette = :Set1, # Changement de palette pour un meilleur contraste
        markersize = 5,
        markerstrokewidth = 0.5
    )

    # Sauvegarde du graphique
    savefig(p, output_filename)
    println("[INFO] Graphique de visualisation sauvegardé dans : $output_filename")

    return p
end


"""
    run_tsne_experiment()

Fonction principale pour lancer l'expérimentation de bout en bout :
1. Chargement des données XML.
2. Création de la matrice de features.
3. Exécution du clustering spectral.
4. Lancement de la visualisation t-SNE sur les résultats.
"""
function run_tsne_experiment(k_manual::Union{Int, Nothing} = nothing) # Ajout du paramètre k_manual
    println("--- Lancement de l'expérimentation t-SNE ---")

    # 1. Charger les données
    folder_path = "../ead_exemples/"
    all_files = [joinpath(folder_path, name) for name in readdir(folder_path) if endswith(name, ".xml")]
    matrix, _, files = build_sparse_matrix(all_files)
    println("[INFO] Matrice construite avec ", length(files), " documents.")

    # 2. Exécuter le clustering pour obtenir les labels
    similarity = build_sparse_similarity_matrix(matrix)

    local labels, k_used # Déclaration des variables locales

    if k_manual !== nothing # Si k_manual est fourni, l'utiliser directement
        k_used = k_manual
        println("[INFO] Clustering avec k manuel = ", k_used)
        labels = run_clustering(similarity, k_used)
    else # Sinon, utiliser l'Eigengap Heuristic
        labels, k_used = cluster_with_eigengap(similarity)
        if k_used == -1
            println("[ERREUR] Le clustering a échoué. Arrêt de l'expérimentation t-SNE.")
            return
        end
        println("[INFO] Clustering terminé. Nombre de clusters trouvé : ", k_used)
    end

    # 3. Lancer la visualisation
    visualize_with_tsne(matrix, labels)

    println("--- Expérimentation t-SNE terminée ---")
end

# Pour lancer l'expérimentation, il suffit d'exécuter ce fichier.
# La ligne ci-dessous est commentée par défaut pour permettre l'inclusion
# du fichier sans exécution automatique.
#
# Décommentez la ligne suivante si vous voulez exécuter le script directement.
run_tsne_experiment() # Appel par défaut (utilise l'Eigengap)
# run_tsne_experiment(3) # Exemple d'appel avec k manuel
