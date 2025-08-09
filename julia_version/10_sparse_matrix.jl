# Vectorisation matrices creuses
include("00_xml_parser.jl")

using SparseArrays
using DataStructures

function build_sparse_matrix(xml_files::Vector{String})::Tuple{SparseMatrixCSC, Dict{Int, String}, Vector{String}}
    # L'idée générale est d'avoir un seul exemplaire de chaque chemin XPATH existant

    # Permet d'associer un chemin xpath unique à un identifiant numérique
    paths_vocabulary = Dict{String, Int64}() 
    # Permet de compter la fréquence d'apparition des chemins XPATH pour chaque ligne/document 
    path_counts = Vector{Dict{Int64, Int64}}()

    for (doc_index, file_path) in enumerate(xml_files)
        # Récupère tous les chemins de chaque doc EAD (contenu dans un vec) 
        paths = extract_xml_paths(file_path)   
        # Tient un dictionnaire de comptes pour ce document
        counts = Dict{Int64, Int64}()

        for path in paths    
            # Si, dans le vocabulaire (dico), le chemin n'est pas déjà une clé...          
            if !haskey(paths_vocabulary, path)
                # On crée un index différent de zéro et qui s'incrèmente au regard de la valeur max du Vocabulaire
                new_index = length(paths_vocabulary) + 1
                # Et enfin ! on ajoute à l'indice courant la nouvelle référence de l'index
                paths_vocabulary[path] = new_index
            end

            path_index = paths_vocabulary[path]

            if !haskey(counts, path_index)
                counts[path_index] = 1
            else
                counts[path_index] += 1
            end
        end
        push!(path_counts, counts)
    end
    rows = Int[]
    cols = Int[]
    vals = Int[]

    n_documents = length(xml_files)
    n_features = length(paths_vocabulary)

    for (doc_index, counts) in enumerate(path_counts)
        for (path_index, count) in counts
            push!(rows, doc_index)
            push!(cols, path_index)
            push!(vals, count)
        end
    end

    matrix = sparse(rows, cols, vals, n_documents, n_features)
    index_to_path = Dict{Int, String}(value => key for (key, value) in paths_vocabulary)
    return matrix, index_to_path, xml_files
end

# build_sparse_matrix(["ead_files/bnf.xml"])