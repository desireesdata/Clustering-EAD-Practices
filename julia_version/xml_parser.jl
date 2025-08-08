using LightXML

function read_xml(xml_file::String)::Vector{String}
    # Bon exemple pour apprendre à lire un XML !
    tree = nothing
    paths = String[]

    try
        tree = parse_file(xml_file)
        root_element = root(tree)
        println(root_element)
        free(tree)
    catch e
        println("Erreur de parsin XML / ($xml_file) : ", e)
    finally
        if tree !== nothing
            free(tree)
        end
    end
    return paths
end

function extract_paths_from_element(element::XMLElement, current_path=""::String)::Vector{String}
    clean_tag = replace(name(element), r"\{[^}]+\}" => "")
    new_path = isempty(current_path) ? clean_tag : current_path * "/" * clean_tag
    paths = Vector{String}()
    push!(paths, new_path)
    for child in child_elements(element)
        append!(paths, extract_paths_from_element(child, new_path))
    end
    return paths
end

function extract_xml_paths(xml_file::String)::Vector{String}
    tree = nothing
    paths = String[]

    try
        tree = parse_file(xml_file)
        root_element = root(tree)
        append!(paths, extract_paths_from_element(root_element))
    catch e
        println("Erreur de parsin XML / ($xml_file) : ", e)
    finally
        if tree !== nothing
            free(tree)
        end
    end
    return paths
end

paths = extract_xml_paths("ead_files/bnf.xml")
println(paths)