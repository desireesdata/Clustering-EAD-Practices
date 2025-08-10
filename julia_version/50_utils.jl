using DataFrames
using CSV
using Dates
using FileWatching 

"""
    export_results_dataframe(df::DataFrame; out_dir::String = "results", filename_base::String = "clustering_results")

Exporte un DataFrame contenant les résultats du clustering dans un dossier dédié, aux formats CSV et XLSX.

Args:
    df: DataFrame contenant les résultats du clustering.
    out_dir: Dossier de sortie.
    filename_base: Nom de base du fichier, sans l'extension.
"""
function export_results_dataframe(
    df::DataFrame;
    out_dir::String = "results",
    filename_base::String = "clustering_results"
)::Nothing
    # Crée le dossier de sortie s'il n'existe pas
    # Il est préférable de vérifier si le dossier existe avant de le créer
    if !isdir(out_dir)
        mkdir(out_dir)
    end
    
    # Génère un horodatage pour un nom de fichier unique
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    # Crée les chemins de fichiers complets
    csv_path = joinpath(out_dir, "$(filename_base)_$(timestamp).csv")
    # xlsx_path = joinpath(out_dir, "$(filename_base)_$(timestamp).xlsx")

    # Exporte le DataFrame
    CSV.write(csv_path, df)
    # XLSX.writetable(xlsx_path, df)

    println("[INFO] Résultats exportés :\n  - $csv_path\n ")
    
    return nothing
end