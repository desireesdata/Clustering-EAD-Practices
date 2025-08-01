import pandas as pd
from pathlib import Path
from datetime import datetime

def export_results_dataframe(
    df: pd.DataFrame,
    out_dir: str = "results",
    filename_base: str = "clustering_results"
) -> None:
    """
    Exporte les résultats dans un dossier dédié en CSV et Excel.

    Args:
        df: DataFrame contenant les résultats
        out_dir: Dossier de sortie
        filename_base: Nom de base du fichier (sans extension)
    """
    Path(out_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = Path(out_dir) / f"{filename_base}_{timestamp}.csv"
    xlsx_path = Path(out_dir) / f"{filename_base}_{timestamp}.xlsx"

    df.to_csv(csv_path, index=False, encoding='utf-8')
    df.to_excel(xlsx_path, index=False)

    print(f"[INFO] Résultats exportés :\n  - {csv_path}\n  - {xlsx_path}")
