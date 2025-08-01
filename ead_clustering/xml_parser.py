import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import List

NAMESPACE_REGEX = re.compile(r'\{[^}]+\}')

def extract_paths_from_element(element, current_path="") -> List[str]:
    """Récursivement, extrait tous les chemins à partir d’un élément XML."""
    clean_tag = NAMESPACE_REGEX.sub('', element.tag)
    new_path = f"{current_path}/{clean_tag}" if current_path else clean_tag
    paths = [new_path]
    for child in element:
        paths.extend(extract_paths_from_element(child, new_path))
    return paths

def extract_xml_paths(xml_file: Path) -> List[str]:
    """
    Parse un fichier XML et retourne une liste des chemins de balises (structure).
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return extract_paths_from_element(root)
    except ET.ParseError as e:
        print(f"[ERREUR] Parsing XML échoué ({xml_file.name}) : {e}")
        return []
