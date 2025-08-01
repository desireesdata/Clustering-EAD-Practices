# Clustering structurel de fichiers EAD

Outil d’analyse structurelle de fichiers XML encodés en EAD.  
Il vise à cartographier les pratiques d'encodage en identifiant des similarités structurelles entre documents, afin de préparer une normalisation ou une harmonisation.

---

## Fonctionnement

Le pipeline extrait la structure des balises XML (chemins XPath) de chaque fichier EAD, puis :

1. Vectorise chaque document selon les chemins présents
2. Calcule des matrices de similarité et de distance
3. Applique le Spectral Clustering
4. Affiche des statistiques sur chaque cluster (signature structurelle)
5. Exporte les résultats

---

## Utilisation

### Lancement

Depuis la racine du projet :

```bash
python3 main.py ./ead --k 4 #(Choix manuel de clusters)
python3 main.py ./ead  # choix automatique de cluster avec la silhouette
```

### Structure du projet 
```
.
├── main.py                   # Script principal
├── ead_clustering/           # Modules Python
│   ├── parser.py             # Extraction des chemins XML
│   ├── features.py           # Construction de la matrice CSR
│   ├── similarity.py         # Calcul des matrices de similarité et distance
│   ├── clustering.py         # Choix de k et Spectral Clustering
│   ├── analysis.py           # Signature des clusters
│   └── utils.py              # Export de résultats
├── ead/                      # Dossier contenant les fichiers EAD à analyser
└── results/                  # Résultats générés


ead_clustering/
├── __init__.py
├── xml_parser.py           ← extraction des chemins XML
├── features.py             ← vectorisation structurelle
├── similarity.py           ← matrices de similarité / distance
├── clustering.py           ← détection de k + clustering
├── analysis.py             ← résumés par cluster
├── main.py                 ← pipeline principal (script d’appel)
├── utils.py                ← outils divers
```