# Clustering structurel de fichiers EAD

Outil d’analyse structurelle de fichiers XML encodés en EAD.  
Il vise à cartographier les pratiques d'encodage en identifiant des similarités structurelles entre documents, afin de préparer une normalisation ou une harmonisation.

- Documentation : [https://desireesdata.fr/Clustering-EAD-Practices/](https://desireesdata.fr/Clustering-EAD-Practices/)

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
python3 main.py ./ead_exemples --k 4 #(Choix manuel de clusters)
python3 main.py ./ead_exemples  # choix automatique de cluster avec la silhouette (recommandé)
```

### Structure du projet 
```

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