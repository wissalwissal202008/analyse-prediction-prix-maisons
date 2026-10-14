# Analyse et prédiction des prix des maisons

Projet pédagogique en Python et Streamlit autour de l’analyse et de la prédiction de prix immobiliers.

## Fonctionnalités

- Exploration d’un petit jeu de données fictif de maisons.
- Statistiques descriptives, variance, écart-type et histogramme des prix.
- Estimations par régression linéaire et arbre de décision.
- Comparaison des prix réels et prédits avec MAE et R².
- Interface Streamlit en français avec estimation interactive.

Les prix sont exprimés en milliers de dirhams (kDH). Les données, y compris les distances au centre-ville, sont fictives et ne permettent pas une estimation immobilière réelle.

## Installation et lancement

Depuis le dossier du projet, sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

L’application est ensuite disponible à l’adresse indiquée par Streamlit, généralement <http://localhost:8501>.

Pour lancer l’analyse en console :

```powershell
python analyse.py
```

## Fichiers

- `app.py` : application Streamlit et modèles de prédiction.
- `analyse.py` : statistiques descriptives et graphique exploratoire.
- `mini_projet_2_analyse_prediction_prix_maisons.txt` : sujet pédagogique d’origine.
- `requirements.txt` : dépendances Python.
