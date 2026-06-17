import numpy as np
import pandas as pd

# Fixer la graine pour avoir toujours les mêmes données
np.random.seed(42)
n = 10

# 1. Votre dictionnaire initial - vous modifier ici les champs plustard pour d'autres exemples
data = {
    "surface":  np.random.randint(20, 150, n),
    "nb_pieces": np.random.randint(1, 6, n),
    "etage":    np.random.randint(0, 15, n),
    "region":   np.random.choice(["Paris", "Lyon", "Bordeaux", "Marseille"], n),
    "annee":    np.random.randint(1960, 2023, n),
    "prix":     None   
}

# 2. Création du DataFrame initial
df = pd.DataFrame(data)

# 3. Simulation d'un prix réaliste (Prix de base au m² selon la région)
prix_m2_region = {
    "Paris": 10000,
    "Lyon": 5000,
    "Bordeaux": 4500,
    "Marseille": 3500
}

# Calcul du prix : (surface * prix_m2_de_la_region) + un petit bruit aléatoire
df["prix"] = df.apply(lambda row: int((row["surface"] * prix_m2_region[row["region"]]) * np.random.uniform(0.9, 1.1)), axis=1)

# 4. Sauvegarde en fichier CSV
df.to_csv("dataset_immobilier.csv", index=False)

print("Fichier 'dataset_immobilier.csv' créé avec succès avec 10 lignes !")
print(df)
