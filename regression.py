# =============================================================
#   RÉGRESSION LINÉAIRE - CAS COMPLET ET RÉALISTE
#   Dataset : Prix des appartements selon plusieurs features
#   Outils  : pandas, scikit-learn, matplotlib
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ──────────────────────────────────────────────
# 1. CRÉATION D'UN DATASET RÉALISTE (simulé)
#    En vrai tu ferais : df = pd.read_csv("appartements.csv")
# ──────────────────────────────────────────────

np.random.seed(42)
n = 200

data = {
    "surface":  np.random.randint(20, 150, n),
    "nb_pieces": np.random.randint(1, 6, n),
    "etage":    np.random.randint(0, 15, n),
    "region":   np.random.choice(["Paris", "Lyon", "Bordeaux", "Marseille"], n),
    "annee":    np.random.randint(1960, 2023, n),
    "prix":     None   # on va le calculer
}

df = pd.DataFrame(data)

# Prix simulé selon une formule + bruit (comme dans la vraie vie)
base = df["surface"] * 3500 + df["nb_pieces"] * 8000 + df["etage"] * 1200
region_bonus = df["region"].map({"Paris": 50000, "Lyon": 20000, "Bordeaux": 15000, "Marseille": 10000})
bruit = np.random.normal(0, 15000, n)
df["prix"] = (base + region_bonus + bruit).astype(int)

# ──────────────────────────────────────────────
# 2. PROBLÈME 1 : INTRODUCTION DE VALEURS MANQUANTES
#    Simule ce qu'on voit dans la vraie vie :
#    des cases vides dans le fichier CSV
# ──────────────────────────────────────────────

# On met des 0 dans surface (cas fréquent : mal rempli)
indices_surface = np.random.choice(df.index, 20, replace=False)
df.loc[indices_surface, "surface"] = 0

# On met des NaN dans etage
indices_etage = np.random.choice(df.index, 15, replace=False)
df.loc[indices_etage, "etage"] = np.nan

# On met des NaN dans prix
indices_prix = np.random.choice(df.index, 10, replace=False)
df.loc[indices_prix, "prix"] = np.nan

print("=" * 55)
print("ÉTAPE 1 — EXPLORATION DU DATASET")
print("=" * 55)
print(df.head(10))
print(f"\nShape : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print("\nInfos types et valeurs non-nulles :")
print(df.info())
print("\nStatistiques descriptives :")
print(df.describe())

# ──────────────────────────────────────────────
# 3. DÉTECTION DES PROBLÈMES
# ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("ÉTAPE 2 — DÉTECTION DES PROBLÈMES")
print("=" * 55)

# Compter les NaN
print("\nNombre de NaN par colonne :")
print(df.isnull().sum())

# Compter les 0 suspects dans surface (un appart de 0m² = erreur)
nb_zeros_surface = (df["surface"] == 0).sum()
print(f"\nSurfaces à 0 (valeurs aberrantes) : {nb_zeros_surface}")

# Compter les doublons
nb_doublons = df.duplicated().sum()
print(f"Lignes dupliquées : {nb_doublons}")

# ──────────────────────────────────────────────
# 4. NETTOYAGE DES DONNÉES
#    Règle d'or : on ne devine jamais, on remplace
#    par la médiane (plus robuste que la moyenne)
# ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("ÉTAPE 3 — NETTOYAGE")
print("=" * 55)

# ── 4a. Les 0 dans surface → on les traite comme des NaN
#   Pourquoi la médiane et pas la moyenne ?
#   → La moyenne est sensible aux valeurs extrêmes (outliers)
#   → La médiane est plus robuste : 50% des données sont en dessous
df["surface"] = df["surface"].replace(0, np.nan)
mediane_surface = df["surface"].median()
df["surface"].fillna(mediane_surface, inplace=True)
print(f"Surfaces à 0 remplacées par la médiane : {mediane_surface:.0f} m²")

# ── 4b. NaN dans etage → médiane
mediane_etage = df["etage"].median()
df["etage"].fillna(mediane_etage, inplace=True)
print(f"Étages manquants remplacés par la médiane : {mediane_etage:.0f}")

# ── 4c. NaN dans prix → on supprime ces lignes
#   Pourquoi ? C'est notre cible (y), on ne peut pas la deviner
df.dropna(subset=["prix"], inplace=True)
print(f"Lignes avec prix=NaN supprimées. Reste : {len(df)} lignes")

# ── 4d. Supprimer les doublons
df.drop_duplicates(inplace=True)

# ── 4e. Vérification finale
print("\nVérification après nettoyage :")
print(df.isnull().sum())

# ──────────────────────────────────────────────
# 5. PROBLÈME 2 : VARIABLES CATÉGORIELLES
#    "region" contient du texte → le modèle ML
#    ne comprend que des nombres !
#    Solution : LabelEncoder ou get_dummies (One-Hot)
# ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("ÉTAPE 4 — ENCODAGE DES VARIABLES CATÉGORIELLES")
print("=" * 55)

# Option A : LabelEncoder (simple, mais crée un ordre implicite)
# Paris→0, Lyon→1 ... le modèle croit que Lyon > Paris en valeur
# encoder = LabelEncoder()
# df["region_encoded"] = encoder.fit_transform(df["region"])

# Option B : get_dummies / One-Hot Encoding (meilleure pratique)
# Crée une colonne par catégorie : region_Paris, region_Lyon, etc.
df = pd.get_dummies(df, columns=["region"], drop_first=True)
# drop_first=True évite la multicolinéarité (dummy variable trap)

print("Colonnes après encodage :")
print(df.columns.tolist())
print(df.head(3))

# ──────────────────────────────────────────────
# 6. SÉPARATION X (features) et y (target)
# ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("ÉTAPE 5 — PRÉPARATION X et y")
print("=" * 55)

# y = ce qu'on veut prédire (le prix)
y = df["prix"]

# X = toutes les features (tout sauf le prix)
X = df.drop(columns=["prix"])

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Features utilisées : {X.columns.tolist()}")

# ──────────────────────────────────────────────
# 7. PROBLÈME 3 : ÉCHELLES DIFFÉRENTES
#    surface (20-150), etage (0-14), annee (1960-2023)
#    → le modèle va favoriser les grandes valeurs
#    Solution : StandardScaler (mean=0, std=1)
# ──────────────────────────────────────────────

# Note : pour la régression linéaire simple c'est optionnel
# mais c'est une bonne habitude à prendre !

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nAprès StandardScaler (extrait) :")
print(X_scaled.describe().round(2))

# ──────────────────────────────────────────────
# 8. TRAIN / TEST SPLIT
#    On coupe le dataset : 80% entraînement, 20% test
#    Le modèle apprend sur train, on évalue sur test
#    (données qu'il n'a jamais vues → évalue la généralisation)
# ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 20% pour le test
    random_state=42     # pour reproduire les mêmes résultats
)

print(f"\nTrain : {X_train.shape[0]} exemples")
print(f"Test  : {X_test.shape[0]} exemples")

# ──────────────────────────────────────────────
# 9. ENTRAÎNEMENT DU MODÈLE
# ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("ÉTAPE 6 — ENTRAÎNEMENT")
print("=" * 55)

model = LinearRegression()
model.fit(X_train, y_train)  # le modèle apprend ici

print(f"Intercept (b0) : {model.intercept_:,.0f} €")
print("\nCoefficients appris :")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:<30} : {coef:>10,.0f}")

# ──────────────────────────────────────────────
# 10. ÉVALUATION DU MODÈLE
# ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("ÉTAPE 7 — ÉVALUATION")
print("=" * 55)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# MSE : Erreur Quadratique Moyenne (sensible aux gros écarts)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test  = mean_squared_error(y_test,  y_pred_test)

# RMSE : Racine du MSE (dans la même unité que y, ici €)
rmse_train = np.sqrt(mse_train)
rmse_test  = np.sqrt(mse_test)

# MAE : Erreur Absolue Moyenne (robuste aux outliers)
mae_test = mean_absolute_error(y_test, y_pred_test)

# R² : coefficient de détermination
# 1.0 = parfait | 0.0 = pas mieux que la moyenne | <0 = très mauvais
r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test,  y_pred_test)

print(f"\n{'Métrique':<25} {'Train':>12} {'Test':>12}")
print("-" * 50)
print(f"{'RMSE (€)':<25} {rmse_train:>12,.0f} {rmse_test:>12,.0f}")
print(f"{'MAE (€)':<25} {'':>12} {mae_test:>12,.0f}")
print(f"{'R²':<25} {r2_train:>12.4f} {r2_test:>12.4f}")

# ── Diagnostic overfitting / underfitting
print("\n── Diagnostic ──")
if r2_test < 0.5:
    print("⚠️  Underfitting : le modèle est trop simple")
elif r2_train - r2_test > 0.15:
    print("⚠️  Overfitting : le modèle a trop mémorisé le train")
else:
    print("✅  Modèle correct : pas de sur/sous-apprentissage détecté")

# ──────────────────────────────────────────────
# 11. VISUALISATION
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Analyse du modèle de régression - Prix Appartements", fontsize=14, fontweight='bold')

# ── Graphique 1 : Valeurs réelles vs prédites
axes[0].scatter(y_test, y_pred_test, alpha=0.5, color='steelblue', s=30, label='Prédictions')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
axes[0].set_xlabel("Prix réel (€)")
axes[0].set_ylabel("Prix prédit (€)")
axes[0].set_title("Réel vs Prédit")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ── Graphique 2 : Distribution des résidus
residus = y_test.values - y_pred_test
axes[1].hist(residus, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Résidu = 0')
axes[1].set_xlabel("Résidu (€)")
axes[1].set_ylabel("Fréquence")
axes[1].set_title("Distribution des résidus")
axes[1].legend()
axes[1].grid(alpha=0.3)
# Les résidus bien distribués autour de 0 = modèle non biaisé

# ── Graphique 3 : Prix vs Surface (relation principale)
axes[2].scatter(df["surface"], df["prix"] / 1000,
                alpha=0.4, color='steelblue', s=20, label='Données')
# Droite de tendance sur la feature principale
surface_range = np.linspace(df["surface"].min(), df["surface"].max(), 100)
prix_moyen = df["prix"].mean()
# Régression simple surface→prix pour l'affichage
reg_simple = LinearRegression()
reg_simple.fit(df[["surface"]], df["prix"])
prix_line = reg_simple.predict(surface_range.reshape(-1, 1)) / 1000
axes[2].plot(surface_range, prix_line, 'r-', linewidth=2, label='Tendance linéaire')
axes[2].set_xlabel("Surface (m²)")
axes[2].set_ylabel("Prix (k€)")
axes[2].set_title("Prix vs Surface")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/regression_analyse.png", dpi=150, bbox_inches='tight')
print("\nGraphiques sauvegardés : regression_analyse.png")

# ──────────────────────────────────────────────
# 12. PRÉDICTION SUR UN NOUVEAU CAS
# ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("ÉTAPE 8 — PRÉDICTION NOUVEAU CAS")
print("=" * 55)

# Appartement à prédire : 75m², 3 pièces, 5ème étage, Paris, 2005
nouveau_cas = pd.DataFrame({
    "surface":         [75],
    "nb_pieces":       [3],
    "etage":           [5],
    "annee":           [2005],
    "region_Lyon":     [0],    # colonnes créées par get_dummies
    "region_Marseille":[0],
    "region_Paris":    [1],    # Paris = 1
})

# IMPORTANT : on utilise le MÊME scaler qu'à l'entraînement
nouveau_scaled = scaler.transform(nouveau_cas)
prix_predit = model.predict(nouveau_scaled)[0]

print(f"\nAppartement : 75m², 3 pièces, étage 5, Paris, 2005")
print(f"Prix prédit  : {prix_predit:,.0f} €  (~{prix_predit/1000:.0f} k€)")

print("\n" + "=" * 55)
print("FIN DU PIPELINE")
print("=" * 55)