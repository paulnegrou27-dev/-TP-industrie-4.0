#!/usr/bin/env python3
"""TP1 - Préparer la télémétrie de l'atelier (vraies données CSV).
Placez ce script dans le même dossier que les CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

# 0. Chargement : accepte atelier_connecte.csv, atelier_connecte__1_.csv, "atelier_connecte (1).csv", etc.
D = Path(__file__).resolve().parent
trouves = sorted(D.glob("atelier_connecte*.csv"))
if not trouves:
    raise FileNotFoundError(
        f"Aucun fichier atelier_connecte*.csv dans {D}\n"
        f"Fichiers présents : {[p.name for p in D.iterdir()]}\n"
        "-> copiez le CSV dans ce dossier (à côté du script).")
CSV = trouves[0]
print("Fichier lu :", CSV.name)
df = pd.read_csv(CSV, parse_dates=["horodatage"]).sort_values(["machine_id", "horodatage"]).reset_index(drop=True)

# 1. Explorer : distinguer mesures, catégories, alarmes, étiquette
mesures = ["temperature", "vibration", "courant", "cadence"]
spectre = ["vibration_pic", "vib_f1", "vib_f2", "vib_f3", "vib_f4"]   # dérivées de la vibration
alarmes = [c for c in df.columns if c.startswith("alarme_")]          # seuils appliqués aux mesures
categories = ["machine_id"]                                           # nominale
etiquette = "defaut"                                                  # jamais en entrée du modèle
# 'donnees_synthetiques' est constante : ignorée

df.info()
print("Valeurs manquantes :\n", df.isna().sum()[mesures])
print("\nMoyennes par machine :\n", df.groupby("machine_id")[mesures].mean().round(2))
print("\nHorodatages par machine (bout à bout, d'où le découpage par machine) :\n",
      df.groupby("machine_id")["horodatage"].agg(["min", "max"]))

# 2. Découpage chronologique PAR MACHINE : 70 % premiers relevés = train, 30 % derniers = test
df["rang"] = df.groupby("machine_id").cumcount()
coupure = int(0.7 * df.groupby("machine_id").size().min())
train = df[df["rang"] < coupure].copy()
test = df[df["rang"] >= coupure].copy()

# 3. Imputer par la médiane (robuste aux pics), calculée sur le train seulement
imputer = SimpleImputer(strategy="median").fit(train[mesures])
X_train = imputer.transform(train[mesures])
X_test = imputer.transform(test[mesures])

# 4. Mettre à l'échelle : fit sur le train, transform sur le train ET le test
std = StandardScaler().fit(X_train)
X_train_std, X_test_std = std.transform(X_train), std.transform(X_test)

mm = MinMaxScaler().fit(X_train)
X_train_01, X_test_01 = mm.transform(X_train), mm.transform(X_test)

# 5. Encoder les catégories : nominal (one-hot) ...
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
cat_train = onehot.fit_transform(train[categories])
cat_test = onehot.transform(test[categories])

# ... et ordinal (ordre explicite). Le CSV de télémétrie n'a pas de colonne ordinale ;
# on l'illustre avec la priorité des tickets de maintenance (basse < moyenne < haute).
interv_path = next(iter(sorted(D.glob("interventions_maintenance*.csv"))), None)
if interv_path is not None:
    interv = pd.read_csv(interv_path)
    ordinal = OrdinalEncoder(categories=[["basse", "moyenne", "haute"]])
    prio = ordinal.fit_transform(interv[["priorite"]])
    print("\nExemple priorité encodée :", interv["priorite"].head(5).tolist(), "->", prio[:5].ravel())

# 6. Effet de l'échelle sur une distance entre deux relevés
def parts_distance(X, n_paires=5000, seed=0):
    r = np.random.default_rng(seed)
    i, j = r.integers(0, len(X), (2, n_paires))
    d2 = ((X[i] - X[j]) ** 2).mean(axis=0)
    return 100 * d2 / d2.sum()

res = pd.DataFrame({"brute_%": parts_distance(X_train),
                    "standardisée_%": parts_distance(X_train_std)}, index=mesures).round(1)
print("\nPart de chaque variable dans le carré de la distance :\n", res)
a, b = 0, 200
print(f"Relevés {a} et {b} : distance brute = {np.linalg.norm(X_train[a]-X_train[b]):.2f}, "
      f"standardisée = {np.linalg.norm(X_train_std[a]-X_train_std[b]):.2f}")

# 7. Contrôles
print(f"\nDécoupage : {len(train)} lignes d'entraînement, {len(test)} lignes de test")
print("Médianes du train :", imputer.statistics_.round(2))
print("Moyennes après standardisation (train / test) :",
      X_train_std.mean(axis=0).round(2), X_test_std.mean(axis=0).round(2))
print("Min-max du test hors [0,1] :", int(((X_test_01 < 0) | (X_test_01 > 1)).sum()), "valeurs")
print("Colonnes one-hot :", onehot.get_feature_names_out())
print("Valeurs manquantes restantes (train) :", int(np.isnan(X_train).sum()))