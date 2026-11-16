#!/usr/bin/env python3
"""Génère un jeu synthétique reproductible pour les TP de l'atelier connecté."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42

def generer(n_par_machine: int = 120, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    machines = [f"M-{i:02d}" for i in range(1, 7)]
    profils = {
        "M-01": (67, 1.4, 11.0, 58), "M-02": (70, 1.7, 12.2, 55),
        "M-03": (76, 2.9, 15.0, 48), "M-04": (78, 3.2, 15.8, 45),
        "M-05": (72, 2.1, 13.0, 52), "M-06": (82, 4.1, 17.5, 41),
    }
    rows = []
    for m_idx, machine in enumerate(machines):
        temp0, vib0, courant0, cadence0 = profils[machine]
        for t in range(n_par_machine):
            # Quelques fenêtres de dérive/incident artificielles, sans référence réelle.
            incident = (t % 37 in (0, 1, 2)) or (machine == "M-06" and t % 19 == 0)
            temp = temp0 + 0.018*t + rng.normal(0, 2.2) + (8 if incident else 0)
            vibration = vib0 + rng.normal(0, 0.42) + (2.7 if incident else 0)
            courant = courant0 + rng.normal(0, 1.0) + (4.2 if incident else 0)
            cadence = cadence0 + rng.normal(0, 3.0) - (10 if incident else 0)
            peak = vibration * (1.65 + rng.normal(0, .12))
            f1 = max(0.05, vibration * (0.42 + rng.normal(0, .06)))
            f2 = max(0.05, vibration * (0.31 + rng.normal(0, .05)))
            f3 = max(0.05, vibration * (0.18 + rng.normal(0, .04)))
            f4 = max(0.05, vibration * (0.09 + rng.normal(0, .025)))
            p_defaut = 0.68 if incident else (0.10 if machine == "M-06" else 0.025)
            defaut = int(rng.random() < p_defaut)
            alarmes = {
                "alarme_temperature": int(temp > 83),
                "alarme_vibration": int(vibration > 4.2),
                "alarme_courant": int(courant > 18.5),
                "alarme_cadence": int(cadence < 38),
            }
            # Petit bruit d'alarmes, pour éviter des règles parfaitement déterministes.
            if rng.random() < .025:
                k = rng.choice(list(alarmes))
                alarmes[k] = 1 - alarmes[k]
            row = {
                "horodatage": (pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5*(m_idx*n_par_machine+t))).isoformat(),
                "machine_id": machine,
                "temperature": temp,
                "vibration": vibration,
                "courant": courant,
                "cadence": cadence,
                "vibration_pic": peak,
                "vib_f1": f1, "vib_f2": f2, "vib_f3": f3, "vib_f4": f4,
                "defaut": defaut,
                **alarmes,
                "donnees_synthetiques": True,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # Valeurs manquantes simulant pertes de télémétrie; défaut/alarmes sont conservés.
    for col, rate in [("temperature", .035), ("vibration", .045), ("courant", .025), ("cadence", .03)]:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df

CAUSES_ALARMES = {
    "alarme_temperature": ("surchauffe", "Contrôle refroidissement / remplacement ventilateur"),
    "alarme_vibration": ("vibrations excessives", "Rééquilibrage / remplacement roulement"),
    "alarme_courant": ("surintensité moteur", "Vérification bobinage et alimentation"),
    "alarme_cadence": ("baisse de cadence", "Réglage mécanisme d'entraînement"),
}
TECHNICIENS = ["T-01", "T-02", "T-03", "T-04"]
PAS_MINUTES = 5
DELAI_MIN_TICKETS = 12  # pas de nouveau ticket correctif sur une même machine pendant 1 h
PERIODE_PREVENTIF = 40  # une maintenance préventive tous les 40 relevés (≈ 3 h 20)


def generer_interventions(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Tickets de maintenance synthétiques, liés aux défauts/alarmes du jeu de télémétrie."""
    # Générateur distinct : la télémétrie reste identique à celle produite sans les tickets.
    rng = np.random.default_rng(seed + 1)
    df = df.assign(_t=pd.to_datetime(df["horodatage"]))
    alarmes = list(CAUSES_ALARMES)
    tickets = []

    def ajouter(machine, debut, type_, cause, action, priorite, duree, cout):
        tickets.append({
            "machine_id": machine,
            "date_ouverture": debut.isoformat(),
            "date_cloture": (debut + pd.Timedelta(minutes=duree)).isoformat(),
            "type_intervention": type_,
            "cause": cause,
            "action": action,
            "priorite": priorite,
            "duree_arret_min": duree,
            "cout_eur": round(float(cout), 2),
            "technicien": rng.choice(TECHNICIENS),
            "statut": "clos",
            "donnees_synthetiques": True,
        })

    for machine, g in df.groupby("machine_id", sort=True):
        g = g.reset_index(drop=True)
        dernier = -DELAI_MIN_TICKETS
        for i, r in g.iterrows():
            actives = [a for a in alarmes if r[a] == 1]
            if r["defaut"] != 1 or i - dernier < DELAI_MIN_TICKETS:
                continue
            # Un défaut accompagné d'alarmes est plus souvent signalé qu'un défaut isolé.
            if rng.random() >= (0.6 if actives else 0.2):
                continue
            dernier = i
            if actives:
                cause, action = CAUSES_ALARMES[str(rng.choice(actives))]
                priorite = "haute" if len(actives) >= 2 else "moyenne"
            else:
                cause, action, priorite = "défaut qualité sans alarme", "Inspection visuelle et réglages", "basse"
            duree = int(max(15, rng.gamma(3.0, 25 if priorite != "haute" else 45)))
            cout = 80 + duree * rng.uniform(1.5, 3.0) + (rng.uniform(50, 400) if priorite == "haute" else 0)
            ajouter(machine, r["_t"] + pd.Timedelta(minutes=PAS_MINUTES), "corrective", cause, action, priorite, duree, cout)
        for i in range(PERIODE_PREVENTIF, len(g), PERIODE_PREVENTIF):
            duree = int(rng.integers(20, 60))
            ajouter(machine, g.loc[i, "_t"], "preventive", "maintenance planifiée",
                    "Graissage, contrôle serrage et nettoyage", "basse", duree, 60 + duree * rng.uniform(1.0, 1.5))
    out = pd.DataFrame(tickets).sort_values(["date_ouverture", "machine_id"], kind="stable").reset_index(drop=True)
    out.insert(0, "ticket_id", [f"TK-{n:04d}" for n in range(1, len(out) + 1)])
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("atelier_connecte.csv"), help="Chemin CSV de sortie")
    parser.add_argument("--output-interventions", type=Path, default=Path(__file__).with_name("interventions_maintenance.csv"), help="Chemin CSV des tickets de maintenance")
    parser.add_argument("--n-par-machine", type=int, default=120)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.n_par_machine < 10:
        parser.error("--n-par-machine doit être au moins égal à 10")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_interventions.parent.mkdir(parents=True, exist_ok=True)
    df = generer(args.n_par_machine, args.seed)
    df.to_csv(args.output, index=False)
    interventions = generer_interventions(df, args.seed)
    interventions.to_csv(args.output_interventions, index=False)
    print(f"CSV synthétique : {args.output} ({len(df)} lignes, {df['machine_id'].nunique()} machines, seed={args.seed})")
    print(f"CSV interventions : {args.output_interventions} ({len(interventions)} tickets)")

if __name__ == "__main__":
    main()
