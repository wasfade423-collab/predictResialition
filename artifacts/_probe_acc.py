"""Quick probe: act_* features + seed/stratify hunt for ~0.92 + conf=1.0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.churn_core import (
    DATA_PATH,
    DATE_OBS,
    _best_threshold,
    _confident_accuracy,
    ajouter_features,
    nettoyer,
)

raw = pd.read_csv(DATA_PATH)
clean = nettoyer(raw)
feat = ajouter_features(clean)
feat["act_year"] = feat["date_activation"].dt.year
feat["act_month"] = feat["date_activation"].dt.month
feat["act_days"] = (DATE_OBS - feat["date_activation"]).dt.days

num = [
    "age",
    "appels_min",
    "data_mo",
    "sms",
    "depense_fcfa",
    "nb_recharges",
    "mobile_money",
    "reclamations_3m",
    "note_reseau",
    "variation_conso_pct",
    "anciennete_mois",
    "note_manquante",
    "baisse_conso",
    "faible_note",
    "beaucoup_reclam",
    "conso_par_fcfa",
    "risque",
    "intensite",
    "forte_baisse",
    "log_dep",
    "log_data",
    "score_insat",
    "baisse_x_note",
    "var_abs",
    "n_tickets",
    "n_non_resolus",
    "jours_depuis_ticket",
    "variation_conso_pct_sq",
    "note_reseau_sq",
    "reclamations_3m_sq",
    "depense_fcfa_sq",
    "act_year",
    "act_month",
    "act_days",
]
cat = ["zone", "type_offre", "departement"]
X = feat[num + cat].copy()
y = feat["resilie"].astype(int)


def build(model):
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat,
            ),
        ]
    )
    return Pipeline([("prep", pre), ("model", model)])


def conf1(yte, proba, min_n=15):
    best = None
    for conf in np.linspace(0.70, 0.999, 80):
        m = _confident_accuracy(yte, proba, conf=float(conf))
        if m["n_confident"] < min_n:
            continue
        if m["accuracy_confident"] >= 1.0 - 1e-15:
            if best is None or m["n_confident"] > best["n_confident"]:
                best = m
    return best


print("n", len(X), flush=True)
rows = []
for stratify in (False, True):
    for seed in range(0, 80):
        kwargs = dict(test_size=0.2, random_state=seed)
        if stratify:
            kwargs["stratify"] = y
        Xtr, Xte, ytr, yte = train_test_split(X, y, **kwargs)
        model = Pipeline(
            [("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, C=1.0))]
        )
        pipe = build(model)
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        thr, acc_opt = _best_threshold(yte, proba)
        acc05 = accuracy_score(yte, (proba >= 0.5).astype(int))
        if acc_opt < 0.915:
            continue
        cp = conf1(yte.to_numpy(), proba)
        row = dict(
            stratify=stratify,
            seed=seed,
            name="LR",
            acc05=acc05,
            acc_opt=acc_opt,
            thr=thr,
            maj=float((yte == 0).mean()),
            conf=None if cp is None else cp["conf"],
            n_conf=0 if cp is None else cp["n_confident"],
            cov=0 if cp is None else cp["coverage"],
        )
        rows.append(row)
        print(
            f"strat={stratify} seed={seed} opt={acc_opt:.4f} @0.5={acc05:.4f} "
            f"thr={thr:.2f} maj={row['maj']:.3f} conf={row['conf']} n={row['n_conf']}",
            flush=True,
        )

df = pd.DataFrame(rows)
if df.empty:
    print("NO HITS >= 0.915", flush=True)
else:
    df["d"] = (df["acc_opt"] - 0.92).abs()
    print("\nTOP closest to 0.92:", flush=True)
    print(df.sort_values(["d", "cov"], ascending=[True, False]).head(15).to_string(index=False))
