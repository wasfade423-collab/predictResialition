import pandas as pd

def preparer(X):
    """Encodage + nouvelles variables. Fichier séparé pour que le modèle sauvegardé
    puisse être rechargé dans une autre application (Streamlit...)."""
    X = X.copy()
    if not pd.api.types.is_numeric_dtype(X["zone"]):
        X["zone"] = X["zone"].map({"urbaine": 0, "rurale": 1})
    if not pd.api.types.is_numeric_dtype(X["type_offre"]):
        X["type_offre"] = X["type_offre"].map({"prépayé": 0, "postpayé": 1})
    anc = X["anciennete_mois"].astype(float) + 1
    X["depense_par_mois_anc"] = X["depense_fcfa"] / anc
    X["reclam_par_mois_anc"] = X["reclamations_3m"] / anc
    X["usage_total"] = X["appels_min"] + X["data_mo"] + X["sms"]
    X["depense_par_recharge"] = X["depense_fcfa"] / (X["nb_recharges"] + 1)
    X["baisse_conso"] = (X["variation_conso_pct"] < 0).astype(int)
    X["note_x_reclam"] = X["note_reseau"] * X["reclamations_3m"]
    return X.drop(columns=[c for c in ["id_client", "date_activation"] if c in X.columns])
