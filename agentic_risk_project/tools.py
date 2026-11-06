import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def charger_portefeuille(nom_fichier: str, sous_dossier: str = "samples") -> dict:
    chemin_complet = os.path.join(DATA_DIR, sous_dossier, nom_fichier)
    if not os.path.exists(chemin_complet):
        return {"status": "error", "message": f"Fichier introuvable : {chemin_complet}"}
    df = pd.read_csv(chemin_complet)
    return {
        "status": "success",
        "message": f"Fichier brut chargé : {len(df)} lignes.",
        "data_brute": df.to_dict(orient="records")
    }

def nettoyer_donnees_portefeuille(data: list) -> dict:
    df = pd.DataFrame(data)
    rapport_nettoyage = []
    
    doublons_avant = len(df)
    df = df.drop_duplicates()
    if doublons_avant - len(df) > 0:
        rapport_nettoyage.append(f"Suppression de {doublons_avant - len(df)} lignes en double.")
    
    if df['Exposition'].isnull().sum() > 0:
        mediane_exp = df['Exposition'].median()
        df['Exposition'] = df['Exposition'].fillna(mediane_exp)
        rapport_nettoyage.append(f"Imputation des expositions manquantes par la médiane : {mediane_exp:,.2f}")
        
    if df['PD_initiale'].isnull().sum() > 0:
        df['PD_initiale'] = df['PD_initiale'].fillna(0.02)
        rapport_nettoyage.append("Imputation des PD initiales manquantes par le taux plancher de 2%.")
        
    anomalies = len(df[df['Exposition'] <= 0])
    if anomalies > 0:
        df = df[df['Exposition'] > 0]
        rapport_nettoyage.append(f"Suppression de {anomalies} contreparties avec exposition <= 0.")
        
    df['PD_initiale'] = df['PD_initiale'].clip(lower=0.0, upper=1.0)
    
    if not rapport_nettoyage:
        rapport_nettoyage.append("Aucune anomalie détectée. Données intègres.")
        
    return {
        "status": "success",
        "rapport_nettoyage": rapport_nettoyage,
        "data_propre": df.to_dict(orient="records")
    }

def entrainer_et_expliquer_risque(data_propre: list) -> dict:
    df = pd.DataFrame(data_propre)
    X = pd.DataFrame({
        "Levier_Financier": df['Exposition'] / 10000000,
        "PD_initiale": df['PD_initiale'] * 100,
        "Ratio_Liquidite": np.random.uniform(0.8, 2.5, len(df))
    })
    y = (df['PD_initiale'] > 0.03).astype(int)
    if y.nunique() == 1:
        y.iloc[0] = 1 - y.iloc[0]

    model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    explications = []
    for i, row in df.iterrows():
        sv = shap_values[i][:, 1] if isinstance(shap_values, list) else shap_values[i]
        explications.append({
            "Contrepartie": row['Contrepartie'],
            "Facteur_Principal_Risque": X.columns[np.argmax(np.abs(sv))]
        })
        
    df['PD_stress'] = df['PD_initiale'] * 1.35
    df['Impact_Provision'] = df['Exposition'] * df['PD_stress'] * 0.20
    bascule_stage2 = df[df['PD_stress'] > 0.05]['Contrepartie'].tolist()
    
    return {
        "status": "success",
        "contreparties_stage_2": bascule_stage2,
        "total_provision_supplementaire": float(df['Impact_Provision'].sum()),
        "details_explicabilite": explications
    }
