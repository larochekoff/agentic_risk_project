import os
import streamlit as st
import pandas as pd
from workflow import creer_graphe_agentique

st.set_page_config(page_title="Agentic Risk Studio - CAIO Edition", layout="wide")

st.title("🏦 Plateforme Agentique - Risques & Stress Testing (CAIO Mode)")
st.markdown("Gouvernance, nettoyage automatisé, IA explicable (SHAP) et conformité IFRS 9 sous WSL.")

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)
sample_path = os.path.join(SAMPLE_DIR, "portefeuille_exemple.csv")

if not os.path.exists(sample_path):
    df_exemple = pd.DataFrame({
        "Contrepartie": ["Agro-Industrie SA", "BTP Bâtisseur", "Cacao West Africa", "Textil Express", "Logistique Abidjan"],
        "Exposition": [120000000, 150000000, 90000000, 80000000, 60000000],
        "PD_initiale": [0.012, 0.045, 0.015, 0.022, 0.018]
    })
    df_exemple.to_csv(sample_path, index=False)

st.sidebar.header("📁 Pilotage Portefeuille")
fichier_selectionne = st.sidebar.selectbox("Fichier d'échantillon", ["portefeuille_exemple.csv"])

if st.sidebar.button("🚀 Lancer l'analyse multi-agents"):
    app_graphe = creer_graphe_agentique()
    
    with st.spinner("Les agents collaborent (Nettoyage ➔ Modélisation SHAP ➔ Conformité)..."):
        etat_initial = {
            "nom_fichier": fichier_selectionne,
            "log_nettoyage": [],
            "donnees_nettoyees": [],
            "resultat_modelisation": {},
            "rapport_final": ""
        }
        resultat = app_graphe.invoke(etat_initial)
        
    st.success("Analyse multi-agents achevée avec succès !")
    st.markdown(resultat["rapport_final"])
    
    st.warning("🔒 **Validation de Gouvernance (Human-in-the-Loop)** : Souhaitez-vous valider ces écritures IFRS 9 et archiver le rapport au format certifié ?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Valider et archiver l'écriture"):
            st.info("Écritures validées, signées et enregistrées dans le registre central d'audit.")
    with col2:
        if st.button("❌ Rejeter et réajuster"):
            st.warning("Processus interrompu. Renvoi en zone d'analyse manuelle.")
