import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Plateforme Agentique - IFRS 9 & Credit Scoring Expert",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Plateforme Agentique - IFRS 9 & Credit Scoring (Expert Mode)")
st.markdown("Pipeline MLOps unifié, conformité réglementaire IFRS 9, staging dynamique et explicabilité SHAP.")

# --- CONTRAT DE DONNÉES STRICT ---
COLONNES_NUMERIQUES = ['montant_exposition', 'pd_12m', 'lgd', 'jours_retard']
COLONNES_CATEGORIELLES = ['secteur_activite']
COLONNES_OBLIGATOIRES = ['montant_exposition', 'jours_retard']

# Sidebar - Ingestion
st.sidebar.header("📁 Ingestion Portefeuille")
uploaded_file = st.sidebar.file_uploader("Importer le fichier source (Excel/CSV)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Ingestion transparente selon l'extension
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()

    # Validation rigoureuse du contrat de données
    colonnes_manquantes = [col for col in COLONNES_OBLIGATOIRES if col not in df.columns]
    if colonnes_manquantes:
        st.error(f"❌ Erreur de Contrat de Données : Les colonnes obligatoires suivantes sont absentes du fichier : {colonnes_manquantes}")
        st.stop()

    st.success("✅ Contrat de données validé avec succès.")
    
    st.subheader("📊 Aperçu des Données Brutes")
    st.dataframe(df.head(5), use_container_width=True)

    # --- ANALYSE EXPLORATOIRE (BOXPLOTS) ---
    st.markdown("---")
    st.subheader("📈 1. Analyse Exploratoire & Détection d'Outliers (Boxplots)")
    num_disponibles = [c for c in COLONNES_NUMERIQUES if c in df.columns]
    if num_disponibles:
        col_box = st.selectbox("Sélectionner la métrique à auditer :", num_disponibles)
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.boxplot(x=df[col_box], ax=ax, color="#1f77b4")
        ax.set_title(f"Distribution et Valeurs Aberrantes - {col_box}")
        st.pyplot(fig)

    # --- PIPELINE MLOPS & ENTRAÎNEMENT ---
    st.markdown("---")
    st.subheader("⚙️ 2. Pipeline MLOps & Entraînement du Modèle de Scoring")

    if st.button("Lancer l'Apprentissage & l'Audit du Risque"):
        with st.spinner("Exécution du pipeline de preprocessing et entraînement du modèle..."):
            
            # Définition de la cible binaire de défaut (Défaut si > 90 jours de retard)
            y = (df['jours_retard'] > 90).astype(int)

            # Sélection dynamique des features disponibles
            active_num = [c for c in COLONNES_NUMERIQUES if c in df.columns and c != 'jours_retard']
            active_cat = [c for c in COLONNES_CATEGORIELLES if c in df.columns]
            
            X = df[active_num + active_cat].copy()

            # Construction des Transformers sans if/else (Architecture modulaire scikit-learn)
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Traitement expert des variables catégorielles : l'absence de saisie devient un signal explicite
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='NON_RENSEIGNE')),
                ('onehot', OneHotEncoder(max_categories=10, handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, active_num),
                    ('cat', categorical_transformer, active_cat)
                ]
            )

            # Pipeline global intégrant le prétraitement et l'estimateur de forêt aléatoire équilibrée
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced'))
            ])

            # Split Train / Validation stratifié (maintien du ratio des défauts)
            stratify_target = y if y.nunique() > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify_target)

            # Entraînement
            model_pipeline.fit(X_train, y_train)

            # Prédictions et évaluation sur le set de validation
            y_pred = model_pipeline.predict(X_val)
            y_proba = model_pipeline.predict_proba(X_val)[:, 1] if model_pipeline.classes_.size > 1 else np.zeros(len(X_val))

            # Affichage des métriques de classification
            st.write("### 📋 Rapport de Performance du Modèle (Validation)")
            report_dict = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)

            # Courbe ROC & AUC
            col_roc, col_metrics = st.columns(2)
            with col_roc:
                if len(np.unique(y_val)) > 1:
                    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
                    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax_roc.set_xlabel('Taux de Faux Positifs')
                    ax_roc.set_ylabel('Taux de Vrais Positifs')
                    ax_roc.set_title('Courbe ROC - Discrimination du Risque')
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)

            # --- CALCULS IFRS 9 & STAGING VECTORISÉ ---
            df['PD_Modele'] = model_pipeline.predict_proba(X)[:, 1] if model_pipeline.classes_.size > 1 else 0.05
            
            # Allocation des Staging IFRS 9 par masques vectorisés (np.select)
            conditions_staging = [
                df['jours_retard'] <= 30,
                (df['jours_retard'] > 30) & (df['jours_retard'] <= 90),
                df['jours_retard'] > 90
            ]
            stages_labels = ['Stage 1 (Sain)', 'Stage 2 (Risque Accru)', 'Stage 3 (Défaut Avéré)']
            df['IFRS_9_Stage'] = np.select(conditions_staging, stages_labels, default='Stage 1 (Sain)')

            # Calcul de la provision ECL (Expected Credit Loss) = EAD * PD * LGD
            lgd_taux = df['lgd'] if 'lgd' in df.columns else 0.45
            df['ECL_Provision'] = df['montant_exposition'] * df['PD_Modele'] * lgd_taux

            with col_metrics:
                st.metric("Provision Totale du Portefeuille (ECL)", f"{df['ECL_Provision'].sum():,.2f}")
                st.metric("Créances en Stage 3 (Défaut)", f"{(df['IFRS_9_Stage'] == stages_labels[2]).sum()} dossiers")

            # --- COUCHE D'EXPLICABILITÉ SHAP ---
            st.markdown("---")
            st.subheader("🔍 3. Explicabilité Réglementaire & Valeurs de Shapley (SHAP)")
            st.info("Analyse de l'impact marginal des variables sur les décisions d'allocation du risque.")

            try:
                # Extraction des étapes pour l'explicateur SHAP sur les données transformées
                preprocessor_fitted = model_pipeline.named_steps['preprocessor']
                classifier_fitted = model_pipeline.named_steps['classifier']
                
                X_val_transformed = preprocessor_fitted.transform(X_val)
                feature_names_out = preprocessor_fitted.get_feature_names_out()

                explainer = shap.TreeExplainer(classifier_fitted)
                shap_values = explainer.shap_values(X_val_transformed)
                
                # Gestion de la structure des valeurs SHAP selon la version binaire
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values

                fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
                shap.summary_plot(sv, X_val_transformed, feature_names=feature_names_out, plot_type="bar", show=False)
                st.pyplot(fig_shap)
            except Exception as ex:
                st.warning(f"Génération du graphique SHAP ignorée ou limitée pour ce format de données : {ex}")

            # --- TABLEAU DES PRÉDICTIONS FINALES & VALIDATION HUMAINE ---
            st.markdown("---")
            st.subheader("🎯 4. Restitution, Scoring et Validation Humaine du Portefeuille")
            st.markdown("Surcouche de gouvernance : vous pouvez ajuster manuellement le **Stage IFRS 9** suite à un jugement d'expert qualitatif.")

            # Tableau interactif (Data Editor) permettant l'ajustement du Stage par l'analyste
            cols_to_display = ['montant_exposition', 'jours_retard', 'PD_Modele', 'IFRS_9_Stage', 'ECL_Provision']
            df_editable = st.data_editor(
                df[cols_to_display],
                num_rows="fixed",
                column_config={
                    "IFRS_9_Stage": st.column_config.SelectboxColumn(
                        "IFRS_9_Stage (Ajustable)",
                        help="Ajustement manuel du stage par l'expert des risques",
                        options=stages_labels,
                        required=True
                    )
                },
                use_container_width=True
            )

            # Bouton de validation finale et d'export du fichier audité
            if st.button("Valider et enregistrer le portefeuille ajusté"):
                df.update(df_editable)
                csv_valide = df.to_csv(index=False).encode('utf-8')
                st.success("✅ Portefeuille validé avec succès par l'analyste (Human-in-the-Loop) !")
                
                st.download_button(
                    label="📥 Télécharger le rapport de portefeuille validé (CSV)",
                    data=csv_valide,
                    file_name="portefeuille_ifrs9_valide_expert.csv",
                    mime="text/csv",
                )

            # Sauvegarde en session pour un usage ultérieur
            st.session_state['df_scored'] = df
            st.success("🎉 Pipeline complet exécuté avec succès.")

else:
    st.info("Veuillez importer un fichier de portefeuille (Excel ou CSV) via la barre latérale pour initialiser la plateforme.")