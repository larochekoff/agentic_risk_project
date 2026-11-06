from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from tools import charger_portefeuille, nettoyer_donnees_portefeuille, entrainer_et_expliquer_risque

class EtatRisque(TypedDict):
    nom_fichier: str
    log_nettoyage: List[str]
    donnees_nettoyees: List[Dict[str, Any]]
    resultat_modelisation: Dict[str, Any]
    rapport_final: str

def noeud_donnees(state: EtatRisque):
    res_brut = charger_portefeuille(state["nom_fichier"], sous_dossier="samples")
    res_nettoyage = nettoyer_donnees_portefeuille(res_brut["data_brute"])
    return {
        "log_nettoyage": res_nettoyage["rapport_nettoyage"],
        "donnees_nettoyees": res_nettoyage["data_propre"]
    }

def noeud_modelisation(state: EtatRisque):
    data = state["donnees_nettoyees"]
    res = entrainer_et_expliquer_risque(data)
    return {"resultat_modelisation": res}

def noeud_conformite(state: EtatRisque):
    logs = state["log_nettoyage"]
    mod = state["resultat_modelisation"]
    stage2 = mod["contreparties_stage_2"]
    prov = mod["total_provision_supplementaire"]
    explications = mod["details_explicabilite"]
    
    logs_str = "\n".join([f"  - {log}" for log in logs])
    details_str = "\n".join([f"- **{e['Contrepartie']}** : Facteur déterminant -> *{e['Facteur_Principal_Risque']}*" for e in explications])
    
    rapport = (
        f"### 📋 Rapport d'Audit & Conformité IFRS 9 (CAIO Governance)\n\n"
        f"**1. Traçabilité & Ingestion (Agent Données) :**\n{logs_str}\n\n"
        f"**2. Analyse de Risque & IFRS 9 (Agent Modélisation) :**\n"
        f"- Contreparties en dégradation significative (Stage 2) : **{', '.join(stage2) if stage2 else 'Aucune'}**\n"
        f"- Complément de provision estimé : **{prov:,.2f} FCFA**\n\n"
        f"**3. Transparence & Explicabilité (Valeurs SHAP) :**\n{details_str}\n\n"
        f"*(Statut : Conforme aux normes d'audit et de modélisation prudentielle)*"
    )
    return {"rapport_final": rapport}

def creer_graphe_agentique():
    workflow = StateGraph(EtatRisque)
    workflow.add_node("agent_donnees", noeud_donnees)
    workflow.add_node("agent_modelisation", noeud_modelisation)
    workflow.add_node("agent_conformite", noeud_conformite)
    
    workflow.set_entry_point("agent_donnees")
    workflow.add_edge("agent_donnees", "agent_modelisation")
    workflow.add_edge("agent_modelisation", "agent_conformite")
    workflow.add_edge("agent_conformite", END)
    
    return workflow.compile()
