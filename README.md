# Analyse des Tweets Engie avec Streamlit

Ce projet vise à analyser les tweets mentionnant **Engie** en utilisant **Streamlit** pour la visualisation interactive des données.

## 📊 Méthodologie de traitement des tweets

1. **Collecte des données**  
   - Les tweets sont récupérés et stockés dans un fichier CSV.
   - Les données incluent la **date de publication**, le **sentiment**, le **type de problème**, les **mots critiques**, etc.

2. **Nettoyage et transformation**  
   - Conversion des dates en format `datetime`.
   - Correction des valeurs manquantes et incohérences.
   - Conversion des colonnes sous forme de listes (ex. : `mot-critique` et `Probleme`).

3. **Calcul des indicateurs**  
   - Nombre total de tweets.
   - Moyenne de tweets par jour.
   - Pourcentage de tweets urgents.
   - Score moyen d'inconfort.

---

## 📈 KPI et leur signification

| KPI | Description |
|------|------------|
| **Total tweets** | Nombre total de tweets analysés |
| **Tweets par jour** | Moyenne quotidienne des tweets publiés |
| **Tweets urgents (%)** | Proportion de tweets signalés comme urgents |
| **Score d’inconfort (%)** | Moyenne du score d’inconfort des tweets |

---

## 🤖 Approche pour l'analyse des sentiments

- Un modèle de **traitement du langage naturel (NLP)** est utilisé pour classifier les sentiments en **positif, neutre ou négatif**.
- Une analyse **temporelle** des sentiments est réalisée pour observer les tendances.

---

## 🎯 Création des agents IA

### 🔍 Logique de détection des types de réclamations
- Les types de problèmes sont extraits en analysant les **mots-clés critiques** et les **contextes** des tweets.
- Une classification est effectuée pour regrouper les tweets par **catégorie de problème**.

### 💬 Prompts et Fine-tuning
- Un modèle NLP a été **fine-tuné** pour comprendre le contexte des réclamations.
- Un agent conversationnel peut être intégré pour **générer des réponses automatiques** aux réclamations.

### 📝 Exemples d'interactions
- **Tweet :** "Mon compteur Linky ne fonctionne plus depuis 2 jours, aucune aide !"  
- **Agent IA :** "Bonjour, nous sommes désolés pour cette situation. Avez-vous essayé de redémarrer votre compteur ? Nous pouvons vous guider."

---

## 📊 Choix technologiques pour la visualisation

| Outil | Raison |
|------|--------|
| **Streamlit** | Interface interactive, facile à utiliser pour l'exploration des données |
| **Plotly** | Graphiques dynamiques pour l'analyse des tendances |
| **Matplotlib & WordCloud** | Nuages de mots et histogrammes |

---

## 🚀 Exécution du projet

### 1️⃣ Prérequis

Installez les dépendances avec :

```bash
pip install -r source/requirements.txt

### 2️⃣ Lancer l'application

```bash
streamlit run source/app.py


