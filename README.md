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
| **Score d’inconfort (%)** | Moyenne du score d'inconfort des tweets |

---

## 🤖 Approche pour l'analyse des sentiments

- Un modèle de **traitement du langage naturel (NLP)** est utilisé pour classifier les sentiments en **positif, neutre ou négatif**.
- Une analyse **temporelle** des sentiments est réalisée pour observer les tendances.

---

## 🎯 Création des agents IA

### 🔍 Logique de détection des types de réclamations
- Les types de problèmes sont extraits en analysant les **mots-clés critiques** et les **contextes** des tweets.
- Une classification est effectuée pour regrouper les tweets par **catégorie de problème**.

### 💬 Prompts
- Un agent conversationnel peut être intégré pour **générer des réponses automatiques** aux réclamations. C'est du **prompt tuning**.
 

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

#### Installation de l'environnement virtuel

```bash
# Créer un environnement virtuel
python -m venv env

# Activer l'environnement virtuel
# Sur Windows
env\Scripts\activate
# Sur macOS/Linux
source env/bin/activate

```

### 2️⃣ Installer les dépendances

Depuis la racine du projet, exécutez :

```bash
pip install -r requirements.txt
```

### 3️⃣ Lancer l'application

```bash
streamlit run src/app.py
```
