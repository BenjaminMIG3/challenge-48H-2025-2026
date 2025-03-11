import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from datetime import timedelta

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse des Tweets Engie",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appliquer CSS personnalisé pour un design épuré
st.markdown("""
<style>
    .main {
        padding: 1rem 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: darkblue;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8BF9;
        color: white;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stPlotlyChart {
        background-color: black;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        padding: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Charger les données avec mise en cache
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file, sep=';', quotechar='"', encoding='utf-8', on_bad_lines='warn')
        
        # Nettoyage et préparation des données
        df['Date_de_publication'] = pd.to_datetime(df['Date_de_publication'], errors='coerce')
        
        # Convertir les chaînes de listes en listes Python
        def safe_literal_eval(x):
            try:
                return ast.literal_eval(x) if pd.notna(x) else []
            except (ValueError, SyntaxError):
                return []

        for col in ['Probleme', 'mot-critique']:
            if col in df.columns:
                df[col] = df[col].apply(safe_literal_eval)
        
        # Assurer que les colonnes numériques sont du bon type
        if 'incomfort' in df.columns:
            df['incomfort'] = pd.to_numeric(df['incomfort'], errors='coerce')
        
        # Convertir Urgence en booléen si nécessaire
        if 'Urgence' in df.columns:
            df['Urgence'] = df['Urgence'].astype(str).str.lower() == 'true'
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier CSV : {str(e)}")
        return None

# Fonction pour afficher les KPI avec animation
def display_kpis(df):
    st.subheader("📈 Indicateurs clés de performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Total tweets", 
                f"{len(df):,}",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            tweets_per_day = round(len(df) / max(df['Date_de_publication'].dt.date.nunique(), 1), 1)
            st.metric(
                "Tweets par jour", 
                f"{tweets_per_day:,}",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'Urgence' in df.columns:
                urgent_pct = df['Urgence'].mean() * 100
                st.metric(
                    "Tweets urgents", 
                    f"{urgent_pct:.1f}%",
                    delta=None
                )
            else:
                st.metric("Tweets urgents", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'incomfort' in df.columns:
                incomfort_avg = df['incomfort'].mean()
                st.metric(
                    "Score d'inconfort", 
                    f"{incomfort_avg:.1f}%",
                    delta=None
                )
            else:
                st.metric("Score d'inconfort", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

# Fonction pour les graphiques interactifs améliorés
def display_charts(df):
    # Créer des onglets pour organiser les visualisations
    tabs = st.tabs(["Chronologie", "Sentiments", "Types de problèmes", "Mots clés", "Données brutes"])
    
    # Calcul des dates min et max
    min_date = pd.Timestamp(df['Date_de_publication'].min()).tz_localize(None)
    max_date = pd.Timestamp(df['Date_de_publication'].max()).tz_localize(None)
    
    # Sidebar pour les filtres
    with st.sidebar:
        st.header("Filtres")
        
        # Ajouter un sélecteur rapide de périodes
        period_options = {
            "Dernière semaine": 7,
            "Dernier mois": 30,
            "Dernier trimestre": 90,
            "Dernière année": 365,
            "Tout": None
        }
        
        selected_period = st.selectbox(
            "Période d'analyse",
            options=list(period_options.keys()),
            index=1,  # Par défaut sur "Dernier mois"
            help="Choisissez une période prédéfinie"
        )
        
        # Autres filtres existants (inchangés)
        if 'type' in df.columns:
            types = df['type'].dropna().unique()
            selected_types = st.multiselect(
                "Types de problèmes",
                options=types,
                default=types[:5] if len(types) > 5 else types
            )
        else:
            selected_types = []
            
        if 'Sentiment' in df.columns:
            sentiments = df['Sentiment'].dropna().unique()
            selected_sentiments = st.multiselect(
                "Sentiments",
                options=sentiments,
                default=sentiments
            )
        else:
            selected_sentiments = []
            
        if st.button("Réinitialiser les filtres"):
            pass

    # Application des filtres de périodes prédéfinies
    filtered_df = df.copy()
    filtered_df['Date_de_publication'] = filtered_df['Date_de_publication'].dt.tz_localize(None)
    
    # Si une période rapide est sélectionnée, ajuster les dates
    if selected_period != "Tout" and period_options[selected_period]:
        days = period_options[selected_period]
        start_date = pd.Timestamp(max_date - timedelta(days=days)).replace(hour=0, minute=0, second=0)
        end_date = pd.Timestamp(max_date).replace(hour=23, minute=59, second=59)
    else:
        start_date = min_date
        end_date = max_date
    
    # Application du filtre temporel avec prise en compte des heures
    filtered_df = filtered_df[
        (filtered_df['Date_de_publication'] >= start_date) & 
        (filtered_df['Date_de_publication'] <= end_date)
    ]
    
    # Application des autres filtres (inchangé)
    if selected_types and 'type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
    if selected_sentiments and 'Sentiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiments)]
    
    # Vérification des données filtrées
    if filtered_df.empty:
        st.warning(f"Aucune donnée pour la période du {start_date.strftime('%d/%m/%Y %H:%M')} au {end_date.strftime('%d/%m/%Y %H:%M')}")
        return
        
    # Onglet 1: Chronologie
    with tabs[0]:
        st.subheader("Évolution temporelle des tweets")
        
        # Grouper par jour et calculer différentes métriques
        tweets_per_day = filtered_df.groupby(filtered_df['Date_de_publication'].dt.date).agg({
            'Date_de_publication': 'count',
            'incomfort': 'mean' if 'incomfort' in filtered_df.columns else 'size',
            'Urgence': 'mean' if 'Urgence' in filtered_df.columns else 'size'
        })
        
        # Renommer pour éviter la duplication de colonne
        tweets_per_day = tweets_per_day.rename(columns={'Date_de_publication': 'count'})
        
        # Reset index sans conserver le nom de l'index comme nouvelle colonne
        tweets_per_day = tweets_per_day.reset_index()
        
        # Graphique d'évolution temporelle amélioré
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=tweets_per_day['Date_de_publication'],
            y=tweets_per_day['count'],
            mode='lines+markers',
            name='Nombre de tweets',
            line=dict(color='#4F8BF9', width=3),
            marker=dict(size=8)
        ))
        
        if 'incomfort' in tweets_per_day.columns:
            fig.add_trace(go.Scatter(
                x=tweets_per_day['Date_de_publication'],
                y=tweets_per_day['incomfort'],
                mode='lines',
                name="Score d'inconfort",
                yaxis='y2',
                line=dict(color='#FF5757', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title='Évolution du volume de tweets et du score d\'inconfort',
            xaxis_title='Date',
            yaxis_title='Nombre de tweets',
            yaxis2=dict(
                title="Score d'inconfort",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tendances sur la période
        st.subheader("Tendances sur la période sélectionnée")
        trend_cols = st.columns(3)
        
        with trend_cols[0]:
            if len(tweets_per_day) > 1:
                first_day = tweets_per_day.iloc[0]['count']
                last_day = tweets_per_day.iloc[-1]['count']
                trend = ((last_day - first_day) / first_day * 100) if first_day > 0 else 0
                st.metric(
                    "Évolution du volume", 
                    f"{last_day:.0f} tweets",
                    f"{trend:.1f}% depuis le début"
                )
            else:
                st.metric("Évolution du volume", "Données insuffisantes")
                
        with trend_cols[1]:
            if 'Urgence' in tweets_per_day.columns and len(tweets_per_day) > 0:
                avg_urgence = tweets_per_day['Urgence'].mean() * 100
                st.metric(
                    "Moyenne d'urgence", 
                    f"{avg_urgence:.1f}%"
                )
            else:
                st.metric("Moyenne d'urgence", "N/A")
                
        with trend_cols[2]:
            if 'incomfort' in tweets_per_day.columns and len(tweets_per_day) > 0:
                avg_incomfort = tweets_per_day['incomfort'].mean()
                st.metric(
                    "Score d'inconfort moyen", 
                    f"{avg_incomfort:.1f}%"
                )
            else:
                st.metric("Score d'inconfort moyen", "N/A")
    
    # Onglet 2: Sentiments
    with tabs[1]:
        if 'Sentiment' in filtered_df.columns:
            st.subheader("Analyse des sentiments")
            
            sentiment_cols = st.columns([2, 1])
            
            with sentiment_cols[0]:
                # Compter les sentiments
                sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'count']
                
                # Total pour calculer les pourcentages
                total = sentiment_counts['count'].sum()
                sentiment_counts['percentage'] = (sentiment_counts['count'] / total * 100).round(1)
                
                # Créer un graphique en anneau plus esthétique
                colors = {
                    'positif': '#36A2EB',
                    'neutre': '#FFCE56',
                    'négatif': '#FF6384',
                    'positive': '#36A2EB', 
                    'neutral': '#FFCE56', 
                    'negative': '#FF6384'
                }
                
                # Obtenir les couleurs pour chaque sentiment
                color_values = [colors.get(s.lower(), '#A9A9A9') for s in sentiment_counts['Sentiment']]
                
                fig = go.Figure(data=[go.Pie(
                    labels=sentiment_counts['Sentiment'],
                    values=sentiment_counts['count'],
                    hole=.4,
                    marker_colors=color_values,
                    textinfo='label+percent',
                    textposition='outside',
                    pull=[0.05 if s.lower() in ['positif', 'positive'] else 0 for s in sentiment_counts['Sentiment']]
                )])
                
                fig.update_layout(
                    title='Répartition des sentiments',
                    annotations=[dict(text=f'{total} tweets', x=0.5, y=0.5, font_size=20, showarrow=False)],
                    height=450,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with sentiment_cols[1]:
                # Tableau des statistiques de sentiment
                st.markdown("### Détails des sentiments")
                
                for index, row in sentiment_counts.iterrows():
                    sentiment = row['Sentiment']
                    count = row['count']
                    percentage = row['percentage']
                    
                    sentiment_color = colors.get(sentiment.lower(), '#A9A9A9')
                    
                    st.markdown(f"""
                    <div style="background-color: {sentiment_color}20; border-left: 5px solid {sentiment_color}; 
                    padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                        <p style="font-size: 18px; margin: 0; color: {sentiment_color};"><strong>{sentiment}</strong></p>
                        <p style="font-size: 22px; margin: 5px 0;">{count:,} tweets ({percentage}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Évolution des sentiments dans le temps si assez de données
                if len(filtered_df) > 10:
                    # Grouper par date et sentiment
                    sentiment_time = filtered_df.groupby([
                        filtered_df['Date_de_publication'].dt.date,
                        'Sentiment'
                    ]).size().reset_index(name='count')
                    
                    # Créer un tableau croisé dynamique pour l'analyse temporelle
                    # Sans nécessiter de reset_index qui peut causer des problèmes
                    pivot_columns = sentiment_time['Sentiment'].unique().tolist()
                    
                    # La visualisation sera ajoutée dans une section séparée ci-dessous
        else:
            st.info("Les données de sentiment ne sont pas disponibles dans ce jeu de données.")
    
    # Onglet 3: Types de problèmes
    with tabs[2]:  # "Types de problèmes" tab
        st.subheader("Analyse des types de problèmes")
        
        # Aggregate data by type
        type_analysis = filtered_df['type'].value_counts().reset_index()
        type_analysis.columns = ['type', 'count']
        
        # Gère les Nan et les empty
        type_analysis = type_analysis.dropna()  # enlève les lignes avec Nan
        type_analysis['count'] = type_analysis['count'].fillna(0).astype(int)  # Vérifie que l'entrée est un entier
        
        # Ensure all sizes are positive (Plotly requires size >= 0)
        type_analysis['size'] = type_analysis['count'].apply(lambda x: max(x, 1))  # Evite les valeures à zéro et les négatives
        
        if not type_analysis.empty:
            # Create scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=type_analysis['count'],
                y=type_analysis['type'],
                mode='markers',
                marker=dict(
                    size=type_analysis['size'],
                    sizemode='area',
                    sizeref=2.*max(type_analysis['size'])/(40.**2),
                    sizemin=4
                ),
                text=type_analysis['type'] + ': ' + type_analysis['count'].astype(str) + ' occurrences',
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title="Répartition des types de problèmes",
                xaxis_title="Nombre d'occurrences",
                yaxis_title="Type de problème",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour les types de problèmes dans la période sélectionnée.")
    
    # Onglet 4: Mots clés
    with tabs[3]:
        if 'mot-critique' in filtered_df.columns:
            st.subheader("Analyse des mots clés critiques")
            
            # Extraction de tous les mots critiques
            all_mot_critique = [word for sublist in filtered_df['mot-critique'] for word in sublist if word]
            
            if all_mot_critique:
                # Créer un compteur de mots
                from collections import Counter
                word_counts = Counter(all_mot_critique)
                
                # Obtenir les mots les plus fréquents pour une visualisation précise
                top_words = dict(word_counts.most_common(50))
                
                # Colonnes pour afficher le nuage de mots et les statistiques
                word_cols = st.columns([3, 2])
                
                with word_cols[0]:
                    wordcloud = WordCloud(
                        width=800, 
                        height=500, 
                        background_color='white',
                        colormap='viridis',
                        max_words=100,
                        contour_width=1,
                        contour_color='steelblue'
                    ).generate_from_frequencies(top_words)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                
                with word_cols[1]:
                    st.markdown("### Mots les plus fréquents")
                    
                    # Créer un graphique à barres des mots les plus fréquents
                    top_20_words = dict(word_counts.most_common(20))
                    
                    # Convertir en dataframe pour la visualisation
                    word_df = pd.DataFrame({
                        'mot': list(top_20_words.keys()),
                        'fréquence': list(top_20_words.values())
                    }).sort_values('fréquence', ascending=True)
                    
                    fig = px.bar(
                        word_df,
                        x='fréquence',
                        y='mot',
                        orientation='h',
                        title='Top 20 des mots critiques',
                        color='fréquence',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        height=500,
                        xaxis_title='Fréquence',
                        yaxis_title='Mot',
                        coloraxis_showscale=False,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucun mot critique trouvé pour la période sélectionnée.")
        else:
            st.info("Les données de mots critiques ne sont pas disponibles dans ce jeu de données.")

    # Onglet 5: Données brutes
    with tabs[4]:
        st.subheader("Données brutes filtrées")
        
        # Options de téléchargement
        csv = filtered_df.to_csv(index=False, sep=';')
        b64 = base64.b64encode(csv.encode()).decode()
        download_button = f'<a href="data:file/csv;base64,{b64}" download="tweets_engie_filtered.csv" class="download-button" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4F8BF9; color: white; text-decoration: none; border-radius: 4px; margin-bottom: 1rem;">Télécharger les données filtrées (CSV)</a>'
        st.markdown(download_button, unsafe_allow_html=True)
        
        # Option de recherche dans les données
        search_term = st.text_input("Rechercher dans les données :")
        
        # Colonnes à afficher (permettre la sélection)
        all_columns = filtered_df.columns.tolist()
        default_display = [c for c in ['Date_de_publication', 'type', 'Sentiment', 'incomfort', 'Urgence'] if c in all_columns]
        display_cols = st.multiselect(
            "Colonnes à afficher :", 
            options=all_columns,
            default=default_display if default_display else all_columns[:5]
        )
        
        # Si aucune colonne n'est sélectionnée, en utiliser quelques-unes par défaut
        if not display_cols:
            display_cols = all_columns[:5]
        
        # Filtrer selon la recherche
        if search_term:
            mask = pd.Series(False, index=filtered_df.index)
            for col in filtered_df.columns:
                if filtered_df[col].dtype == object:  # Uniquement pour les colonnes texte
                    mask = mask | filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
            display_df = filtered_df[mask]
        else:
            display_df = filtered_df
            
        # Afficher les données avec pagination
        st.dataframe(
            display_df[display_cols].reset_index(drop=True),
            height=400,
            use_container_width=True
        )
        
        # Informations sur le résultat
        st.info(f"Affichage de {len(display_df)} tweets sur {len(filtered_df)} filtrés (total: {len(df)})")

# Fonction principale
def main():
    st.title("📊 Tableau de Bord - Analyse des Tweets Engie")
    
    # Ajouter une description
    st.markdown("""
    Ce tableau de bord interactif vous permet d'analyser les tweets mentionnant Engie. 
    Utilisez les filtres dans la barre latérale pour affiner votre analyse.
    """)
    
    # Charger le fichier CSV
    uploaded_file = st.sidebar.file_uploader("💾 Charger un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner('Chargement et analyse des données...'):
            df = load_data(uploaded_file)
            
        if df is not None:
            # Affiche les informations sur le jeu de données
            st.sidebar.success(f"✅ Fichier chargé : {len(df):,} tweets de {df['Date_de_publication'].min().date()} à {df['Date_de_publication'].max().date()}")
            
            # Affiche les KPI
            display_kpis(df)
            
            # Affiche les graphiques
            display_charts(df)
    else:
        # Écran d'accueil
        st.info("👈 Veuillez charger un fichier CSV pour commencer l'analyse")
        
        # Exemple de format attendu
        st.markdown("""
        ### Format de données attendu
        Le fichier CSV doit contenir les colonnes suivantes :
        - `Date_de_publication` : date de publication du tweet
        - `type` : catégorie du problème
        - `Sentiment` : sentiment détecté (positif, neutre, négatif)
        - `incomfort` : score d'inconfort (%)
        - `Urgence` : indicateur d'urgence (True/False)
        - `Probleme` : liste des problèmes identifiés
        - `mot-critique` : liste des mots critiques
        """)

if __name__ == "__main__":
    main()