import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from datetime import timedelta
from client_mistral import MistralAgent
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

# Fonction pour exécuter l'analyse et mettre en cache les résultats
@st.cache_data
def run_mistral_analysis(file_path):
    try:
        agent = MistralAgent()
        response = agent.process_file(file_path)
        st.write(f"Débogage : Résultat de l'analyse = {response[:100]}...")  # Débogage
        return response
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de l'analyse : {str(e)}")
        return None

# Fonction pour générer un PDF à partir du texte
def create_pdf(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    for line in text.split('\n'):
        if line.startswith('# '):
            story.append(Paragraph(line.strip('# ').strip(), styles['Heading1']))
        elif line.startswith('## '):
            story.append(Paragraph(line.strip('## ').strip(), styles['Heading2']))
        elif line.strip():
            story.append(Paragraph(line, styles['BodyText']))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse des Tweets Engie",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appliquer CSS personnalisé
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: darkblue;
        border-radius: 4px 4px 0 0; gap: 1px; padding: 10px 16px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background-color: #4F8BF9; color: white; }
    h1, h2, h3 { color: #1E3A8A; }
    .stPlotlyChart {
        background-color: black; border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); padding: 10px; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Charger les données avec mise en cache
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file, sep=';', quotechar='"', encoding='utf-8', on_bad_lines='warn')
        df['Date_de_publication'] = pd.to_datetime(df['Date_de_publication'], errors='coerce')
        
        def safe_literal_eval(x):
            try:
                return ast.literal_eval(x) if pd.notna(x) else []
            except (ValueError, SyntaxError):
                return []
        
        for col in ['Probleme', 'mot-critique']:
            if col in df.columns:
                df[col] = df[col].apply(safe_literal_eval)
        
        if 'incomfort' in df.columns:
            df['incomfort'] = pd.to_numeric(df['incomfort'], errors='coerce')
        
        if 'Urgence' in df.columns:
            df['Urgence'] = df['Urgence'].astype(str).str.lower() == 'true'
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier CSV : {str(e)}")
        return None

# Fonction pour afficher les KPI
def display_kpis(df):
    st.subheader("📈 Indicateurs clés de performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total tweets", f"{len(df):,}", delta=None)
    with col2:
        tweets_per_day = round(len(df) / max(df['Date_de_publication'].dt.date.nunique(), 1), 1)
        st.metric("Tweets par jour", f"{tweets_per_day:,}", delta=None)
    with col3:
        if 'Urgence' in df.columns:
            urgent_pct = df['Urgence'].mean() * 100
            st.metric("Tweets urgents", f"{urgent_pct:.1f}%", delta=None)
        else:
            st.metric("Tweets urgents", "N/A")
    with col4:
        if 'incomfort' in df.columns:
            incomfort_avg = df['incomfort'].mean()
            st.metric("Score d'inconfort", f"{incomfort_avg:.1f}%", delta=None)
        else:
            st.metric("Score d'inconfort", "N/A")

# Fonction pour les graphiques et recommandations
def display_charts(df, uploaded_file):
    tabs = st.tabs(["Chronologie", "Sentiments", "Types de problèmes", "Mots clés", "Données brutes", "Recommandations"])
    
    min_date = pd.Timestamp(df['Date_de_publication'].min()).tz_localize(None)
    max_date = pd.Timestamp(df['Date_de_publication'].max()).tz_localize(None)
    
    with st.sidebar:
        st.header("Filtres")
        period_options = {
            "Dernière semaine": 7, "Dernier mois": 30, "Dernier trimestre": 90,
            "Dernière année": 365, "Tout": None
        }
        selected_period = st.selectbox("Période d'analyse", options=list(period_options.keys()), index=1)
        
        if 'type' in df.columns:
            types = sorted(df['type'].dropna().unique())
            selected_types = st.multiselect("Types de problèmes", options=types, default=types[:5] if len(types) > 5 else types)
        else:
            selected_types = []
            
        if 'Sentiment' in df.columns:
            sentiments = df['Sentiment'].dropna().unique()
            selected_sentiments = st.multiselect("Sentiments", options=sentiments, default=sentiments)
        else:
            selected_sentiments = []
            
        if st.button("Réinitialiser les filtres"):
            pass

    filtered_df = df.copy()
    filtered_df['Date_de_publication'] = filtered_df['Date_de_publication'].dt.tz_localize(None)
    
    if selected_period != "Tout" and period_options[selected_period]:
        days = period_options[selected_period]
        start_date = pd.Timestamp(max_date - timedelta(days=days)).replace(hour=0, minute=0, second=0)
        end_date = pd.Timestamp(max_date).replace(hour=23, minute=59, second=59)
    else:
        start_date = min_date
        end_date = max_date
    
    filtered_df = filtered_df[
        (filtered_df['Date_de_publication'] >= start_date) & 
        (filtered_df['Date_de_publication'] <= end_date)
    ]
    
    if selected_types and 'type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
    if selected_sentiments and 'Sentiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiments)]
    
    if filtered_df.empty:
        st.warning(f"Aucune donnée pour la période du {start_date.strftime('%d/%m/%Y %H:%M')} au {end_date.strftime('%d/%m/%Y %H:%M')}")
        return None

    # Onglet 1: Chronologie (inchangé)
    with tabs[0]:
        st.subheader("Évolution temporelle des tweets")
        tweets_per_day = filtered_df.groupby(filtered_df['Date_de_publication'].dt.date).agg({
            'Date_de_publication': 'count',
            'incomfort': 'mean' if 'incomfort' in filtered_df.columns else 'size',
            'Urgence': 'mean' if 'Urgence' in filtered_df.columns else 'size'
        }).rename(columns={'Date_de_publication': 'count'}).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tweets_per_day['Date_de_publication'], y=tweets_per_day['count'], mode='lines+markers', name='Nombre de tweets', line=dict(color='#4F8BF9', width=3), marker=dict(size=8)))
        if 'incomfort' in tweets_per_day.columns:
            fig.add_trace(go.Scatter(x=tweets_per_day['Date_de_publication'], y=tweets_per_day['incomfort'], mode='lines', name="Score d'inconfort", yaxis='y2', line=dict(color='#FF5757', width=2, dash='dot')))
        fig.update_layout(title='Évolution du volume de tweets et du score d\'inconfort', xaxis_title='Date', yaxis_title='Nombre de tweets', yaxis2=dict(title="Score d'inconfort", overlaying='y', side='right', showgrid=False), hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    # Onglet 2: Sentiments (inchangé)
    with tabs[1]:
        if 'Sentiment' in filtered_df.columns:
            st.subheader("Analyse des sentiments")
            sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'count']
            total = sentiment_counts['count'].sum()
            sentiment_counts['percentage'] = (sentiment_counts['count'] / total * 100).round(1)
            colors = {'positif': '#36A2EB', 'neutre': '#FFCE56', 'négatif': '#FF6384', 'positive': '#36A2EB', 'neutral': '#FFCE56', 'negative': '#FF6384'}
            color_values = [colors.get(s.lower(), '#A9A9A9') for s in sentiment_counts['Sentiment']]
            fig = go.Figure(data=[go.Pie(labels=sentiment_counts['Sentiment'], values=sentiment_counts['count'], hole=.4, marker_colors=color_values, textinfo='label+percent', textposition='outside', pull=[0.05 if s.lower() in ['positif', 'positive'] else 0 for s in sentiment_counts['Sentiment']])])
            fig.update_layout(title='Répartition des sentiments', annotations=[dict(text=f'{total} tweets', x=0.5, y=0.5, font_size=20, showarrow=False)], height=450, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Les données de sentiment ne sont pas disponibles.")

    # Onglet 3: Types de problèmes (inchangé)
    with tabs[2]:
        st.subheader("Analyse des types de problèmes")
        type_analysis = filtered_df['type'].value_counts().reset_index()
        type_analysis.columns = ['type', 'count']
        type_analysis = type_analysis.dropna()
        type_analysis['count'] = type_analysis['count'].fillna(0).astype(int)
        type_analysis['size'] = type_analysis['count'].apply(lambda x: max(x, 1))
        if not type_analysis.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=type_analysis['count'], y=type_analysis['type'], mode='markers', marker=dict(size=type_analysis['size'], sizemode='area', sizeref=2.*max(type_analysis['size'])/(40.**2), sizemin=4), text=type_analysis['type'] + ': ' + type_analysis['count'].astype(str) + ' occurrences', hoverinfo='text'))
            fig.update_layout(title="Répartition des types de problèmes", xaxis_title="Nombre d'occurrences", yaxis_title="Type de problème", height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour les types de problèmes.")

    # Onglet 4: Mots clés (inchangé)
    with tabs[3]:
        if 'mot-critique' in filtered_df.columns:
            st.subheader("Analyse des mots clés critiques")
            all_mot_critique = [word for sublist in filtered_df['mot-critique'] for word in sublist if word]
            if all_mot_critique:
                from collections import Counter
                word_counts = Counter(all_mot_critique)
                top_words = dict(word_counts.most_common(50))
                word_cols = st.columns([3, 2])
                with word_cols[0]:
                    wordcloud = WordCloud(width=800, height=500, background_color='white', colormap='viridis', max_words=100, contour_width=1, contour_color='steelblue').generate_from_frequencies(top_words)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                with word_cols[1]:
                    st.markdown("### Mots les plus fréquents")
                    top_20_words = dict(word_counts.most_common(20))
                    word_df = pd.DataFrame({'mot': list(top_20_words.keys()), 'fréquence': list(top_20_words.values())}).sort_values('fréquence', ascending=True)
                    fig = px.bar(word_df, x='fréquence', y='mot', orientation='h', title='Top 20 des mots critiques', color='fréquence', color_continuous_scale='Viridis')
                    fig.update_layout(height=500, xaxis_title='Fréquence', yaxis_title='Mot', coloraxis_showscale=False, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucun mot critique trouvé.")
        else:
            st.info("Les données de mots critiques ne sont pas disponibles.")

    # Onglet 5: Données brutes (inchangé)
    with tabs[4]:
        st.subheader("Données brutes filtrées")
        csv = filtered_df.to_csv(index=False, sep=';')
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="tweets_engie_filtered.csv" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4F8BF9; color: white; text-decoration: none; border-radius: 4px; margin-bottom: 1rem;">Télécharger les données filtrées (CSV)</a>', unsafe_allow_html=True)
        
        search_term = st.text_input("Rechercher dans les données :")
        all_columns = filtered_df.columns.tolist()
        default_display = [c for c in ['Date_de_publication', 'type', 'Sentiment', 'incomfort', 'Urgence'] if c in all_columns]
        display_cols = st.multiselect("Colonnes à afficher :", options=all_columns, default=default_display if default_display else all_columns[:5])
        if not display_cols:
            display_cols = all_columns[:5]
        if search_term:
            mask = pd.Series(False, index=filtered_df.index)
            for col in filtered_df.columns:
                if filtered_df[col].dtype == object:
                    mask = mask | filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
            display_df = filtered_df[mask]
        else:
            display_df = filtered_df
        st.dataframe(display_df[display_cols].reset_index(drop=True), height=400, use_container_width=True)
        st.info(f"Affichage de {len(display_df)} tweets sur {len(filtered_df)} filtrés (total: {len(df)})")

    # Onglet 6: Recommandations
    with tabs[5]:
        st.subheader("Recommandations et Préconisations")
        
        # Initialiser session_state pour les recommandations si non défini
        if 'recommendations' not in st.session_state:
            st.session_state['recommendations'] = None
        
        analysis_scope = st.radio(
            "Portée de l'analyse :",
            options=["Fichier complet", "Données filtrées"],
            help="Choisissez si l'analyse doit porter sur l'ensemble du fichier ou uniquement sur les données filtrées selon les critères actuels."
        )
        
        if st.button("Lancer l'analyse"):
            with st.spinner('Analyse en cours (cela peut prendre plusieurs minutes)...'):
                if analysis_scope == "Fichier complet":
                    st.session_state['recommendations'] = run_mistral_analysis(uploaded_file)
                else:
                    temp_csv = "temp_filtered_data.csv"
                    filtered_df.to_csv(temp_csv, sep=';', index=False)
                    st.session_state['recommendations'] = run_mistral_analysis(temp_csv)
                    if os.path.exists(temp_csv):
                        os.remove(temp_csv)
        
        # Afficher les recommandations si elles existent dans session_state
        if st.session_state['recommendations']:
            st.markdown(
                """
                <div style='background-color: transparent; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                    <h3 style='color: white;'>Résultats de l'analyse</h3>
                    <pre style='white-space: pre-wrap; color: white;'>{}</pre>
                </div>
                """.format(st.session_state['recommendations']),
                unsafe_allow_html=True
            )
            
            # Champ pour spécifier un nom de fichier personnalisé
            file_name = st.text_input("Nom du fichier (laissez vide pour 'recommandations')", "")
            if not file_name.strip():
                file_name = "recommandations"
            # Nettoyer le nom pour éviter des caractères invalides
            file_name = "".join(c for c in file_name if c.isalnum() or c in " _-").strip()
            
            # Sélection du format de téléchargement
            download_format = st.selectbox(
                "Choisir le format de téléchargement :",
                options=["Markdown (.md)", "PDF (.pdf)", "Texte (.txt)"],
                index=0,
                key="download_format"
            )
            
            if download_format == "Markdown (.md)":
                b64 = base64.b64encode(st.session_state['recommendations'].encode()).decode()
                st.markdown(
                    f'<a href="data:text/markdown;base64,{b64}" download="{file_name}.md" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4F8BF9; color: white; text-decoration: none; border-radius: 4px; margin-top: 1rem;">Télécharger en Markdown</a>',
                    unsafe_allow_html=True
                )
            
            elif download_format == "PDF (.pdf)":
                pdf_data = create_pdf(st.session_state['recommendations'])
                b64 = base64.b64encode(pdf_data).decode()
                st.markdown(
                    f'<a href="data:application/pdf;base64,{b64}" download="{file_name}.pdf" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4F8BF9; color: white; text-decoration: none; border-radius: 4px; margin-top: 1rem;">Télécharger en PDF</a>',
                    unsafe_allow_html=True
                )
            
            elif download_format == "Texte (.txt)":
                b64 = base64.b64encode(st.session_state['recommendations'].encode()).decode()
                st.markdown(
                    f'<a href="data:text/plain;base64,{b64}" download="{file_name}.txt" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4F8BF9; color: white; text-decoration: none; border-radius: 4px; margin-top: 1rem;">Télécharger en Texte</a>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Aucune analyse n'a été effectuée. Cliquez sur 'Lancer l'analyse' pour générer des recommandations.")
    
    return st.session_state.get('recommendations')

# Fonction principale
def main():
    st.title("📊 Tableau de Bord - Analyse des Tweets Engie")
    st.markdown("""
    Ce tableau de bord interactif vous permet d'analyser les tweets mentionnant Engie. 
    Utilisez les filtres dans la barre latérale pour affiner votre analyse.
    """)
    
    uploaded_file = st.sidebar.file_uploader("💾 Charger un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner('Chargement et analyse des données...'):
            df = load_data(uploaded_file)
        if df is not None:
            st.sidebar.success(f"✅ Fichier chargé : {len(df):,} tweets de {df['Date_de_publication'].min().date()} à {df['Date_de_publication'].max().date()}")
            display_kpis(df)
            display_charts(df, uploaded_file)
    else:
        st.info("👈 Veuillez charger un fichier CSV pour commencer l'analyse")
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