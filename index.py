import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="NBA Stats Explorer", layout="wide")

# Header and Introduction
st.markdown("""
# Project Title: NBA Player Stats Explorer App  
---

### Group Members:
👩‍💻 **Maria Anees**  
📚 Registration Number: **[21-NTU-CS-1336]**  

👨‍💻 **Muhammad Hanzala**  
📚 Registration Number: **[21-NTU-CS-1352]**  

👩‍💻 **Minahil Ismail**  
📚 Registration Number: **[21-NTU-CS-1339]**
""")

# Load Data
@st.cache_data
def load_data(year):
    try:
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
        html = pd.read_html(url, header=0)
        df = html[0]
        raw = df.drop(df[df.Age == 'Age'].index)
        raw = raw.fillna(0)
        playerstats = raw.drop(['Rk'], axis=1)
        playerstats = playerstats[playerstats['Team'] != 0]
        playerstats['Awards'] = playerstats['Awards'].astype(str)
        return playerstats
    except Exception as e:
        st.error(f"Error loading data for year {year}: {e}")
        return pd.DataFrame()

# AI/ML Features
class NBAInsightsAI:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
        self.feature_cols = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                           '3PA', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']
        
    def prepare_features(self, data):
        return self.scaler.fit_transform(data[self.feature_cols])

    def find_similar_players(self, player_name, n_neighbors=5):
        X = self.prepare_features(self.df)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
        nbrs.fit(X)
        
        player_idx = self.df[self.df['Player'] == player_name].index[0]
        distances, indices = nbrs.kneighbors(X[player_idx].reshape(1, -1))
        
        similar_players = self.df.iloc[indices[0][1:]]
        similarity_scores = 1 / (1 + distances[0][1:])
        
        return pd.DataFrame({
            'Player': similar_players['Player'],
            'Similarity Score': similarity_scores,
            'Position': similar_players['Pos']
        })

    def predict_future_performance(self, player_data, seasons_ahead=1):
        X = self.prepare_features(self.df)
        
        predictions = {}
        for stat in ['PTS', 'AST', 'TRB']:
            y = self.df[stat]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            player_features = self.prepare_features(player_data)
            pred = model.predict(player_features)
            predictions[stat] = pred[0]
            
        return predictions

# Main App
def main():
    # Sidebar controls
    st.sidebar.header('User Input Features')
    selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2025))))

    # Load data
    try:
        playerstats = load_data(selected_year)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Team and position selection
    sorted_unique_team = sorted(playerstats.Team.unique())
    selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team[:4])
    
    unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
    selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos[:3])

    # Filter data
    df_selected_team = playerstats[(playerstats.Team.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

    # Basic Stats Section
    st.header('📊 Basic Player Stats')
    tab1, tab2 = st.tabs(["Player Stats", "AI Features"])

    with tab1:
        st.dataframe(df_selected_team, use_container_width=True)
        st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)
        
        create_basic_visualizations(df_selected_team)

    with tab2:
        if not df_selected_team.empty:
            add_ai_features_ui(df_selected_team)
        else:
            st.warning("Please select teams and positions to enable AI features.")

def create_basic_visualizations(df_selected_team):
    if not df_selected_team.empty:
        # Scoring Efficiency
        st.header('🎯 Scoring Efficiency Analysis')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_selected_team, x='FGA', y='PTS', hue='Pos', ax=ax)
        st.pyplot(fig)

        # Assist-to-Turnover Ratio Analysis
        st.header('🏀 Assist-to-Turnover Ratio Analysis')
        st.write("Assessing playmaking efficiency through the Assist-to-Turnover Ratio (AST/TOV).")
        df_selected_team = df_selected_team.copy()
        if 'TOV' in df_selected_team.columns and 'AST' in df_selected_team.columns:
            df_selected_team['AST_to_TOV'] = df_selected_team['AST'] / (df_selected_team['TOV'] + 1e-9)
            top_ast_tov = df_selected_team[['Player', 'AST_to_TOV']].sort_values(by='AST_to_TOV', ascending=False).head(10)
            st.write("Top 10 Players by Assist-to-Turnover Ratio:")
            st.dataframe(top_ast_tov, use_container_width=True)
        else:
            st.warning("Required columns 'TOV' or 'AST' are missing in the dataset.")


        # Free Throw Efficiency Analysis
        st.header('⚡ Free Throw Efficiency Analysis')
        st.write("Comparing Free Throw Percentage (FT%) with Free Throw Attempts (FTA).")
        if not df_selected_team.empty:
            if 'FT%' in df_selected_team.columns and 'FTA' in df_selected_team.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                df_selected_team['FT%'] = pd.to_numeric(df_selected_team['FT%'], errors='coerce')
                sns.scatterplot(data=df_selected_team, x='FTA', y='FT%', hue='Pos', ax=ax)
                ax.set_title("Free Throw Efficiency: FT% vs. FTA")
                ax.set_xlabel("Free Throw Attempts (FTA)")
                ax.set_ylabel("Free Throw Percentage (FT%)")
                st.pyplot(fig)
            else:
                st.warning("Required columns 'FT%' or 'FTA' are missing in the dataset.")

        # Role-Based Analysis: Starters vs. Bench Players
        st.header('🧑‍🤝‍🧑 Starters vs. Bench Players Analysis')
        st.write("Comparing performance metrics between starters and bench players.")
        df_selected_team = df_selected_team.copy()
        if 'GS' in df_selected_team.columns and 'PTS' in df_selected_team.columns:
            df_selected_team['Role'] = df_selected_team['GS'].apply(lambda x: 'Starter' if x > 0 else 'Bench')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_selected_team, x='Role', y='PTS', ax=ax)
            ax.set_title("Points Scored: Starters vs. Bench Players")
            ax.set_xlabel("Role")
            ax.set_ylabel("Points Scored (PTS)")
            st.pyplot(fig)
        else:
            st.warning("Required columns 'GS' are missing in the dataset.")

        # Shooting Efficiency Analysis
        st.header('🤾‍♂️ Shooting Efficiency by Position')
        st.write("Exploring the shooting efficiency (Field Goal Percentage) across different positions.")
        if 'FG%' in df_selected_team.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_selected_team, x='Pos', y='FG%', ax=ax)
            ax.set_title("Field Goal Percentage by Position")
            ax.set_xlabel("Position")
            ax.set_ylabel("Field Goal Percentage (FG%)")
            st.pyplot(fig)
        else:
            st.warning("Required column 'FG%' is missing in the dataset.")

        # Rebounding Analysis
        st.header('⛹️‍♂️ Rebounding Analysis by Position')
        st.write("Analyzing the total rebounds (TRB) by position.")
        if 'TRB' in df_selected_team.columns and 'Pos' in df_selected_team.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_selected_team, x='Pos', y='TRB', ax=ax)
            ax.set_title("Total Rebounds by Position")
            ax.set_xlabel("Position")
            ax.set_ylabel("Total Rebounds (TRB)")
            st.pyplot(fig)
        else:
            st.warning("Required column 'TRB' is missing in the dataset.")


        # Heatmap of Intercorrelation Matrix
        st.header("🔥 Intercorrelation Matrix Heatmap")
        numeric_df = df_selected_team.select_dtypes(include=['float64', 'int64']) 
        if not numeric_df.empty:
            corr = numeric_df.corr() 
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            fig, ax = plt.subplots(figsize=(10, 8))
            with sns.axes_style("white"):
                sns.heatmap(corr, mask=mask, vmax=1, square=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found in the dataset for correlation analysis.")

        # Analyzing Scores and Awards
        st.header("🏆 Awards and Performance")
        awards_analysis = df_selected_team.groupby('Player')[['PTS', 'Awards']].sum().sort_values(by='PTS', ascending=False)
        st.dataframe(awards_analysis, use_container_width=True)

        # Scoring Efficiency (Points per Minute)
        st.header("⛹️‍♀️ Scoring Efficiency (Points per Minute)")
        if 'TRB' in df_selected_team.columns and 'Pos' in df_selected_team.columns:
            df_selected_team['PTS_per_Min'] = df_selected_team['PTS'] / df_selected_team['MP']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_selected_team, x='MP', y='PTS', hue='Pos', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Required columns 'PTS' or 'MP' are missing in the dataset.")

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

def add_ai_features_ui(df):
    st.header("🤖 AI-Powered Insights")

    ai_tab1, ai_tab2 = st.tabs(["Player Similarity", "Performance Prediction"])

    # Initialize AI models
    try:
        nba_ai = NBAInsightsAI(df)
        
    except Exception as e:
        st.error("Error initializing AI models. Please check the data or try again later.")
        return

    with ai_tab1:
        try:
            st.subheader("Find Similar Players")
            selected_player = st.selectbox("Select a player:", df['Player'].unique())
            if selected_player:
                similar_players = nba_ai.find_similar_players(selected_player)
                if similar_players.empty:
                    st.warning(f"No similar players found for {selected_player}.")
                else:
                    fig = px.bar(similar_players,
                                 x='Player',
                                 y='Similarity Score',
                                 color='Position',
                                 title=f"Players Similar to {selected_player}")
                    st.plotly_chart(fig)
                    st.dataframe(similar_players)
        except Exception as e:
            st.error("No similar players found.")

    with ai_tab2:
        try:
            st.subheader("Performance Prediction")
            pred_player = st.selectbox("Select player for prediction:", df['Player'].unique(), key='pred_player')
            seasons = st.slider("Seasons ahead:", 1, 3, 1)

            if pred_player:
                player_data = df[df['Player'] == pred_player]
                if player_data.empty:
                    st.warning(f"No data found for {pred_player}.")
                else:
                    predictions = nba_ai.predict_future_performance(player_data, seasons)
                    if predictions:
                        categories = ['Points', 'Assists', 'Rebounds']
                        current_stats = [
                            player_data['PTS'].iloc[0],
                            player_data['AST'].iloc[0],
                            player_data['TRB'].iloc[0]
                        ]
                        predicted_stats = [predictions['PTS'], predictions['AST'], predictions['TRB']]

                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=current_stats,
                            theta=categories,
                            fill='toself',
                            name='Current Stats'
                        ))
                        fig.add_trace(go.Scatterpolar(
                            r=predicted_stats,
                            theta=categories,
                            fill='toself',
                            name=f'Predicted Stats ({seasons} seasons ahead)'
                        ))

                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, max(max(current_stats), max(predicted_stats))])),
                            showlegend=True,
                            title=f"Performance Prediction for {pred_player}"
                        )

                        st.plotly_chart(fig)
                    else:
                        st.warning("Unable to generate predictions for the selected player.")
        except Exception as e:
            st.error("Error predicting player performance. Please try again.")

if __name__ == "__main__":
    main()