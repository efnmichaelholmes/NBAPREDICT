import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    leaguegamefinder, playergamelog, teamdashboardbygeneralsplits, 
    shotchartdetail, commonplayerinfo
)

st.set_page_config(page_title="NBA Player Performance Predictor 🏀 Complete",
                   page_icon="🏀", layout="wide", initial_sidebar_state="expanded")

# Load your trained models and encoders
@st.cache_resource
def load_models():
    with open("trained_features.pkl", "rb") as f:
        trained_features = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("xgb_models.pkl", "rb") as f:
        xgb_models = pickle.load(f)
    try:
        with open("lgbm_models.pkl", "rb") as f:
            lgbm_models = pickle.load(f)
    except FileNotFoundError:
        lgbm_models = {}
    try:
        with open("rf_models.pkl", "rb") as f:
            rf_models = pickle.load(f)
    except FileNotFoundError:
        rf_models = {}
    return trained_features, label_encoders, scaler, xgb_models, lgbm_models, rf_models

trained_features, label_encoders, scaler, xgb_models, lgbm_models, rf_models = load_models()

def get_active_players(seasons=["2023-24", "2022-23"]):
    all_players = players.get_players()
    active_ids = set()
    for season in seasons:
        try:
            df = leaguegamefinder.LeagueGameFinder(season_nullable=season).get_data_frames()[0]
            if 'PLAYER_ID' in df.columns and not df.empty:
                active_ids.update(df['PLAYER_ID'].unique())
        except:
            pass
    if not active_ids:
        return sorted(p['full_name'] for p in all_players)
    return sorted(p['full_name'] for p in all_players if p['id'] in active_ids)


def fetch_player_game_logs(player_id, season="2023-24"):
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = logs.get_data_frames()[0]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df


def fetch_team_stats(team_name, season="2023-24"):
    nba_teams = teams.get_teams()
    team_info = next((t for t in nba_teams if t["full_name"] == team_name), None)
    if not team_info:
        return {}
    dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(team_id=team_info["id"], season=season)
    df_dash = dashboard.get_data_frames()[0]
    stats = {}
    if not df_dash.empty:
        stats["PACE"] = df_dash.get("PACE", [100])[0]
        stats["DEF_RATING"] = df_dash.get("DEF_RATING", [110])[0]
        stats["OPP_ORB_PCT"] = df_dash.get("OPP_OREB_PCT", [0.3])[0]
        stats["OPP_3PT_PCT"] = df_dash.get("OPP_3P_PCT", [0.35])[0]
    return stats


def encode_value(encoder, val):
    return encoder.transform([val])[0] if val in encoder.classes_ else -1


def get_player_id(player_name):
    if not player_name or player_name == "Select a player...":
        return None
    nba_players = players.get_players()
    player = next((p for p in nba_players if p["full_name"].lower() == player_name.lower()), None)
    return player["id"] if player else None


def nonlinear_def_adj(def_rating):
    base = 105
    if def_rating <= 0:
        return 1
    return (105 / def_rating) ** 1.15


def aggregate_opponent_multiplier(player_df, opp_games, team_stats):
    if opp_games.empty or len(opp_games) < 2:
        return {stat: 1.0 for stat in ["PTS", "REB", "AST", "FG3M", "BLK", "STL"]}
    ratios = {}
    for stat in ["PTS", "REB", "AST", "FG3M", "BLK", "STL"]:
        player_avg = max(player_df[stat].mean(), 1)
        opp_avg = opp_games[stat].mean()
        ratios[stat] = np.clip(opp_avg / player_avg, 0.7, 1.3)
    def_mult = nonlinear_def_adj(team_stats.get("DEF_RATING", 110))
    pace_factor = (team_stats.get("PACE", 100) / 100) ** 0.3
    for s in ratios:
        ratios[s] *= def_mult * pace_factor
    return ratios


def predict_player_performance(player_name, opponent_team, apply_adj=True, roll_window=5):
    targets = ["PTS", "REB", "AST", "FG3M", "BLK", "STL"]
    if not player_name or player_name == "Select a player...":
        st.warning("Please select a player")
        dummy = pd.DataFrame({"Stat": targets, "Prediction_Avg": [0] * 6, "Chance_Avg (%)": [0] * 6})
        return dummy, None
    player_id = get_player_id(player_name)
    if not player_id:
        st.warning(f"Player {player_name} not found")
        dummy = pd.DataFrame({"Stat": targets, "Prediction_Avg": [0] * 6, "Chance_Avg (%)": [0] * 6})
        return dummy, None
    player_data = fetch_player_game_logs(player_id)
    if player_data.empty:
        return pd.DataFrame({"Stat": targets, "Prediction_Avg": [0] * 6, "Chance_Avg (%)": [0] * 6}), None
    team_info = next((t for t in teams.get_teams() if t["full_name"] == opponent_team), None)
    if not team_info:
        st.warning(f"Opponent team {opponent_team} not found")
        return pd.DataFrame({"Stat": targets, "Prediction_Avg": [0] * 6, "Chance_Avg (%)": [0] * 6}), None
    opponent_abbr = team_info["abbreviation"]
    opp_filter = player_data["MATCHUP"].str.contains(opponent_abbr, case=False, na=False)
    opp_games = player_data[opp_filter].copy()
    team_stats = fetch_team_stats(opponent_team)
    recent_games = player_data.tail(roll_window)
    recent_games["HOME"] = recent_games["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
    home_factor = 1 + (recent_games["HOME"].mean() - 0.5) * 0.2
    weights = np.linspace(1, 3, len(recent_games))
    rolling_averages = {f"{s}_L5G": np.average(recent_games[s], weights=weights) * home_factor for s in targets}
    if apply_adj:
        match_multipliers = aggregate_opponent_multiplier(player_data, opp_games, team_stats)
        for s in targets:
            if f"{s}_L5G" in rolling_averages:
                rolling_averages[f"{s}_L5G"] *= match_multipliers[s]
    sample_input = pd.DataFrame(columns=trained_features)
    sample_input.loc[0] = 0
    if "TEAM" in label_encoders:
        sample_input.at[0, "TEAM"] = encode_value(label_encoders["TEAM"], opponent_team)
    for feat in ["DEF_RATING", "PACE", "OPP_ORB_PCT", "OPP_3PT_PCT"]:
        if feat in sample_input.columns:
            sample_input.at[0, feat] = team_stats.get(feat, 0)
    for col, val in rolling_averages.items():
        if col in sample_input.columns:
            sample_input.at[0, col] = val
    sample_input = sample_input[trained_features]
    input_scaled = scaler.transform(sample_input)
    preds = []
    for stat in targets:
        preds_list = []
        if stat in xgb_models:
            preds_list.append(xgb_models[stat].predict(input_scaled)[0])
        if stat in lgbm_models:
            preds_list.append(lgbm_models[stat].predict(input_scaled)[0])
        if stat in rf_models:
            preds_list.append(rf_models[stat].predict(input_scaled)[0])
        pred = np.mean(preds_list) if preds_list else 0
        preds.append(round(max(pred, 0), 1))
    predicted_df = pd.DataFrame({
        "Stat": targets,
        "Prediction_Avg": preds,
        "Chance_Avg (%)": [round(match_multipliers[s] * 100, 1) if apply_adj else 100 for s in targets]
    })
    return predicted_df, None


def get_player_averages(player_name):
    player_id = get_player_id(player_name)
    if not player_id:
        return pd.DataFrame({"Stat": ["PTS", "REB", "AST", "FG3M", "BLK", "STL"], "Average": [0] * 6})
    logs = fetch_player_game_logs(player_id)
    if logs.empty:
        return pd.DataFrame({"Stat": ["PTS", "REB", "AST", "FG3M", "BLK", "STL"], "Average": [0] * 6})
    return pd.DataFrame({
        "Stat": ["PTS", "REB", "AST", "FG3M", "BLK", "STL"],
        "Average": [round(logs[s].mean(), 1) for s in ["PTS", "REB", "AST", "FG3M", "BLK", "STL"]]
    })


def show_summary(pred_df, avg_df):
    cols = st.columns(len(pred_df))
    for i, col in enumerate(cols):
        stat = pred_df.iloc[i]["Stat"]
        pred_val = pred_df.iloc[i]["Prediction_Avg"]
        avg_val = avg_df[avg_df["Stat"] == stat]["Average"].values[0]
        progress = float(min(pred_val / max(avg_val, 1), 1.0))
        with col:
            st.metric(stat, f"{pred_val:.1f}", delta=f"{pred_val - avg_val:+.1f}")
            st.progress(progress)


def radar_chart(pred_df, avg_df):
    categories = pred_df["Stat"].tolist()
    pred_values = pred_df["Prediction_Avg"].tolist()
    avg_values = avg_df["Average"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pred_values, theta=categories, fill="toself",
                                  name="Prediction", fillcolor="rgba(224,58,62,0.25)",
                                  line=dict(color="#E03A3E")))
    fig.add_trace(go.Scatterpolar(r=avg_values, theta=categories, fill="toself",
                                  name="Recent Avg", fillcolor="rgba(11,61,145,0.25)",
                                  line=dict(color="#0B3D91")))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def compare_chart(avg_df, pred_df):
    comp_df = pd.merge(avg_df, pred_df, on="Stat")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp_df["Stat"], y=comp_df["Average"], mode="lines+markers",
                             name="Player Average", line=dict(color="#E03A3E")))
    fig.add_trace(go.Scatter(x=comp_df["Stat"], y=comp_df["Prediction_Avg"], mode="lines+markers",
                             name="Prediction", line=dict(color="#0B3D91")))
    fig.update_layout(title="Player Average vs Prediction", xaxis_title="Stat", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)


def get_player_current_team_id(player_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df_info = info.get_data_frames()[0]
        if 'TEAM_ID' in df_info.columns and 'TEAM_ABBREVIATION' in df_info.columns:
            team_id = df_info.loc[0, 'TEAM_ID']
            team_abbr = df_info.loc[0, 'TEAM_ABBREVIATION']
            return team_id, team_abbr
    except Exception as e:
        st.warning(f"Failed to get player's current team info: {e}")
    return None, None


def plot_shot_chart(player_id, season="2023-24", season_type="Regular Season"):
    team_id, team_abbr = get_player_current_team_id(player_id)
    if not team_id or not team_abbr:
        st.info("Could not determine player's current NBA team for shot chart.")
        return

    shotchart = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=player_id,
        season_nullable=season,
        season_type_all_star=season_type,
        context_measure_simple="FGA"
    )
    df_shots = shotchart.get_data_frames()[0]
    if df_shots.empty:
        st.info("No shot data available for this player and season.")
        return

    import matplotlib.patches as patches

    def draw_court(ax=None, color='black', lw=2, outer_lines=False):
        if ax is None:
            ax = plt.gca()
        hoop = patches.Circle((0, 0), 7.5, linewidth=lw, color=color, fill=False)
        backboard = plt.Line2D([-30, 30], [-7.5, -7.5], linewidth=lw, color=color)
        outer_box = patches.Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
        inner_box = patches.Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
        top_free_throw = patches.Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color)
        bottom_free_throw = patches.Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, linestyle='dashed', color=color)
        restricted = patches.Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
        corner_three_a = plt.Line2D([-220, -220], [-47.5, 92.5], linewidth=lw, color=color)
        corner_three_b = plt.Line2D([220, 220], [-47.5, 92.5], linewidth=lw, color=color)
        three_arc = patches.Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
        center_outer_arc = patches.Circle((0, 422.5), 60, linewidth=lw, color=color, fill=False)
        center_inner_arc = patches.Circle((0, 422.5), 20, linewidth=lw, color=color, fill=False)
        court_elements = [hoop, backboard, outer_box, inner_box,
                          top_free_throw, bottom_free_throw, restricted,
                          corner_three_a, corner_three_b, three_arc,
                          center_outer_arc, center_inner_arc]
        for element in court_elements:
            if isinstance(element, plt.Line2D):
                ax.add_line(element)
            else:
                ax.add_patch(element)
        if outer_lines:
            outer_lines_rect = patches.Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)
            ax.add_patch(outer_lines_rect)
        return ax

    made_shots = df_shots[df_shots['SHOT_MADE_FLAG'] == 1]
    missed_shots = df_shots[df_shots['SHOT_MADE_FLAG'] == 0]

    fig, ax = plt.subplots(figsize=(12, 11))
    ax = draw_court(ax, outer_lines=True)
    ax.scatter(made_shots["LOC_X"] / 10, made_shots["LOC_Y"] / 10,
               marker='o', color='green', label='Made Shots', alpha=0.7, edgecolors='black', s=60)
    ax.scatter(missed_shots["LOC_X"] / 10, missed_shots["LOC_Y"] / 10,
               marker='x', color='red', label='Missed Shots', alpha=0.7, s=60)
    ax.set_xlim(-250 / 10, 250 / 10)
    ax.set_ylim(-47.5 / 10, 422.5 / 10)
    ax.axis('off')
    ax.set_title("Shot Chart - Made vs Missed Shots")
    ax.legend(loc='upper right')
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 11))
    ax2 = draw_court(ax2, outer_lines=True)
    sns.kdeplot(x=df_shots["LOC_X"] / 10, y=df_shots["LOC_Y"] / 10,
                shade=True, shade_lowest=False, cmap="Reds", alpha=0.6, ax=ax2)
    ax2.set_xlim(-250 / 10, 250 / 10)
    ax2.set_ylim(-47.5 / 10, 422.5 / 10)
    ax2.axis('off')
    ax2.set_title("Shot Frequency Heatmap")
    st.pyplot(fig2)

# Sidebar UI
active_players = get_active_players()
active_players = ["Select a player..."] + active_players if active_players else ["Select a player..."]
teams_list = sorted([t["full_name"] for t in teams.get_teams()])

selected_player = st.sidebar.selectbox("Select Player:", active_players)
selected_team = st.sidebar.selectbox("Select Opponent Team:", teams_list)
opp_adjustment = st.sidebar.checkbox("Apply Opponent Adjustment", True)
rolling_window = st.sidebar.slider("Rolling Average Window (games)", 5, 20, 5)

st.title("🏀 NBA Player Performance Predictor Complete")

with st.spinner("Calculating predictions..."):
    predicted, _ = predict_player_performance(selected_player, selected_team, opp_adjustment, rolling_window)
average_df = get_player_averages(selected_player)

tabs = st.tabs([
    "Summary", "Radar Chart", "Detailed Stats",
    "Matchup History", "Opponent Info", "Shot Charts", "About"
])

with tabs[0]:
    st.header("Performance Summary")
    show_summary(predicted, average_df)

with tabs[1]:
    st.header("Radar Chart")
    radar_chart(predicted, average_df)

with tabs[2]:
    st.header("Detailed Stats Comparison")
    compare_chart(average_df, predicted)

with tabs[3]:
    st.header(f"{selected_player} vs. {selected_team} Historical Matchups")
    player_id = get_player_id(selected_player)
    if player_id and selected_team != "Select a player...":
        game_logs = fetch_player_game_logs(player_id)
        if not game_logs.empty:
            opp_abbr = next((t["abbreviation"] for t in teams.get_teams() if t["full_name"] == selected_team), "")
            opp_games = game_logs[game_logs["MATCHUP"].str.contains(opp_abbr, case=False, na=False)]
            if opp_games.empty:
                st.info("No historical games found vs this opponent.")
            else:
                cols_show = ['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL']
                st.dataframe(opp_games[cols_show].sort_values('GAME_DATE', ascending=False).style.format({"GAME_DATE": lambda d: d.strftime('%Y-%m-%d')}))
        else:
            st.info("No game log data available.")
    else:
        st.info("Select a player and opponent team to see matchup history.")

with tabs[4]:
    st.header(f"{selected_team} Recent Form & Defensive Profile")
    team_info = next((t for t in teams.get_teams() if t["full_name"] == selected_team), None)
    if team_info:
        try:
            team_games_df = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_info["id"], season_nullable="2023-24").get_data_frames()[0]
            last5 = team_games_df.sort_values('GAME_DATE', ascending=False).head(5)
            wins = sum(last5['WL'] == 'W')
            losses = sum(last5['WL'] == 'L')
            st.metric("Last 5 Games Record", f"{wins} - {losses}")

            # Dynamically choose columns to display
            display_cols = ['GAME_DATE', 'MATCHUP', 'WL', 'PTS']
            if 'PTS_OPP' in last5.columns:
                display_cols.append('PTS_OPP')

            st.write(last5[display_cols].style.format({"GAME_DATE": lambda d: pd.to_datetime(d).strftime('%Y-%m-%d')}))

            team_stats = fetch_team_stats(selected_team)
            if team_stats:
                st.markdown(f"**Defensive Rating:** {team_stats.get('DEF_RATING', 'N/A'):.2f}")
                st.markdown(f"**Pace:** {team_stats.get('PACE', 'N/A'):.2f}")
                st.markdown(f"**Opponent ORB %:** {team_stats.get('OPP_ORB_PCT', 'N/A'):.3f}")
                st.markdown(f"**Opponent 3PT %:** {team_stats.get('OPP_3PT_PCT', 'N/A'):.3f}")
            else:
                st.info("No defensive stats available.")
        except Exception as e:
            st.error(f"Error fetching team data: {e}")
    else:
        st.info("Select an opponent team to see team info.")

with tabs[5]:
    st.header(f"Shot Charts & Heatmaps for {selected_player}")
    player_id = get_player_id(selected_player)
    if player_id and selected_player != "Select a player...":
        plot_shot_chart(player_id)
    else:
        st.info("Select a player to view shot charts.")

with tabs[6]:
    st.header("About This App")
    st.markdown("""
    This NBA Player Performance Predictor integrates extensive NBA API data:
    - Opponent adjusted statistical predictions
    - Historical matchup logs
    - Opponent recent form and defensive analytics
    - Shot charts and heatmaps visualizing player shooting tendencies

    Built with Streamlit and nba_api Python package.
    """)
