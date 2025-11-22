from flask import Flask, request, jsonify, render_template
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor



app = Flask(__name__)

rf_model = None
nba_data = None

def initialize_model():
    global rf_model, nba_data
    print("Connecting to database and training model...")

    try:
        conn = sqlite3.connect('nba.db')

        df = pd.read_sql("SELECT * FROM featured_games", conn)
        print(f"Successfully loaded {len(df)} rows from database.")
    except Exception as e:
        print(f"Error loading from databse: {e}")
        return
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    if 'OPP_DEF_RATING' not in df.columns:
        print("Opponent features missing. Re-calculating on the fly...")
        numeric_cols = ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'MIN']
        team_stats = df.groupby(['Game_ID', 'TEAM', 'GAME_DATE'])[numeric_cols].sum().reset_index()
        team_stats['POSS'] = 0.96 * (team_stats['FGA'] + 0.44 * team_stats['FTA'] - team_stats['OREB'] + team_stats['TOV'])
        
        team_stats_opp = team_stats[['Game_ID', 'TEAM', 'PTS', 'POSS']].copy()
        team_stats_opp.columns = ['Game_ID', 'OPP_TEAM', 'OPP_PTS_SCORED', 'OPP_POSS']
        merged_stats = pd.merge(team_stats, team_stats_opp, on='Game_ID')
        merged_stats = merged_stats[merged_stats['TEAM'] != merged_stats['OPP_TEAM']]
        
        merged_stats['DEF_RATING'] = 100 * (merged_stats['OPP_PTS_SCORED'] / merged_stats['POSS'])
        merged_stats['PTS_ALLOWED'] = merged_stats['OPP_PTS_SCORED']
        
        merged_stats = merged_stats.sort_values(['TEAM', 'GAME_DATE'])
        grp_team = merged_stats.groupby('TEAM')
        merged_stats['AVG_DEF_RATING'] = grp_team['DEF_RATING'].transform(lambda x: x.expanding().mean().shift(1))
        merged_stats['AVG_PTS_ALLOWED'] = grp_team['PTS_ALLOWED'].transform(lambda x: x.expanding().mean().shift(1))
        
        opp_stats = merged_stats[['Game_ID', 'TEAM', 'AVG_DEF_RATING', 'AVG_PTS_ALLOWED']].rename(columns={
            'TEAM': 'OPPONENT', 'AVG_DEF_RATING': 'OPP_DEF_RATING', 'AVG_PTS_ALLOWED': 'OPP_PTS_ALLOWED'
        })
        df = pd.merge(df, opp_stats, on=['Game_ID', 'OPPONENT'], how='left')

    # Fill NaNs
    if 'OPP_DEF_RATING' in df.columns:
        df['OPP_DEF_RATING'] = df['OPP_DEF_RATING'].fillna(df['OPP_DEF_RATING'].mean())
        df['OPP_PTS_ALLOWED'] = df['OPP_PTS_ALLOWED'].fillna(df['OPP_PTS_ALLOWED'].mean())

    # 3. Train Model
    features = ['AVG_PTS_LAST_3', 'AVG_PTS_LAST_5', 'AVG_MIN_LAST_3', 'STREAK', 'REST_DAYS', 'HOME', 'OPP_DEF_RATING', 'OPP_PTS_ALLOWED']
    
    # Drop rows where we don't have enough history to train
    train_df = df.dropna(subset=features + ['PTS'])
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(train_df[features], train_df['PTS'])
    
    rf_model = model
    nba_data = df
    print("Model trained successfully!")

#Run Setup
initialize_model()

def get_player_id(name):
    player_list = players.get_players()
    for p in player_list:
        if p['full_name'].lower() == name.lower():
            return p['id']
    return None

@app.route('/api/predict', methods=['POST'])
def predict():
    if rf_model is None:
        return jsonify({"error: Model failed to load. Check server logs"}), 500
    
    data = request.get_json()
    player_name = data.get('player')
    matchup_str = data.get('matchup')
    try:
        point_spread = float(data.get('spread'))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid spread value"}), 400

    # 1. Parse Matchup
    if ' vs. ' in matchup_str:
        team, opp = matchup_str.split(' vs. ')
        home = 1
    elif ' @ ' in matchup_str:
        team, opp = matchup_str.split(' @ ')
        home = 0
    else:
        return jsonify({"error": "Invalid Matchup Format. Use 'Team vs. Opp' or 'Team @ Opp'"}), 400
    # 2. Get Player History
    player_hist = nba_data[nba_data['Player'] == player_name].sort_values('GAME_DATE')
    if player_hist.empty:
        return jsonify({"error": f"Player {player_name} not found in database"}), 404

    # 3. Build Input
    # Use tail(3)/tail(5) to get the very latest stats available in the DB
    input_row = pd.DataFrame([{
        'AVG_PTS_LAST_3': player_hist['PTS'].tail(3).mean(),
        'AVG_PTS_LAST_5': player_hist['PTS'].tail(5).mean(),
        'AVG_MIN_LAST_3': player_hist['MIN'].tail(3).mean(),
        'STREAK': player_hist['STREAK'].iloc[-1],
        'REST_DAYS': 1,
        'HOME': home,
        'OPP_DEF_RATING': nba_data[nba_data['OPPONENT'] == opp]['OPP_DEF_RATING'].iloc[-1] if opp in nba_data['OPPONENT'].values else nba_data['OPP_DEF_RATING'].mean(),
        'OPP_PTS_ALLOWED': nba_data[nba_data['OPPONENT'] == opp]['OPP_PTS_ALLOWED'].iloc[-1] if opp in nba_data['OPPONENT'].values else nba_data['OPP_PTS_ALLOWED'].mean()
    }])

    # 4. Predict
    pred_pts = rf_model.predict(input_row)[0]
    
    # 5. Return Result
    return jsonify({
        "player": player_name,
        "spread": point_spread,
        "projected_points": f"{pred_pts:.1f}",
        "pick": "OVER" if pred_pts > point_spread else "UNDER",
        "edge": f"{abs(pred_pts - point_spread):.1f}",
        "confidence_note": "Based on historical regression model"
    })

@app.route('/api/player/<player_name>')
def player_stats(player_name):
    player_id = get_player_id(player_name)

    if player_id is None:
        return jsonify({"error": "Player not found"}), 404
    logs = playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
    df = logs.get_data_frames()[0]

    last_game = df.iloc[0]
    
    stats = {
        "player": str(player_name),
        "pts": int(last_game["PTS"]),
        "reb": int(last_game["REB"]),
        "ast": int(last_game["AST"]),
        "min": str(last_game["MIN"]),       # minutes is usually a string like "37"
        "fg_pct": float(last_game["FG_PCT"]),
        "opp": str(last_game["MATCHUP"])
    }

    return jsonify(stats)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/api/players')
def get_players():
    if nba_data is None:
        return jsonify([])
    try:
        players_list = sorted(nba_data['Player'].dropna().unique().tolist())
        return jsonify(players_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)