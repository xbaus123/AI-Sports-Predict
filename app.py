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
MODEL_FEATURES = ['AVG_PTS_LAST_3', 'AVG_PTS_LAST_5', 'AVG_MIN_LAST_3', 'STREAK', 'REST_DAYS', 'HOME', 'OPP_DEF_RATING', 'OPP_PTS_ALLOWED']
def initialize_model():
    global rf_model, nba_data
    print("Connecting to database and training model...")

    try:
        conn = sqlite3.connect('nba.db')
        df = pd.read_sql("SELECT * FROM featured_games", conn)
        print(f"Successfully loaded {len(df)} rows from database.")
    except Exception as e:
        print(f"Error loading from database: {e}")
        return
        
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Feature Engineering (Recalculating if needed)
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

    # Train Model
    features = ['AVG_PTS_LAST_3', 'AVG_PTS_LAST_5', 'AVG_MIN_LAST_3', 'STREAK', 'REST_DAYS', 'HOME', 'OPP_DEF_RATING', 'OPP_PTS_ALLOWED']
    train_df = df.dropna(subset=features + ['PTS'])
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(train_df[features], train_df['PTS'])
    
    rf_model = model
    nba_data = df
    print("Model trained successfully!")

# Run Setup
initialize_model()

def get_player_id(name):
    player_list = players.get_players()
    for p in player_list:
        if p['full_name'].lower() == name.lower():
            return p['id']
    return None

@app.route('/api/teams')
def get_teams():
    if nba_data is None:
        return jsonify([])
    try:
        # Get unique list of teams from our dataset
        teams_list = sorted(nba_data['TEAM'].dropna().unique().tolist())
        return jsonify(teams_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if rf_model is None:
        return jsonify({"error: Model failed to load. Check server logs"}), 500
    
    data = request.get_json()
    player_name = data.get('player')
    opponent = data.get('opponent') 
    location = data.get('location') # Expect 'vs' (Home) or '@' (Away)
    
    try:
        point_spread = float(data.get('spread'))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid spread value"}), 400

    # Home/Away logic
    if location == 'vs':
        home = 1
    elif location == '@':
        home = 0
    else:
        return jsonify({"error": "Invalid location. Use 'vs' or '@'"}), 400

    # Get Player History
    player_hist = nba_data[nba_data['Player'] == player_name].sort_values('GAME_DATE')
    if player_hist.empty:
        return jsonify({"error": f"Player {player_name} not found in database"}), 404

    # Build Input
    # Safety check for opponent data
    opp_def_rating = nba_data['OPP_DEF_RATING'].mean()
    opp_pts_allowed = nba_data['OPP_PTS_ALLOWED'].mean()
    
    if opponent in nba_data['OPPONENT'].values:
        opp_data = nba_data[nba_data['OPPONENT'] == opponent]
        if not opp_data.empty:
            opp_def_rating = opp_data['OPP_DEF_RATING'].iloc[-1]
            opp_pts_allowed = opp_data['OPP_PTS_ALLOWED'].iloc[-1]

    input_row = pd.DataFrame([{
        'AVG_PTS_LAST_3': player_hist['PTS'].tail(3).mean(),
        'AVG_PTS_LAST_5': player_hist['PTS'].tail(5).mean(),
        'AVG_MIN_LAST_3': player_hist['MIN'].tail(3).mean(),
        'STREAK': player_hist['STREAK'].iloc[-1],
        'REST_DAYS': 1,
        'HOME': home,
        'OPP_DEF_RATING': opp_def_rating,
        'OPP_PTS_ALLOWED': opp_pts_allowed
    }])

    #Prediction and Confidence Calculation
    #1. Get the aggregate prediction
    
    # Predict
    pred_pts = rf_model.predict(input_row)[0]
    
    #Calculate confidence based on tree votes
    # Ask every single tree in the forest for its prediction
    all_tree_preds = [tree.predict(input_row)[0] for tree in rf_model.estimators_]

    #Count how many trees agree with the over vs under
    over_votes = sum(1 for p in all_tree_preds if p > point_spread)
    total_trees = len(all_tree_preds)

    over_prob = over_votes / total_trees
    under_prob = 1.0 - over_prob

    pick = "OVER" if pred_pts > point_spread else "UNDER"

    #The confidence is the probability of the chosen outcome
    confidence_score = over_prob if pick == "OVER" else under_prob

    # Get Feature Importance (Global Factors)
    # Match feature names with their importance scores
    importances = rf_model.feature_importances_
    feature_importance_list = []
    for name, imp in zip(MODEL_FEATURES, importances):
        feature_importance_list.append({"name": name, "score": imp})

    #Sort by score descending and take top 3
    top_factors = sorted(feature_importance_list, key=lambda x: x['score'], reverse=True)[:3]

    return jsonify({
        "player": player_name,
        "spread": point_spread,
        "projected_points": f"{pred_pts:.1f}",
        "pick": "OVER" if pred_pts > point_spread else "UNDER",
        "edge": f"{abs(pred_pts - point_spread):.1f}",
        "confidence": f"{confidence_score * 100:.0f}%",
        "top_factors": top_factors,
        "confidence_note": "Based on historical regression model"
    })

@app.route('/api/player/<player_name>')
def player_stats(player_name):
    player_id = get_player_id(player_name)

    if player_id is None:
        return jsonify({"error": "Player not found"}), 404
    
    try:
        # 1. Try fetching current season (2025-26)
        logs = playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
        df = logs.get_data_frames()[0]

        # 2. Fallback to previous season if no games found
        if df.empty:
             logs = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
             df = logs.get_data_frames()[0]

        if df.empty:
             return jsonify({"error": "No games found for this player"}), 404

        last_game = df.iloc[0]
        
        # 3. EXTRACT TEAM FROM MATCHUP (e.g. "BOS @ NYK" -> "BOS")
        matchup_str = str(last_game["MATCHUP"])
        # Split by space and take the first part (The player's team)
        current_team = matchup_str.split(' ')[0]

        stats = {
            "player": str(player_name),
            "team": current_team,  # Using the extracted team
            "pts": int(last_game["PTS"]),
            "reb": int(last_game["REB"]),
            "ast": int(last_game["AST"]),
            "min": str(last_game["MIN"]),
            "fg_pct": float(last_game["FG_PCT"]),
            "opp": matchup_str
        }
        return jsonify(stats)
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return jsonify({"error": str(e)}), 500

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