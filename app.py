from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import logging
from routes.players import players_bp
from routes.submissions import submissions_bp
import pandas as pd
import numpy as np
from datetime import datetime
import json
import psycopg2
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes with specific configuration
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://master--budgetgm1.netlify.app",
            "https://budgetgm1.netlify.app",
            "http://localhost:3000",
            "http://localhost:5000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'budgetgm'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '')
    )
    return conn

# Load the model and scaler
try:
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logger.info("Successfully loaded model and scaler")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

# Load player data
players_df = pd.read_csv('nba_players_final_updated.csv')

# Store submissions in memory (in production, use a database)
submissions = {}

# Register blueprints
app.register_blueprint(players_bp)
app.register_blueprint(submissions_bp)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/players', methods=['GET'])
def get_players():
    return jsonify(players_df.to_dict('records'))

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    players = data.get('players', [])
    nickname = data.get('nickname', '')
    
    if not players or len(players) != 5:
        return jsonify({'error': 'Invalid team size'}), 400
    
    # Get player data
    player_data = []
    for player in players:
        player_row = players_df[players_df['Player ID'] == player['id']].iloc[0]
        player_data.append(player_row)
    
    # Calculate team stats
    team_stats = {
        'points': sum(p['Points Per Game (Avg)'] for p in player_data),
        'rebounds': sum(p['Rebounds Per Game (Avg)'] for p in player_data),
        'assists': sum(p['Assists Per Game (Avg)'] for p in player_data),
        'steals': sum(p['Steals Per Game (Avg)'] for p in player_data),
        'blocks': sum(p['Blocks Per Game (Avg)'] for p in player_data),
        'turnovers': sum(p['Turnovers Per Game (Avg)'] for p in player_data),
        'fg_pct': sum(p['Field Goal % (Avg)'] for p in player_data) / 5,
        'ft_pct': sum(p['Free Throw % (Avg)'] for p in player_data) / 5,
        'three_pct': sum(p['Three Point % (Avg)'] for p in player_data) / 5,
        'plus_minus': sum(p['Plus Minus (Avg)'] for p in player_data) / 5,
        'off_rating': sum(p['Offensive Rating (Avg)'] for p in player_data) / 5,
        'def_rating': sum(p['Defensive Rating (Avg)'] for p in player_data) / 5,
        'net_rating': sum(p['Net Rating (Avg)'] for p in player_data) / 5,
        'usage': sum(p['Usage % (Avg)'] for p in player_data) / 5,
        'pie': sum(p['Player Impact Estimate (Avg)'] for p in player_data) / 5
    }
    
    # Scale the features
    features = np.array([[
        team_stats['points'], team_stats['rebounds'], team_stats['assists'],
        team_stats['steals'], team_stats['blocks'], team_stats['turnovers'],
        team_stats['fg_pct'], team_stats['ft_pct'], team_stats['three_pct'],
        team_stats['plus_minus'], team_stats['off_rating'], team_stats['def_rating'],
        team_stats['net_rating'], team_stats['usage'], team_stats['pie']
    ]])
    scaled_features = scaler.transform(features)
    
    # Make prediction
    predicted_wins = model.predict(scaled_features)[0]
    predicted_wins = max(8, min(74, predicted_wins))  # Keep within reasonable bounds
    
    # Store submission
    today = datetime.now().strftime('%Y-%m-%d')
    if today not in submissions:
        submissions[today] = []
    
    submission = {
        'nickname': nickname,
        'players': [{'id': p['id'], 'name': p['name'], 'value': p['value']} for p in players],
        'predicted_wins': float(predicted_wins),
        'timestamp': datetime.now().isoformat()
    }
    submissions[today].append(submission)
    
    return jsonify({
        'predicted_wins': float(predicted_wins),
        'team_stats': team_stats
    })

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    try:
        # Get current date in Eastern time
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get submissions for today, ordered by wins
        cur.execute(
            """
            SELECT nickname, players, results
            FROM submissions
            WHERE submission_date = %s
            ORDER BY (results->>'wins')::integer DESC
            """,
            (current_date,)
        )
        
        submissions = []
        for row in cur.fetchall():
            submissions.append({
                'nickname': row[0],
                'players': row[1],
                'results': row[2]
            })
        
        return jsonify({
            'date': current_date,
            'submissions': submissions
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_leaderboard: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 