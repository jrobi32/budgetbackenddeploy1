from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from routes.players import players_bp
from routes.submissions import submissions_bp
import pandas as pd
import numpy as np
from datetime import datetime
import json
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

# Load player data
players_df = pd.read_csv('nba_players_final_updated.csv')

# Register blueprints
app.register_blueprint(players_bp)
app.register_blueprint(submissions_bp)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

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
    
    return jsonify({
        'team_stats': team_stats
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 