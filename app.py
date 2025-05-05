from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pytz
import joblib
import os
import json
import logging
from routes.players import players_bp
from routes.submissions import submissions_bp

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

# Load the model and scaler
try:
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logger.info("Successfully loaded model and scaler")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

# Load the player data
try:
    df = pd.read_csv('nba_players_final_updated.csv')
    logger.info(f"Successfully loaded {len(df)} players from CSV")
    logger.info(f"Available dollar values: {df['Dollar Value'].unique()}")
    logger.info(f"CSV columns: {df.columns.tolist()}")
except Exception as e:
    logger.error(f"Error loading player data: {str(e)}")
    raise

# File to store the daily pool
DAILY_POOL_FILE = 'daily_pool.json'

# Register blueprints
app.register_blueprint(players_bp)
app.register_blueprint(submissions_bp)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

def generate_daily_pool():
    try:
        # Get current date in Eastern time
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern)
        logger.info(f"Current time: {current_time}")
        
        # Check if we need to generate a new pool
        if os.path.exists(DAILY_POOL_FILE):
            try:
                with open(DAILY_POOL_FILE, 'r') as f:
                    pool_data = json.load(f)
                    last_generated = datetime.fromisoformat(pool_data['last_generated'])
                    if last_generated.date() == current_time.date():
                        logger.info("Using existing daily pool")
                        return pool_data['players']
            except Exception as e:
                logger.error(f"Error reading daily pool file: {str(e)}")
                # Continue to generate new pool if reading fails
        
        logger.info("Generating new daily pool")
        pool = []
        
        # Generate new pool
        for dollar_value in range(1, 6):
            # Filter players for current dollar value
            dollar_players = df[df['Dollar Value'] == dollar_value].copy()
            logger.info(f"Found {len(dollar_players)} players for ${dollar_value}")
            
            if len(dollar_players) < 5:
                logger.error(f"Not enough players for dollar value {dollar_value}")
                raise ValueError(f"Not enough players for dollar value {dollar_value}")
            
            # Randomly select 5 players
            selected_players = dollar_players.sample(n=5, random_state=current_time.date().toordinal())
            
            # Add to pool
            for _, player in selected_players.iterrows():
                pool.append(player.to_dict())
        
        # Save the new pool
        pool_data = {
            'last_generated': current_time.isoformat(),
            'players': pool
        }
        
        try:
            with open(DAILY_POOL_FILE, 'w') as f:
                json.dump(pool_data, f)
        except Exception as e:
            logger.error(f"Error saving daily pool: {str(e)}")
            # Continue even if saving fails
        
        logger.info(f"Generated new pool with {len(pool)} players")
        return pool
    except Exception as e:
        logger.error(f"Error generating daily pool: {str(e)}")
        raise

@app.route('/api/players', methods=['GET', 'OPTIONS'])
def get_players():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        logger.info("Received request for players")
        pool = generate_daily_pool()
        logger.info(f"Returning {len(pool)} players")
        response = jsonify(pool)
        return response
    except Exception as e:
        logger.error(f"Error in get_players: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        selected_players = data.get('players', [])
        
        if not selected_players:
            return jsonify({'error': 'No players selected'}), 400
        
        # Calculate team statistics
        team_stats = {
            'points': sum(float(p['Points Per Game (Avg)']) for p in selected_players),
            'rebounds': sum(float(p['Rebounds Per Game (Avg)']) for p in selected_players),
            'assists': sum(float(p['Assists Per Game (Avg)']) for p in selected_players),
            'steals': sum(float(p['Steals Per Game (Avg)']) for p in selected_players),
            'blocks': sum(float(p['Blocks Per Game (Avg)']) for p in selected_players),
            'turnovers': sum(float(p['Turnovers Per Game (Avg)']) for p in selected_players),
            'fg_pct': sum(float(p['Field Goal % (Avg)']) for p in selected_players) / len(selected_players),
            'ft_pct': sum(float(p['Free Throw % (Avg)']) for p in selected_players) / len(selected_players),
            'three_pct': sum(float(p['Three Point % (Avg)']) for p in selected_players) / len(selected_players),
            'plus_minus': sum(float(p['Plus Minus (Avg)']) for p in selected_players) / len(selected_players),
            'off_rating': sum(float(p['Offensive Rating (Avg)']) for p in selected_players) / len(selected_players),
            'def_rating': sum(float(p['Defensive Rating (Avg)']) for p in selected_players) / len(selected_players),
            'net_rating': sum(float(p['Net Rating (Avg)']) for p in selected_players) / len(selected_players),
            'usage': sum(float(p['Usage % (Avg)']) for p in selected_players) / len(selected_players),
            'pie': sum(float(p['Player Impact Estimate (Avg)']) for p in selected_players) / len(selected_players)
        }
        
        # Scale the features
        features = np.array([[
            team_stats['points'],
            team_stats['rebounds'],
            team_stats['assists'],
            team_stats['steals'],
            team_stats['blocks'],
            team_stats['fg_pct'],
            team_stats['ft_pct'],
            team_stats['three_pct']
        ]])
        
        scaled_features = scaler.transform(features)
        
        # Make prediction
        predicted_wins = model.predict(scaled_features)[0]
        
        # Ensure prediction stays within reasonable bounds
        predicted_wins = max(8, min(74, predicted_wins))
        
        return jsonify({'predicted_wins': predicted_wins})
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 