from flask import Blueprint, request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import joblib
import os
import json

submissions_bp = Blueprint('submissions', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and scaler
try:
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logger.info("Successfully loaded model and scaler")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

# File to store submissions
SUBMISSIONS_FILE = 'submissions.csv'

def ensure_submissions_file():
    """Ensure the submissions file exists with the correct columns"""
    if not os.path.exists(SUBMISSIONS_FILE):
        df = pd.DataFrame(columns=[
            'submission_date', 'nickname', 'players', 'results',
            'predicted_wins', 'team_stats'
        ])
        df.to_csv(SUBMISSIONS_FILE, index=False)
        logger.info("Created new submissions file")

def load_submissions():
    """Load submissions from CSV file"""
    try:
        if not os.path.exists(SUBMISSIONS_FILE):
            ensure_submissions_file()
        df = pd.read_csv(SUBMISSIONS_FILE)
        # Convert string representations back to Python objects
        df['players'] = df['players'].apply(eval)
        df['results'] = df['results'].apply(eval)
        df['team_stats'] = df['team_stats'].apply(eval)
        return df
    except Exception as e:
        logger.error(f"Error loading submissions: {str(e)}")
        raise

def save_submission(submission_date, nickname, players, results, predicted_wins, team_stats):
    """Save a new submission to the CSV file"""
    try:
        df = load_submissions()
        
        # Check if user already submitted today
        if len(df[(df['submission_date'] == submission_date) & (df['nickname'] == nickname)]) > 0:
            raise ValueError("You have already submitted a team today")
        
        # Add new submission
        new_row = pd.DataFrame([{
            'submission_date': submission_date,
            'nickname': nickname,
            'players': str(players),  # Convert to string for CSV storage
            'results': str(results),
            'predicted_wins': predicted_wins,
            'team_stats': str(team_stats)
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(SUBMISSIONS_FILE, index=False)
        logger.info(f"Saved submission for {nickname} on {submission_date}")
        
    except Exception as e:
        logger.error(f"Error saving submission: {str(e)}")
        raise

@submissions_bp.route('/api/submit-team', methods=['POST'])
def submit_team():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['nickname', 'players', 'results']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get current date in Eastern time
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate team statistics
        team_stats = {
            'points': sum(float(p['Points Per Game (Avg)']) for p in data['players']),
            'rebounds': sum(float(p['Rebounds Per Game (Avg)']) for p in data['players']),
            'assists': sum(float(p['Assists Per Game (Avg)']) for p in data['players']),
            'steals': sum(float(p['Steals Per Game (Avg)']) for p in data['players']),
            'blocks': sum(float(p['Blocks Per Game (Avg)']) for p in data['players']),
            'turnovers': sum(float(p['Turnovers Per Game (Avg)']) for p in data['players']),
            'fg_pct': sum(float(p['Field Goal % (Avg)']) for p in data['players']) / len(data['players']),
            'ft_pct': sum(float(p['Free Throw % (Avg)']) for p in data['players']) / len(data['players']),
            'three_pct': sum(float(p['Three Point % (Avg)']) for p in data['players']) / len(data['players'])
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
        predicted_wins = max(8, min(74, predicted_wins))  # Keep within reasonable bounds
        
        # Save the submission
        save_submission(
            current_date,
            data['nickname'],
            data['players'],
            data['results'],
            predicted_wins,
            team_stats
        )
        
        return jsonify({
            'message': 'Team submitted successfully',
            'predicted_wins': predicted_wins
        }), 201
            
    except ValueError as e:
        logger.error(f"Validation error in submit_team: {str(e)}")
        return jsonify({'error': str(e)}), 409
        
    except Exception as e:
        logger.error(f"Error in submit_team: {str(e)}")
        return jsonify({'error': str(e)}), 500

@submissions_bp.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    try:
        # Get date from query parameter or use current date
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        logger.info(f"Fetching leaderboard for date: {date}")
        
        # Load submissions
        df = load_submissions()
        
        # Filter for the requested date and sort by predicted wins
        daily_submissions = df[df['submission_date'] == date].sort_values('predicted_wins', ascending=False)
        
        # Convert to list of dictionaries
        submissions = []
        for _, row in daily_submissions.iterrows():
            submissions.append({
                'nickname': row['nickname'],
                'players': row['players'],
                'results': row['results'],
                'predicted_wins': row['predicted_wins'],
                'team_stats': row['team_stats']
            })
        
        logger.info(f"Found {len(submissions)} submissions for date {date}")
        return jsonify({
            'date': date,
            'submissions': submissions
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_leaderboard: {str(e)}")
        return jsonify({'error': str(e)}), 500

@submissions_bp.route('/api/predict', methods=['POST', 'OPTIONS'])
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
            'three_pct': sum(float(p['Three Point % (Avg)']) for p in selected_players) / len(selected_players)
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