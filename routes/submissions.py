from flask import Blueprint, request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import joblib
import os
import json
import pytz

submissions_bp = Blueprint('submissions', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File to store submissions - use a directory that persists on Render
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
SUBMISSIONS_FILE = os.path.join(DATA_DIR, 'submissions.json')

# File to store game states
GAME_STATES_FILE = 'game_states.csv'

def ensure_data_directory():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            logger.info(f"Created data directory: {DATA_DIR}")
        except Exception as e:
            logger.error(f"Error creating data directory: {str(e)}")
            raise

def ensure_submissions_file():
    """Ensure the submissions file exists."""
    ensure_data_directory()
    if not os.path.exists(SUBMISSIONS_FILE):
        try:
            with open(SUBMISSIONS_FILE, 'w') as f:
                json.dump({}, f)
            # Ensure file permissions are set correctly
            os.chmod(SUBMISSIONS_FILE, 0o666)
            logger.info(f"Created new submissions file: {SUBMISSIONS_FILE}")
        except Exception as e:
            logger.error(f"Error creating submissions file: {str(e)}")
            raise

def load_submissions():
    """Load all submissions from the file."""
    ensure_submissions_file()
    try:
        with open(SUBMISSIONS_FILE, 'r') as f:
            data = json.load(f)
            logger.info(f"Loaded {sum(len(submissions) for submissions in data.values())} total submissions")
            return data
    except Exception as e:
        logger.error(f"Error loading submissions: {str(e)}")
        return {}

def save_submissions(submissions):
    """Save all submissions to the file."""
    try:
        with open(SUBMISSIONS_FILE, 'w') as f:
            json.dump(submissions, f, indent=2)
        logger.info(f"Saved {sum(len(submissions) for submissions in submissions.values())} total submissions")
    except Exception as e:
        logger.error(f"Error saving submissions: {str(e)}")
        raise

def get_current_date():
    """Get current date in Eastern time."""
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern).strftime('%Y-%m-%d')

def get_submissions_for_date(date=None):
    """Get submissions for a specific date."""
    if date is None:
        date = get_current_date()
    
    submissions = load_submissions()
    date_submissions = submissions.get(date, [])
    logger.info(f"Found {len(date_submissions)} submissions for date {date}")
    return date_submissions

def add_submission(nickname, players, results):
    """Add a new submission for the current date."""
    date = get_current_date()
    submissions = load_submissions()
    
    if date not in submissions:
        submissions[date] = []
    
    # Check if user already submitted for today
    for submission in submissions[date]:
        if submission['nickname'] == nickname:
            return False, "You have already submitted a team for today"
    
    # Add new submission
    submissions[date].append({
        'nickname': nickname,
        'players': players,
        'results': results,
        'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat()
    })
    
    try:
        save_submissions(submissions)
        logger.info(f"Added submission for {nickname} on {date}")
        return True, "Submission successful"
    except Exception as e:
        logger.error(f"Error saving submission: {str(e)}")
        return False, "Error saving submission"

def get_leaderboard(date=None):
    """Get leaderboard for a specific date."""
    if date is None:
        date = get_current_date()
    
    submissions = get_submissions_for_date(date)
    
    # Sort by wins
    sorted_submissions = sorted(submissions, key=lambda x: x['results']['wins'], reverse=True)
    logger.info(f"Generated leaderboard for {date} with {len(sorted_submissions)} submissions")
    
    return {
        'date': date,
        'submissions': sorted_submissions
    }

def get_user_history(nickname):
    """Get submission history for a user."""
    submissions = load_submissions()
    user_history = []
    
    for date, date_submissions in submissions.items():
        for submission in date_submissions:
            if submission['nickname'] == nickname:
                user_history.append({
                    'date': date,
                    'players': submission['players'],
                    'results': submission['results']
                })
    
    return sorted(user_history, key=lambda x: x['date'], reverse=True)

def ensure_game_states_file():
    """Ensure the game states file exists with the correct columns"""
    try:
        if not os.path.exists(GAME_STATES_FILE):
            df = pd.DataFrame(columns=[
                'date',
                'player_stats'
            ])
            df.to_csv(GAME_STATES_FILE, index=False)
            logger.info("Created new game states file")
            # Ensure file permissions are set correctly
            os.chmod(GAME_STATES_FILE, 0o666)
    except Exception as e:
        logger.error(f"Error creating game states file: {str(e)}")
        raise

def save_game_state(date, player_stats):
    """Save the current game state"""
    try:
        if not os.path.exists(GAME_STATES_FILE):
            ensure_game_states_file()
        
        df = pd.read_csv(GAME_STATES_FILE)
        
        # Check if state already exists for this date
        if len(df[df['date'] == date]) > 0:
            logger.info(f"Game state already exists for {date}")
            return
        
        # Convert player_stats to string if it's not already
        if not isinstance(player_stats, str):
            player_stats = json.dumps(player_stats)
        
        # Add new game state
        new_row = pd.DataFrame([{
            'date': date,
            'player_stats': player_stats
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(GAME_STATES_FILE, index=False)
        logger.info(f"Saved game state for {date} with {len(eval(player_stats))} players")
        
    except Exception as e:
        logger.error(f"Error saving game state: {str(e)}")
        raise

def load_game_state(date):
    """Load game state for a specific date"""
    try:
        if not os.path.exists(GAME_STATES_FILE):
            logger.warning(f"Game states file not found: {GAME_STATES_FILE}")
            return None
            
        df = pd.read_csv(GAME_STATES_FILE)
        state = df[df['date'] == date]
        
        if state.empty:
            logger.warning(f"No game state found for date: {date}")
            return None
            
        # Convert string representation back to Python object
        player_stats = eval(state.iloc[0]['player_stats'])
        logger.info(f"Loaded game state for {date} with {len(player_stats)} players")
        return player_stats
        
    except Exception as e:
        logger.error(f"Error loading game state: {str(e)}")
        return None

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
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern)
        today = current_time.strftime('%Y-%m-%d')
        
        # Check if user already submitted today
        submissions = load_submissions()
        existing_submission = get_submissions_for_date(today)
        
        if existing_submission:
            for submission in existing_submission:
                if submission['nickname'] == data['nickname']:
                    return jsonify({'error': 'You have already submitted a team today.'}), 409

        # Calculate team statistics
        team_stats = {
            'points': sum(float(p['Points Per Game (Avg)']) for p in data['players']),
            'rebounds': sum(float(p['Rebounds Per Game (Avg)']) for p in data['players']),
            'assists': sum(float(p['Assists Per Game (Avg)']) for p in data['players']),
            'steals': sum(float(p['Steals Per Game (Avg)']) for p in data['players']),
            'blocks': sum(float(p['Blocks Per Game (Avg)']) for p in data['players']),
            'turnovers': sum(float(p['TOV']) for p in data['players']),
            'fg_pct': sum(float(p['Field Goal % (Avg)']) for p in data['players']) / len(data['players']),
            'ft_pct': sum(float(p['Free Throw % (Avg)']) for p in data['players']) / len(data['players']),
            'three_pct': sum(float(p['Three Point % (Avg)']) for p in data['players']) / len(data['players'])
        }
        
        # Use the frontend's predicted wins
        predicted_wins = data['results']['wins']
        
        # Save the submission
        success, message = add_submission(
            data['nickname'],
            data['players'],
            data['results']
        )
        
        if not success:
            return jsonify({'error': message}), 409
        
        # Save the game state if it doesn't exist
        if not os.path.exists(GAME_STATES_FILE) or len(pd.read_csv(GAME_STATES_FILE)[pd.read_csv(GAME_STATES_FILE)['date'] == today]) == 0:
            save_game_state(today, data['players'])
        
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
def get_leaderboard_route():
    try:
        # Get date from query parameter or use current date
        date = request.args.get('date')
        if not date:
            # Get current date in Eastern time
            eastern = pytz.timezone('US/Eastern')
            current_time = datetime.now(eastern)
            date = current_time.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching leaderboard for date: {date}")
        
        leaderboard_data = get_leaderboard(date)
        
        return jsonify(leaderboard_data), 200
        
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
            'turnovers': sum(float(p['TOV']) for p in selected_players),
            'fg_pct': sum(float(p['Field Goal % (Avg)']) for p in selected_players) / len(selected_players),
            'ft_pct': sum(float(p['Free Throw % (Avg)']) for p in selected_players) / len(selected_players),
            'three_pct': sum(float(p['Three Point % (Avg)']) for p in selected_players) / len(selected_players)
        }
        
        # Scale the features
        features = np.array([[
            team_stats['points'],  # Let the scaler handle the scaling
            team_stats['rebounds'],
            team_stats['assists'],
            team_stats['steals'],
            team_stats['blocks'],
            team_stats['turnovers'],
            team_stats['fg_pct'],
            team_stats['ft_pct'],
            team_stats['three_pct']
        ]])
        
        scaled_features = scaler.transform(features)
        
        # Make prediction
        predicted_wins = model.predict(scaled_features)[0]
        
        # Ensure prediction stays within reasonable bounds
        predicted_wins = max(0, min(74, predicted_wins))
        
        return jsonify({'predicted_wins': predicted_wins})
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

@submissions_bp.route('/api/history', methods=['GET'])
def get_history():
    try:
        # Get list of available dates
        if not os.path.exists(GAME_STATES_FILE):
            logger.warning(f"Game states file not found: {GAME_STATES_FILE}")
            return jsonify({'dates': []}), 200
            
        df = pd.read_csv(GAME_STATES_FILE)
        dates = df['date'].tolist()
        logger.info(f"Found {len(dates)} available dates")
        
        # Get user's submission history if nickname provided
        nickname = request.args.get('nickname')
        if nickname:
            user_history = get_user_history(nickname)
            logger.info(f"Found {len(user_history)} played dates for user {nickname}")
        else:
            user_history = []
        
        return jsonify({
            'dates': dates,
            'played_dates': [h['date'] for h in user_history]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@submissions_bp.route('/api/game-state/<date>', methods=['GET'])
def get_game_state(date):
    try:
        state = load_game_state(date)
        if state is None:
            return jsonify({'error': 'Game state not found for this date'}), 404
            
        return jsonify({
            'date': date,
            'player_stats': state
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_game_state: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Initialize data directory and submissions file when module is imported
ensure_data_directory()
ensure_submissions_file() 