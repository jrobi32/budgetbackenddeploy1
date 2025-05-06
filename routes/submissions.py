from flask import Blueprint, request, jsonify
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import os
import numpy as np
import logging
import joblib
from psycopg2 import pool

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

# Create a connection pool
try:
    connection_pool = pool.SimpleConnectionPool(
        1,  # minconn
        10,  # maxconn
        host=os.getenv('DB_HOST', 'dpg-d07hog3uibrs73fg9c20-a.oregon-postgres.render.com'),
        database=os.getenv('DB_NAME', 'budgetgm'),
        user=os.getenv('DB_USER', 'budgetgm_user'),
        password=os.getenv('DB_PASSWORD', 'aqXhpXpEGGBmI5WvgG8YqPbqEBKRBqSx'),
        sslmode='require'
    )
    logger.info("Successfully created connection pool")
except Exception as e:
    logger.error(f"Error creating connection pool: {str(e)}")
    raise

def get_db_connection():
    try:
        conn = connection_pool.getconn()
        if conn:
            logger.info("Successfully got connection from pool")
            return conn
        else:
            raise Exception("Failed to get connection from pool")
    except Exception as e:
        logger.error(f"Error getting connection from pool: {str(e)}")
        raise

def release_db_connection(conn):
    try:
        connection_pool.putconn(conn)
        logger.info("Successfully released connection back to pool")
    except Exception as e:
        logger.error(f"Error releasing connection back to pool: {str(e)}")

@submissions_bp.route('/api/submit-team', methods=['POST'])
def submit_team():
    conn = None
    cur = None
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['nickname', 'players', 'results']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get current date in Eastern time
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Try to insert the submission
        try:
            cur.execute(
                """
                INSERT INTO submissions (nickname, submission_date, players, results)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (data['nickname'], current_date, Json(data['players']), Json(data['results']))
            )
            submission_id = cur.fetchone()[0]
            conn.commit()
            
            return jsonify({
                'message': 'Team submitted successfully',
                'submission_id': submission_id
            }), 201
            
        except psycopg2.IntegrityError:
            conn.rollback()
            return jsonify({
                'error': 'You have already submitted a team today'
            }), 409
            
    except Exception as e:
        logger.error(f"Error in submit_team: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({'error': str(e)}), 500
        
    finally:
        if cur:
            cur.close()
        if conn:
            release_db_connection(conn)

@submissions_bp.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    conn = None
    cur = None
    try:
        # Get date from query parameter or use current date
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        logger.info(f"Fetching leaderboard for date: {date}")
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get submissions for the specified date, ordered by wins
        cur.execute(
            """
            SELECT nickname, players, results
            FROM submissions
            WHERE submission_date = %s
            ORDER BY (results->>'wins')::integer DESC
            """,
            (date,)
        )
        
        submissions = []
        for row in cur.fetchall():
            submissions.append({
                'nickname': row[0],
                'players': row[1],
                'results': row[2]
            })
        
        logger.info(f"Found {len(submissions)} submissions for date {date}")
        return jsonify({
            'date': date,
            'submissions': submissions
        }), 200
        
    except psycopg2.Error as e:
        logger.error(f"Database error in get_leaderboard: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({'error': 'Database error occurred'}), 500
        
    except Exception as e:
        logger.error(f"Error in get_leaderboard: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if cur:
            cur.close()
        if conn:
            release_db_connection(conn)

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