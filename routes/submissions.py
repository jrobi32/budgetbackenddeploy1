from flask import Blueprint, request, jsonify
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import os

submissions_bp = Blueprint('submissions', __name__)

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'budgetgm'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '')
    )
    return conn

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
        return jsonify({'error': str(e)}), 500
        
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

@submissions_bp.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    try:
        # Get date from query parameter or use current date
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        
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
        
        return jsonify({
            'date': date,
            'submissions': submissions
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close() 