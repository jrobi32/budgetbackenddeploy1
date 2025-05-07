from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import json
import os
import logging

players_bp = Blueprint('players', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File to store the daily pool
DAILY_POOL_FILE = 'daily_pool.json'

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
                    last_generated = eastern.localize(last_generated.replace(tzinfo=None))
                    
                    # Save current pool as yesterday's game state before generating new pool
                    from routes.submissions import save_game_state
                    yesterday = (current_time - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    save_game_state(yesterday, pool_data['players'])
                    logger.info(f"Saved current pool as game state for {yesterday}")
                    
                    # Force new pool generation
                    logger.info("Forcing new pool generation")
                    raise Exception("Force new pool generation")
                    
            except Exception as e:
                if "Force new pool generation" not in str(e):
                    logger.error(f"Error reading daily pool file: {str(e)}")
                # Continue to generate new pool
        
        logger.info("Generating new daily pool")
        pool = []
        
        # Load the player data
        df = pd.read_csv('nba_players_final_updated.csv')
        
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
                
            # Save the game state for this date
            from routes.submissions import save_game_state
            save_game_state(current_time.strftime('%Y-%m-%d'), pool)
            logger.info(f"Saved game state for {current_time.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            logger.error(f"Error saving daily pool: {str(e)}")
            # Continue even if saving fails
        
        logger.info(f"Generated new pool with {len(pool)} players")
        return pool
    except Exception as e:
        logger.error(f"Error generating daily pool: {str(e)}")
        raise

@players_bp.route('/api/players', methods=['GET', 'OPTIONS'])
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