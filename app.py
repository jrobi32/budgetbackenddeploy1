from flask import Flask, jsonify
from flask_cors import CORS
import joblib
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

# Register blueprints
app.register_blueprint(players_bp)
app.register_blueprint(submissions_bp)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 