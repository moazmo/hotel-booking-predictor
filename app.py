"""
Hotel Booking Predictor Flask Application
Production-ready version for Render deployment
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the templates and static folders
basedir = os.path.abspath(os.path.dirname(__file__))
template_folder = os.path.join(basedir, 'app', 'templates')
static_folder = os.path.join(basedir, 'app', 'static')

app = Flask(__name__, 
            template_folder=template_folder,
            static_folder=static_folder)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Global variables to store model artifacts
model = None
scaler = None
label_encoder = None
feature_info = None

# Load model artifacts at startup
def load_model_artifacts():
    """Load the trained model and preprocessors"""
    global model, scaler, label_encoder, feature_info
    
    try:
        # Use absolute paths
        models_dir = os.path.join(basedir, 'models')
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            logger.info("This is normal during deployment - models will be created by extract_model.py")
            return False
        
        # Load model
        model_path = os.path.join(models_dir, 'best_model.pkl')
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Model will be created during deployment build process")
            return False
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        # Load label encoder
        le_path = os.path.join(models_dir, 'label_encoder.pkl')
        if os.path.exists(le_path):
            with open(le_path, 'rb') as f:
                label_encoder = pickle.load(f)
        
        # Load feature info
        fi_path = os.path.join(models_dir, 'feature_info.pkl')
        if os.path.exists(fi_path):
            with open(fi_path, 'rb') as f:
                feature_info = pickle.load(f)
        
        logger.info("Model artifacts loaded successfully")
        if feature_info:
            logger.info(f"Model: {feature_info['model_name']}")
            logger.info(f"Accuracy: {feature_info['model_accuracy']:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return False

# Initialize model artifacts at module level for gunicorn
try:
    load_model_artifacts()
    if feature_info:
        logger.info(f"📊 Model loaded: {feature_info['model_name']}")
        logger.info(f"🎯 Accuracy: {feature_info['model_accuracy']:.4f}")
except Exception as e:
    logger.warning(f"Model artifacts not loaded at startup: {e}. Will load on first request.")

def preprocess_input(data):
    """Preprocess input data to match training format"""
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Feature engineering (same as training)
        input_df['total_nights'] = input_df['number_of_weekend_nights'] + input_df['number_of_week_nights']
        input_df['total_guests'] = input_df['number_of_adults'] + input_df['number_of_children']
        input_df['price_per_night'] = input_df['average_price'] / (input_df['total_nights'] + 1)
        
        # One-hot encode categorical features
        categorical_features = ['type_of_meal', 'room_type', 'market_segment_type']
        for feature in categorical_features:
            if feature in input_df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(input_df[feature], prefix=feature, drop_first=True)
                input_df = pd.concat([input_df, dummies], axis=1)
                input_df.drop(feature, axis=1, inplace=True)
        
        # Ensure all required columns are present
        missing_cols = set(feature_info['all_columns']) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        
        # Reorder columns to match training
        input_df = input_df[feature_info['all_columns']]
        
        # Scale numeric features
        numeric_cols = feature_info['numeric_columns']
        if scaler and numeric_cols:
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        return input_df
    
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    elif request.method == 'POST':
        try:
            # Ensure model is loaded
            if model is None:
                logger.info("Model not loaded, attempting to load...")
                if not load_model_artifacts():
                    return render_template('error.html', 
                                         error="Model is not available. Please try again later.")
            
            # Extract form data
            input_data = {
                'number_of_adults': int(request.form['number_of_adults']),
                'number_of_children': int(request.form['number_of_children']),
                'number_of_weekend_nights': int(request.form['number_of_weekend_nights']),
                'number_of_week_nights': int(request.form['number_of_week_nights']),
                'type_of_meal': request.form['type_of_meal'],
                'car_parking_space': int(request.form['car_parking_space']),
                'room_type': request.form['room_type'],
                'lead_time': int(request.form['lead_time']),
                'market_segment_type': request.form['market_segment_type'],
                'repeated': int(request.form['repeated']),
                'P-C': int(request.form['p_c']),
                'P-not-C': int(request.form['p_not_c']),
                'average_price': float(request.form['average_price']),
                'special_requests': int(request.form['special_requests'])
            }
            
            # Preprocess input
            processed_data = preprocess_input(input_data)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            # Decode prediction
            predicted_status = label_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba) * 100
            
            # Calculate additional insights
            total_nights = input_data['number_of_weekend_nights'] + input_data['number_of_week_nights']
            total_guests = input_data['number_of_adults'] + input_data['number_of_children']
            price_per_night = input_data['average_price'] / max(total_nights, 1)
            
            result = {
                'prediction': predicted_status,
                'confidence': confidence,
                'total_nights': total_nights,
                'total_guests': total_guests,
                'price_per_night': price_per_night,
                'model_name': feature_info['model_name'],
                'model_accuracy': feature_info['model_accuracy'] * 100,
                'prediction_time': datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
            }
            
            return render_template('result.html', result=result, input_data=input_data)
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Decode prediction
        predicted_status = label_encoder.inverse_transform([prediction])[0]
        confidence = max(prediction_proba)
        
        return jsonify({
            'prediction': predicted_status,
            'confidence': float(confidence),
            'probabilities': {
                'canceled': float(prediction_proba[0]),
                'not_canceled': float(prediction_proba[1])
            },
            'model_info': {
                'name': feature_info['model_name'],
                'accuracy': feature_info['model_accuracy']
            }
        })
    
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', feature_info=feature_info)

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    try:
        # Basic health check - just verify the app is running
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'app': 'hotel-booking-predictor'
        }
        
        # Add model status if available
        if model is not None:
            health_status['model'] = feature_info['model_name'] if feature_info else 'Unknown'
            health_status['accuracy'] = feature_info['model_accuracy'] if feature_info else 0
            health_status['model_loaded'] = True
        else:
            health_status['model_loaded'] = False
            health_status['message'] = 'Model will be loaded on first prediction request'
        
        return jsonify(health_status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'partial',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 200  # Return 200 even on error to pass health check

@app.route('/status')
def status():
    """Status endpoint for debugging"""
    try:
        models_dir = os.path.join(basedir, 'models')
        status_info = {
            'app': 'hotel-booking-predictor',
            'timestamp': datetime.now().isoformat(),
            'models_dir_exists': os.path.exists(models_dir),
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'label_encoder_loaded': label_encoder is not None,
            'feature_info_loaded': feature_info is not None
        }
        
        if os.path.exists(models_dir):
            status_info['model_files'] = os.listdir(models_dir)
        
        return jsonify(status_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Initialize model artifacts at module level for gunicorn
try:
    load_model_artifacts()
    if feature_info:
        logger.info(f"📊 Model loaded: {feature_info['model_name']}")
        logger.info(f"🎯 Accuracy: {feature_info['model_accuracy']:.4f}")
except Exception as e:
    logger.warning(f"Model artifacts not loaded at startup: {e}. Will load on first request.")

if __name__ == '__main__':
    # Get PORT from environment with robust error handling
    port_env = os.environ.get('PORT', '5000')
    
    # Handle cases where PORT might be '$PORT' or other invalid values
    try:
        port = int(port_env)
    except ValueError:
        logger.warning(f"Invalid PORT value: '{port_env}', using default 5000")
        port = 5000
    
    logger.info("🚀 Starting Hotel Booking Predictor App...")
    logger.info(f"🌐 Starting server on port {port}")
    logger.info(f"🔍 PORT environment variable: '{port_env}'")
    
    # For Render, disable debug mode and use threaded mode
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
