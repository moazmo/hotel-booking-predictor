"""
Hotel Booking Predictor Flask Application
Production-ready version for Railway deployment
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

def load_model_artifacts():
    """Load the trained model and preprocessors"""
    global model, scaler, label_encoder, feature_info
    
    try:
        # Use absolute paths
        models_dir = os.path.join(basedir, 'models')
        
        # Load model
        with open(os.path.join(models_dir, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(models_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load feature info
        with open(os.path.join(models_dir, 'feature_info.pkl'), 'rb') as f:
            feature_info = pickle.load(f)
        
        logger.info("Model artifacts loaded successfully")
        logger.info(f"Model: {feature_info['model_name']}")
        logger.info(f"Accuracy: {feature_info['model_accuracy']:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return False

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
    """Health check endpoint for Railway"""
    try:
        if model is None:
            return jsonify({'status': 'unhealthy', 'reason': 'model not loaded'}), 503
        return jsonify({
            'status': 'healthy',
            'model': feature_info['model_name'] if feature_info else 'Unknown',
            'accuracy': feature_info['model_accuracy'] if feature_info else 0,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load model artifacts
    if not load_model_artifacts():
        logger.error("❌ Failed to load model artifacts. Please run extract_model.py first.")
        exit(1)
    
    logger.info("🚀 Starting Hotel Booking Predictor App...")
    logger.info(f"📊 Model: {feature_info['model_name']}")
    logger.info(f"🎯 Accuracy: {feature_info['model_accuracy']:.4f}")
    
    # Use environment PORT for Railway deployment, fallback to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
