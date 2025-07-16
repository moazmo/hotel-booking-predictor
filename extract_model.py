"""
Script to extract the trained model and preprocessors from the Jupyter notebook
Run this script after running the notebook to save the model artifacts
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the data following the same steps as in the notebook"""
    
    # Load the dataset - try multiple locations
    possible_paths = [
        'first intern project.csv',  # Current directory
        os.path.join(os.path.dirname(__file__), 'first intern project.csv'),  # Script directory
        os.path.join(os.path.dirname(__file__), '..', 'first intern project.csv'),  # Parent directory
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Dataset 'first intern project.csv' not found in any of these locations: {possible_paths}")
    
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Data preprocessing
    df_clean = df.dropna()
    df_clean.columns = df_clean.columns.str.strip()
    df_clean['date of reservation'] = pd.to_datetime(df_clean['date of reservation'], format='mixed', errors='coerce')
    df_clean = df_clean[df_clean['date of reservation'].notna()]
    df_clean = df_clean.drop_duplicates()
    
    if 'Booking_ID' in df_clean.columns:
        df_clean = df_clean.drop('Booking_ID', axis=1)
    
    # Outlier removal using IQR method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    df_no_outliers = df_clean.copy()
    
    for col in numeric_cols:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]
    
    # Feature engineering
    df_features = df_no_outliers.copy()
    df_features['total_nights'] = df_features['number of weekend nights'] + df_features['number of week nights']
    df_features['total_guests'] = df_features['number of adults'] + df_features['number of children']
    df_features['price_per_night'] = df_features['average price'] / (df_features['total_nights'] + 1)
    
    # Remove date column
    df_features = df_features.drop(['date of reservation'], axis=1)
    
    # Handle multicollinearity - remove highly correlated features
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    correlation_matrix = df_features[numeric_features].corr()
    
    # Find high correlations (>0.8)
    high_corr = np.where(np.abs(correlation_matrix) > 0.8)
    high_corr_pairs = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y]) 
                       for x, y in zip(*high_corr) if x != y and x < y]
    
    # Remove highly correlated features
    features_to_remove = []
    for pair in high_corr_pairs:
        feature1, feature2 = pair[0], pair[1]
        if feature1 not in features_to_remove and feature2 not in features_to_remove:
            if feature1 > feature2:  # Alphabetical comparison
                features_to_remove.append(feature1)
            else:
                features_to_remove.append(feature2)
    
    numeric_features_final = [col for col in numeric_features if col not in features_to_remove]
    
    # Categorical data transformation
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    
    # Label encode target variable
    le_target = LabelEncoder()
    df_features['booking_status_encoded'] = le_target.fit_transform(df_features['booking status'])
    
    # One-hot encode other categorical features
    categorical_features = [col for col in categorical_cols if col != 'booking status']
    df_encoded = pd.get_dummies(df_features, columns=categorical_features, drop_first=True)
    
    # Prepare final dataset
    y = df_encoded['booking_status_encoded']
    numeric_cols_final = [col for col in numeric_features_final if col in df_encoded.columns]
    categorical_encoded_cols = [col for col in df_encoded.columns if col not in df_features.columns]
    
    X = df_encoded[numeric_cols_final + categorical_encoded_cols]
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numeric_cols_final] = scaler.fit_transform(X[numeric_cols_final])
    
    return X, y, scaler, le_target, numeric_cols_final, categorical_encoded_cols

def train_and_save_models():
    """Train models and save the best one along with preprocessors"""
    
    print("Loading and preprocessing data...")
    X, y, scaler, le_target, numeric_cols_final, categorical_encoded_cols = load_and_preprocess_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)  # Added probability=True for predict_proba
    }
    
    results = {}
    trained_models = {}
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        trained_models[name] = model
        print(f"{name}: {accuracy:.4f}")
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]
    best_accuracy = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name} ({best_accuracy:.4f})")
    
    # Create models directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)
    
    # Save the best model
    with open('./models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save the scaler
    with open('./models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the label encoder
    with open('./models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le_target, f)
    
    # Save feature information
    feature_info = {
        'numeric_columns': numeric_cols_final,
        'categorical_columns': categorical_encoded_cols,
        'all_columns': list(X.columns),
        'target_classes': list(le_target.classes_),
        'model_name': best_model_name,
        'model_accuracy': best_accuracy
    }
    
    with open('./models/feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"\nModel artifacts saved to ./models/")
    print(f"- best_model.pkl: {best_model_name}")
    print(f"- scaler.pkl: StandardScaler")
    print(f"- label_encoder.pkl: LabelEncoder for target")
    print(f"- feature_info.pkl: Feature metadata")
    
    return feature_info

if __name__ == "__main__":
    try:
        print("🚀 Starting model extraction...")
        print(f"📂 Current working directory: {os.getcwd()}")
        print(f"📄 Script location: {os.path.dirname(__file__)}")
        
        # List files in current directory
        print(f"📋 Files in current directory: {os.listdir('.')}")
        
        feature_info = train_and_save_models()
        print("\n✅ Model extraction completed successfully!")
        print(f"🎯 Best model: {feature_info['model_name']}")
        print(f"📊 Accuracy: {feature_info['model_accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Error during model extraction: {str(e)}")
        print(f"📍 Error type: {type(e).__name__}")
        import traceback
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
        raise
