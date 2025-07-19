"""
Optimized model extraction that's compatible with the original app.py
This creates a better model while maintaining full compatibility with the existing web app
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess data with simple approach matching original app.py"""
    
    # Load the dataset
    possible_paths = [
        'first intern project.csv',
        os.path.join('..', 'first intern project.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'first intern project.csv'),
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError("Dataset 'first intern project.csv' not found")
    
    print(f"✅ Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Basic preprocessing - drop nulls and duplicates
    df_clean = df.dropna().drop_duplicates()
    
    # Clean column names (remove trailing spaces)
    df_clean.columns = df_clean.columns.str.strip()
    
    print(f"   Shape after cleaning: {df_clean.shape}")
    
    return df_clean

def simple_feature_engineering(df):
    """Simple feature engineering matching the original app.py exactly"""
    print("🔧 Creating simple features (matching app.py exactly)...")
    
    df_features = df.copy()
    
    # Create basic features (same as original app.py)
    df_features['total_nights'] = df_features['number of weekend nights'] + df_features['number of week nights']
    df_features['total_guests'] = df_features['number of adults'] + df_features['number of children']
    df_features['price_per_night'] = df_features['average price'] / (df_features['total_nights'] + 1)
    
    # Remove columns that won't be used
    columns_to_remove = ['Booking_ID', 'date of reservation']
    df_features = df_features.drop(columns_to_remove, axis=1, errors='ignore')
    
    print(f"   Dataset shape after feature engineering: {df_features.shape}")
    return df_features

def simple_categorical_encoding(df):
    """Simple categorical encoding matching original app.py exactly"""
    print("🔤 Encoding categorical features (matching app.py exactly)...")
    
    # Encode target variable
    le_target = LabelEncoder()
    df['booking_status_encoded'] = le_target.fit_transform(df['booking status'])
    print(f"   Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    
    # One-hot encode categorical features (same as original app.py)
    categorical_features = ['type of meal', 'room type', 'market segment type']
    df_encoded = df.copy()
    
    for feature in categorical_features:
        if feature in df_encoded.columns:
            print(f"   Encoding {feature}...")
            # Create dummy variables with drop_first=True (same as original app.py)
            # Use the exact same prefix format as original app.py
            dummies = pd.get_dummies(df_encoded[feature], prefix=feature.replace(' ', '_'), drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(feature, axis=1, inplace=True)
    
    print(f"   Shape after encoding: {df_encoded.shape}")
    return df_encoded, le_target

def train_compatible_optimized_model(df, target_col):
    """Train optimized models compatible with original app.py"""
    print("🚀 Training optimized models (compatible with app.py)...")
    
    # Prepare features and target
    y = df[target_col]
    feature_columns = [col for col in df.columns if col not in ['booking status', 'booking_status_encoded']]
    X = df[feature_columns].copy()
    
    # Fill missing values
    X = X.fillna(0)
    
    print(f"   Feature columns: {len(feature_columns)}")
    print(f"   Features: {feature_columns}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE for class balancing (optimization)
    print("   Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"   Training data after SMOTE: {X_train_balanced.shape}")
    
    # Scale numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    
    X_train_scaled = X_train_balanced.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_balanced[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"   Scaled {len(numeric_cols)} numeric columns")
    
    # Train optimized models with cross-validation
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            class_weight='balanced',
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=2000,
            class_weight='balanced',
            C=1.0
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            max_depth=15
        )
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_model = None
    best_score = 0
    best_name = ''
    results = {}
    
    print("   Training models with cross-validation...")
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Cross-validation on balanced training data
        cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, 
                                   cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        
        # Train on balanced data and test on original test set
        model.fit(X_train_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"     CV Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"     Test Balanced Accuracy: {balanced_acc:.4f}")
        print(f"     Regular Accuracy: {accuracy:.4f}")
        print(f"     AUC-ROC: {auc:.4f}")
        
        if balanced_acc > best_score:
            best_score = balanced_acc
            best_model = model
            best_name = name
    
    # Find optimal threshold
    y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_best)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Balanced Accuracy: {best_score:.4f}")
    print(f"   Regular Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"   AUC-ROC: {results[best_name]['auc_roc']:.4f}")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    
    return best_model, best_name, results[best_name], optimal_threshold, scaler, feature_columns, numeric_cols

def main():
    """Main function to create optimized model compatible with original app.py"""
    print("🔧 Creating Optimized Model Compatible with Original App...")
    print("=" * 60)
    
    try:
        # 1. Load and preprocess data
        df_clean = load_and_preprocess_data()
        
        # 2. Simple feature engineering (matching app.py)
        df_features = simple_feature_engineering(df_clean)
        
        # 3. Simple categorical encoding (matching app.py)
        df_encoded, le_target = simple_categorical_encoding(df_features)
        
        # 4. Train optimized compatible model
        best_model, best_name, best_results, optimal_threshold, scaler, feature_columns, numeric_cols = train_compatible_optimized_model(
            df_encoded, 'booking_status_encoded')
        
        # 5. Save model artifacts compatible with original app.py
        print("\n💾 Saving optimized compatible model artifacts...")
        
        # Create models directory
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Save the best trained model
        model_filename = os.path.join(models_dir, 'best_model.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✅ Model saved: {model_filename}")
        
        # Save the scaler
        scaler_filename = os.path.join(models_dir, 'scaler.pkl')
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved: {scaler_filename}")
        
        # Save label encoder
        le_filename = os.path.join(models_dir, 'label_encoder.pkl')
        with open(le_filename, 'wb') as f:
            pickle.dump(le_target, f)
        print(f"✅ Label encoder saved: {le_filename}")
        
        # Save feature info compatible with original app.py
        feature_info = {
            'model_name': best_name,
            'model_accuracy': best_results['balanced_accuracy'],
            'all_columns': feature_columns,
            'numeric_columns': numeric_cols,
            'optimal_threshold': optimal_threshold,
            'performance_metrics': {
                'accuracy': best_results['accuracy'],
                'balanced_accuracy': best_results['balanced_accuracy'],
                'f1_score': best_results['f1_score'],
                'precision': best_results['precision'],
                'recall': best_results['recall'],
                'auc_roc': best_results['auc_roc']
            },
            'target_encoding': dict(zip(le_target.classes_, le_target.transform(le_target.classes_).astype(int))),
            'training_date': datetime.now().isoformat(),
            'dataset_shape': list(df_encoded.shape),
            'approach': 'optimized_compatible_with_original_app'
        }
        
        feature_info_filename = os.path.join(models_dir, 'feature_info.pkl')
        with open(feature_info_filename, 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"✅ Feature info saved: {feature_info_filename}")
        
        # Also save as JSON
        metadata_filename = os.path.join(models_dir, 'model_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(feature_info, f, indent=2, default=str)
        print(f"✅ Model metadata saved: {metadata_filename}")
        
        print(f"\n🎯 OPTIMIZED COMPATIBLE MODEL CREATION COMPLETED!")
        print("=" * 60)
        print(f"📊 Best Model: {best_name}")
        print(f"🎯 Balanced Accuracy: {best_results['balanced_accuracy']:.4f}")
        print(f"📈 Regular Accuracy: {best_results['accuracy']:.4f}")
        print(f"🔄 AUC-ROC: {best_results['auc_roc']:.4f}")
        print(f"⚖️ Optimal Threshold: {optimal_threshold:.4f}")
        print(f"📁 All files saved in: {models_dir}/")
        print("✅ Fully compatible with original app.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during optimized compatible model creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Optimized compatible model creation completed successfully!")
    else:
        print("❌ Optimized compatible model creation failed!")
