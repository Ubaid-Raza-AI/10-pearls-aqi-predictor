"""
Optimized Inference Pipeline - Direct Predictions for 24h, 48h, 72h
"""

import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import joblib
import json

from feature_pipeline import AQIFeatureEngineer  # Import from the correct file

load_dotenv()


class AQIInferencePipeline:
    
    def __init__(self):
        self.hopsworks_api_key = os.getenv('HOPSWORKS_API_KEY')
        self.project_name = os.getenv('HOPSWORKS_PROJECT_NAME', 'ubaidrazaaqi')
        self.latitude = float(os.getenv('LATITUDE', '31.558'))
        self.longitude = float(os.getenv('LONGITUDE', '74.3507'))
        
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        self.feature_engineer = AQIFeatureEngineer()
        self.models = {}  # Store multiple models
        self.scalers = {}
        self.metadatas = {}
        self.fs = None  # Will store feature store connection
        
        print(f"✓ Initialized inference pipeline")
        print(f"✓ Location: {self.latitude}°N, {self.longitude}°E")
    
    def fetch_recent_data(self, lookback_hours=48):  # Changed to 48 hours
        # Calculate date range: end yesterday, start lookback from yesterday
        yesterday = datetime.now() - timedelta(days=1)
        end_date = yesterday
        start_date = end_date - timedelta(hours=lookback_hours)
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"\nFetching recent data from {start_str} to {end_str} (yesterday)")
        
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ["pm10", "pm2_5", "carbon_monoxide", "sulphur_dioxide", 
                      "nitrogen_dioxide", "ozone", "us_aqi"],
            "start_date": start_str,
            "end_date": end_str,
        }
        
        responses = self.openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Process hourly data
        hourly = response.Hourly()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "pm10": hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
            "carbon_monoxide": hourly.Variables(2).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(3).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(4).ValuesAsNumpy(),
            "ozone": hourly.Variables(5).ValuesAsNumpy(),
            "us_aqi": hourly.Variables(6).ValuesAsNumpy()
        }
        
        df = pd.DataFrame(data=hourly_data)
        print(f"✓ Fetched {len(df)} records")
        print(f"  Latest data point: {df['date'].max()}")
        print(f"  Latest AQI: {df['us_aqi'].iloc[-1]:.1f}")
        
        return df
    
    def load_model_locally(self, horizon):
        """Load model for specific horizon"""
        model_dir = f"aqi_model_{horizon}h"
        print(f"\nLoading {horizon}h model from {model_dir}")
        
        if not os.path.exists(model_dir):
            print(f"❌ Directory '{model_dir}' not found!")
            return False
        
        try:
            keras_path = os.path.join(model_dir, "model.h5")
            pkl_path = os.path.join(model_dir, "model.pkl")
            
            if os.path.exists(keras_path):
                from tensorflow import keras
                self.models[horizon] = keras.models.load_model(keras_path)
            elif os.path.exists(pkl_path):
                self.models[horizon] = joblib.load(pkl_path)
            else:
                print(f"❌ No model file found in {model_dir}")
                return False
            
            self.scalers[horizon] = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            
            with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
                self.metadatas[horizon] = json.load(f)
            
            print(f"✓ {horizon}h model loaded: {self.metadatas[horizon]['model_type']}")
            print(f"  Metrics - RMSE: {self.metadatas[horizon]['metrics']['RMSE']:.4f}, R²: {self.metadatas[horizon]['metrics']['R2']:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {horizon}h model: {e}")
            return False
    
    def connect_to_hopsworks(self):
        """Connect to Hopsworks Feature Store"""
        print(f"\nConnecting to Hopsworks...")
        import hopsworks
        project = hopsworks.login(api_key_value=self.hopsworks_api_key, project=self.project_name)
        self.fs = project.get_feature_store()
        print("✓ Connected to Hopsworks Feature Store")
        return self.fs
    
    def save_to_feature_store(self, df_features):
        """Save/append recent data to Hopsworks Feature Store"""
        print("\nSaving recent data to Feature Store...")
        
        try:
            # Connect to Hopsworks if not already connected
            if self.fs is None:
                self.connect_to_hopsworks()
            
            # Get the existing feature group
            fg = self.fs.get_feature_group(name="aqi_features", version=1)
            
            # Add NaN for target columns to match schema
            target_cols = ['target_aqi_1h', 'target_aqi_6h', 'target_aqi_12h', 
                          'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
            
            for col in target_cols:
                if col not in df_features.columns:
                    df_features[col] = np.nan
                df_features[col] = df_features[col].astype('float32')
            
            # Insert data (will append to existing data)
            fg.insert(df_features, write_options={"wait_for_job": False})
            
            print(f"✓ Appended {len(df_features)} records to Feature Store")
            
        except Exception as e:
            print(f"⚠ Could not save to Feature Store: {e}")
            print("  (Continuing with predictions...)")
    
    def predict_all_horizons(self, df_features):
        """Predict for 24h, 48h, 72h using respective models"""
        predictions = []
        latest_row = df_features.iloc[-1:].copy()
        current_time = pd.to_datetime(latest_row['date'].values[0])
        current_aqi = latest_row['us_aqi'].values[0]
        
        horizons = [24, 48, 72]
        
        for horizon in horizons:
            # Load model if not already loaded
            if horizon not in self.models:
                if not self.load_model_locally(horizon):
                    print(f"❌ Skipping {horizon}h prediction due to load failure")
                    continue
            
            metadata = self.metadatas[horizon]
            feature_names = metadata['feature_names']
            
            # Prepare features
            X = latest_row[feature_names].fillna(0)  # Fill any missing for prediction
            X_scaled = self.scalers[horizon].transform(X)
            
            # Predict
            model = self.models[horizon]
            if hasattr(model, 'predict') and 'keras' in str(type(model)).lower():
                prediction = model.predict(X_scaled, verbose=0).flatten()[0]
            else:
                prediction = model.predict(X_scaled)[0]
            
            predictions.append({
                'horizon': f'{horizon}h',
                'hours_ahead': horizon,
                'prediction_time': current_time + timedelta(hours=horizon),
                'predicted_aqi': float(prediction)  # Ensure float for display
            })
            
            print(f"  ✓ {horizon}h Prediction: {prediction:.1f} (Model: {metadata['model_type']})")
        
        return {
            'current_time': current_time,
            'current_aqi': current_aqi,
            'predictions': predictions
        }
    
    def get_aqi_category(self, aqi):
        if aqi <= 50:
            return "Good", "🟢", "Air quality is satisfactory"
        elif aqi <= 100:
            return "Moderate", "🟡", "Acceptable for most people"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "🟠", "Sensitive groups may be affected"
        elif aqi <= 200:
            return "Unhealthy", "🔴", "Everyone may experience health effects"
        elif aqi <= 300:
            return "Very Unhealthy", "🟣", "Health alert"
        else:
            return "Hazardous", "🟤", "Emergency conditions"
    
    def display_prediction(self, result):
        print("\n" + "="*80)
        print("AQI 3-DAY FORECAST")
        print("="*80)
        
        curr_cat, curr_emoji, curr_desc = self.get_aqi_category(result['current_aqi'])
        print(f"\n📍 Latest Value in Data:")
        print(f"  Time: {result['current_time'].strftime('%Y-%m-%d %H:%M')}")
        print(f"  AQI: {result['current_aqi']:.1f} {curr_emoji} {curr_cat}")
        print(f"  {curr_desc}")
        
        print(f"\n🔮 3-DAY FORECAST:")
        print(f"{'Horizon':<10} {'Time':<20} {'AQI':<8} {'Category':<30} {'Change'}")
        print("-" * 80)
        
        for pred in result['predictions']:
            pred_aqi = pred['predicted_aqi']
            pred_cat, pred_emoji, pred_desc = self.get_aqi_category(pred_aqi)
            pred_time_str = pred['prediction_time'].strftime('%Y-%m-%d %H:%M')
            
            change = pred_aqi - result['current_aqi']
            change_str = f"{change:+.1f}"
            
            print(f"{pred['horizon']:<10} {pred_time_str:<20} {pred_aqi:<7.1f} {pred_emoji} {pred_cat:<25} {change_str}")
        
        # Overall trend
        changes = [p['predicted_aqi'] - result['current_aqi'] for p in result['predictions']]
        avg_change = np.mean(changes)
        if avg_change > 5:
            trend = "📈 Overall: Worsening air quality expected"
        elif avg_change < -5:
            trend = "📉 Overall: Improving air quality expected"
        else:
            trend = "➡️ Overall: Stable air quality expected"
        print(f"\n{trend}")
        
        print("\n" + "="*80)
    
    def save_prediction(self, result):
        
        csv_dir = "csv_files"
        os.makedirs(csv_dir, exist_ok=True)
        rows = []
        for pred in result['predictions']:
            rows.append({
                'timestamp': datetime.now(),
                'current_time': result['current_time'],
                'current_aqi': result['current_aqi'],
                'horizon': pred['horizon'],
                'prediction_time': pred['prediction_time'],
                'predicted_aqi': pred['predicted_aqi'],
                'hours_ahead': pred['hours_ahead']
            })
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(csv_dir, 'aqi_predictions_3day.csv')
        
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Saved predictions to {csv_path}")
    
    def run_inference(self, save_log=True):
        print("="*80)
        print("AQI INFERENCE - 3-DAY FORECAST (24h, 48h, 72h)")
        print("="*80)
        
        df_raw = self.fetch_recent_data(lookback_hours=48)
        
        print("\nEngineering features...")
        df_features = self.feature_engineer.engineer_features(df_raw)
        print(f"✓ Features ready ({len(df_features.columns)} total)")
        
        # Save to Feature Store
        self.save_to_feature_store(df_features)
        
        print("\nLoading models and generating predictions...")
        result = self.predict_all_horizons(df_features)
        
        if not result['predictions']:
            print("❌ No predictions generated. Ensure model directories exist: aqi_model_24h/, aqi_model_48h/, aqi_model_72h/")
            return None
        
        self.display_prediction(result)
        
        if save_log:
            self.save_prediction(result)
        
        print("\n" + "="*80)
        print("INFERENCE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return result


if __name__ == "__main__":
    try:
        inference = AQIInferencePipeline()
        result = inference.run_inference(save_log=True)
        
        if result:
            print("\n💡 Run this hourly for updated forecasts!")
            print("💡 Check 'aqi_predictions_3day.csv' for historical predictions")
            
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()