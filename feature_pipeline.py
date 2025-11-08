"""
AQI Feature Pipeline - Complete Script with Reusable Feature Engineering Class
Fetches air quality data, engineers features, and stores in Hopsworks Feature Store
"""
import argparse
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import hopsworks
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AQIFeatureEngineer:
    """
    Reusable feature engineering class for AQI prediction
    """
    
    def engineer_features(self, df):
        """
        Engineer all features from raw data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw data with columns: date, pm10, pm2_5, carbon_monoxide, 
            sulphur_dioxide, nitrogen_dioxide, ozone, us_aqi
        
        Returns:
        --------
        pd.DataFrame : Engineered features
        """
        df = df.copy()
        
        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # ==================== TIME-BASED FEATURES ====================
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)  # 1:Winter, 2:Spring, 3:Summer, 4:Fall
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # ==================== LAG FEATURES ====================
        # Previous values of AQI (target variable)
        for lag in [1, 3, 6, 12, 24]:
            df[f'aqi_lag_{lag}h'] = df['us_aqi'].shift(lag)
        
        # Previous values of key pollutants
        for pollutant in ['pm2_5', 'pm10']:
            for lag in [1, 6, 24]:
                df[f'{pollutant}_lag_{lag}h'] = df[pollutant].shift(lag)
        
        # ==================== ROLLING STATISTICS ====================
        # Rolling windows for AQI
        for window in [3, 6, 12, 24]:
            df[f'aqi_rolling_mean_{window}h'] = df['us_aqi'].rolling(window=window, min_periods=1).mean()
            df[f'aqi_rolling_std_{window}h'] = df['us_aqi'].rolling(window=window, min_periods=1).std()
            df[f'aqi_rolling_min_{window}h'] = df['us_aqi'].rolling(window=window, min_periods=1).min()
            df[f'aqi_rolling_max_{window}h'] = df['us_aqi'].rolling(window=window, min_periods=1).max()
        
        # Rolling statistics for PM2.5
        for window in [6, 24]:
            df[f'pm2_5_rolling_mean_{window}h'] = df['pm2_5'].rolling(window=window, min_periods=1).mean()
            df[f'pm2_5_rolling_std_{window}h'] = df['pm2_5'].rolling(window=window, min_periods=1).std()
        
        # ==================== DERIVED FEATURES ====================
        # AQI change rate
        df['aqi_change_1h'] = df['us_aqi'].diff(1)
        df['aqi_change_3h'] = df['us_aqi'].diff(3)
        df['aqi_change_24h'] = df['us_aqi'].diff(24)
        df['aqi_change_rate_1h'] = df['aqi_change_1h'] / (df['us_aqi'].shift(1) + 1)
        
        # Pollutant ratios (important for understanding pollution sources)
        df['pm2_5_to_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1)
        df['no2_to_co_ratio'] = df['nitrogen_dioxide'] / (df['carbon_monoxide'] + 1)
        
        # Total pollutant load
        df['total_particulates'] = df['pm2_5'] + df['pm10']
        df['total_gases'] = df['carbon_monoxide'] + df['nitrogen_dioxide'] + df['sulphur_dioxide'] + df['ozone']
        
        # ==================== INTERACTION FEATURES ====================
        # Hour-based patterns for key pollutants
        df['pm2_5_x_hour'] = df['pm2_5'] * df['hour']
        df['pm10_x_hour'] = df['pm10'] * df['hour']
        
        # Weekend vs weekday pollution patterns
        df['pm2_5_x_weekend'] = df['pm2_5'] * df['is_weekend']
        df['aqi_x_weekend'] = df['us_aqi'] * df['is_weekend']
        
        # ==================== STATISTICAL FEATURES ====================
        # Exponential moving averages
        df['aqi_ema_12h'] = df['us_aqi'].ewm(span=12, adjust=False).mean()
        df['aqi_ema_24h'] = df['us_aqi'].ewm(span=24, adjust=False).mean()
        
        # Momentum indicators
        df['aqi_momentum_6h'] = df['us_aqi'] - df['us_aqi'].shift(6)
        df['aqi_momentum_24h'] = df['us_aqi'] - df['us_aqi'].shift(24)
        
        # Fill NaN values created by lag and rolling operations
        # Forward fill for initial rows
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df


class AQIFeaturePipeline:
    """
    Feature Pipeline for AQI Prediction
    Fetches raw data, engineers features, and stores in Hopsworks
    """
    
    def __init__(self):
        """Initialize with Hopsworks credentials from environment variables"""
        self.hopsworks_api_key = os.getenv('HOPSWORKS_API_KEY')
        self.project_name = os.getenv('HOPSWORKS_PROJECT_NAME', 'ubaidrazaaqi')
        self.latitude = float(os.getenv('LATITUDE', '31.558'))  # Lahore coordinates
        self.longitude = float(os.getenv('LONGITUDE', '74.3507'))
        
        if not self.hopsworks_api_key:
            raise ValueError("HOPSWORKS_API_KEY not found in .env file")
        
        # Setup Open-Meteo API client
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # Initialize feature engineer
        self.feature_engineer = AQIFeatureEngineer()
        
        print(f"✓ Initialized pipeline for project: {self.project_name}")
        print(f"✓ Location: {self.latitude}°N, {self.longitude}°E")
    
    def fetch_raw_data(self, start_date, end_date):
        """
        Fetch raw air quality data from Open-Meteo API
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        
        Returns:
        --------
        pd.DataFrame : Raw air quality data
        """
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ["pm10", "pm2_5", "carbon_monoxide", "sulphur_dioxide", 
                      "nitrogen_dioxide", "ozone", "us_aqi"],
            "start_date": start_date,
            "end_date": end_date,
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
        print(f"✓ Fetched {len(df)} records from {start_date} to {end_date}")
        return df
    
    def prepare_target_variable(self, df):
        """
        Create target variable for prediction
        We want to predict AQI for next 1, 6, 12, 24, and 72 hours
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature engineered data
        
        Returns:
        --------
        pd.DataFrame : Data with target variables
        """
        df = df.copy()
        
        # Create target variables (future AQI values)
        df['target_aqi_1h'] = df['us_aqi'].shift(-1)
        df['target_aqi_6h'] = df['us_aqi'].shift(-6)
        df['target_aqi_12h'] = df['us_aqi'].shift(-12)
        df['target_aqi_24h'] = df['us_aqi'].shift(-24)
        df['target_aqi_48h'] = df['us_aqi'].shift(-48)
        df['target_aqi_72h'] = df['us_aqi'].shift(-72)
        
        # Drop rows where we can't predict (last 72 hours)
        df = df[:-72]
        
        print(f"✓ Created target variables, final dataset: {len(df)} records")
        
        return df
    
    def connect_to_hopsworks(self):
        """Connect to Hopsworks and get feature store"""
        print(f"\nConnecting to Hopsworks project: {self.project_name}")
        
        project = hopsworks.login(
            api_key_value=self.hopsworks_api_key,
            project=self.project_name
        )
        
        fs = project.get_feature_store()
        print("✓ Connected to Hopsworks Feature Store")
        
        return project, fs
    
    def create_feature_group(self, fs, df, feature_group_name="aqi_features", version=1):
        """
        Create or get feature group in Hopsworks
        
        Parameters:
        -----------
        fs : FeatureStore
            Hopsworks feature store object
        df : pd.DataFrame
            Data to store
        feature_group_name : str
            Name of the feature group
        version : int
            Version of the feature group
        
        Returns:
        --------
        FeatureGroup : Created or retrieved feature group
        """
        print(f"\nCreating feature group: {feature_group_name}_v{version}")
        
        # Ensure date column is in correct format
        df['date'] = pd.to_datetime(df['date'])
        
        try:
            # Get or create feature group
            fg = fs.get_or_create_feature_group(
                name=feature_group_name,
                version=version,
                primary_key=['date'],
                event_time='date',
                description="AQI prediction features with engineered time-based and derived features",
                online_enabled=False,
            )
            
            print(f"✓ Feature group '{feature_group_name}' (version {version}) ready")
            return fg
            
        except Exception as e:
            print(f"Error creating feature group: {e}")
            raise
    
    def insert_data_to_feature_store(self, fg, df):
        """
        Insert data into Hopsworks feature group
        
        Parameters:
        -----------
        fg : FeatureGroup
            Hopsworks feature group
        df : pd.DataFrame
            Data to insert
        """
        print(f"\nInserting {len(df)} records into feature store...")
        
        try:
            fg.insert(df, write_options={"wait_for_job": True})
            print("✓ Data successfully inserted into feature store")
            
        except Exception as e:
            print(f"Error inserting data: {e}")
            raise
    
    def run_pipeline(self, start_date, end_date, feature_group_name="aqi_features", version=1):
        """
        Run the complete feature pipeline
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        feature_group_name : str
            Name for the feature group
        version : int
            Version number
        """
        print("="*60)
        print("AQI FEATURE PIPELINE - STARTING")
        print("="*60)
        
        # Step 1: Fetch raw data
        print("\n[1/6] Fetching raw data from Open-Meteo API...")
        df_raw = self.fetch_raw_data(start_date, end_date)
        
        # Step 2: Engineer features using the reusable class
        print("\n[2/6] Engineering features...")
        df_features = self.feature_engineer.engineer_features(df_raw)
        print(f"✓ Engineered {len(df_features.columns)} features")
        
        # Step 3: Create target variables
        print("\n[3/6] Creating target variables...")
        df_final = self.prepare_target_variable(df_features)
        
        # Step 4: Connect to Hopsworks
        print("\n[4/6] Connecting to Hopsworks...")
        project, fs = self.connect_to_hopsworks()
        
        # Step 5: Create feature group
        print("\n[5/6] Setting up feature group...")
        fg = self.create_feature_group(fs, df_final, feature_group_name, version)
        
        # Step 6: Insert data
        print("\n[6/6] Storing features in Hopsworks...")
        self.insert_data_to_feature_store(fg, df_final)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nFeature Summary:")
        print(f"  • Total records: {len(df_final)}")
        print(f"  • Total features: {len(df_final.columns)}")
        print(f"  • Date range: {df_final['date'].min()} to {df_final['date'].max()}")
        print(f"  • Feature group: {feature_group_name}_v{version}")
        
        return df_final, fg


# ==================== MAIN EXECUTION ====================

# Replace the entire if __name__ == "__main__": block with this
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Feature Pipeline")
    parser.add_argument('--mode', choices=['historical', 'recent'], default='historical',
                        help="Mode: 'historical' for backfill or 'recent' for last 24h")
    args = parser.parse_args()
    
    try:
        pipeline = AQIFeaturePipeline()
        
        if args.mode == 'recent':
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d')
            print(f"Running recent mode: {start_date} to {end_date}")
        else:  # historical
            start_date = "2023-01-01"
            end_date = "2025-11-05"
            print(f"Running historical mode: {start_date} to {end_date}")
        
        df_features, feature_group = pipeline.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            feature_group_name="aqi_features",
            version=1
        )
        
        # Keep your existing display code here (SAMPLE FEATURES, etc.)
        print("\n" + "="*60)
        print("SAMPLE FEATURES (first 5 rows):")
        print("="*60)
        print(df_features.head())
        
        print("\n" + "="*60)
        print("FEATURE COLUMNS:")
        print("="*60)
        for i, col in enumerate(df_features.columns, 1):
            print(f"{i:3d}. {col}")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        raise