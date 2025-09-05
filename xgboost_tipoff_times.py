import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# NBA Arena mapping - maps team names to their home arenas
NBA_ARENAS = {
    "Atlanta Hawks": "State Farm Arena",
    "Boston Celtics": "TD Garden",
    "Brooklyn Nets": "Barclays Center",
    "Charlotte Hornets": "Spectrum Center",
    "Chicago Bulls": "United Center",
    "Cleveland Cavaliers": "Rocket Mortgage FieldHouse",
    "Dallas Mavericks": "American Airlines Center",
    "Denver Nuggets": "Ball Arena",
    "Detroit Pistons": "Little Caesars Arena",
    "Golden State Warriors": "Chase Center",
    "Houston Rockets": "Toyota Center",
    "Indiana Pacers": "Gainbridge Fieldhouse",
    "Los Angeles Clippers": "Crypto.com Arena",
    "Los Angeles Lakers": "Crypto.com Arena",
    "Memphis Grizzlies": "FedExForum",
    "Miami Heat": "Kaseya Center",  # Updated from FTX Arena
    "Milwaukee Bucks": "Fiserv Forum",
    "Minnesota Timberwolves": "Target Center",
    "New Orleans Pelicans": "Smoothie King Center",
    "New York Knicks": "Madison Square Garden",
    "Oklahoma City Thunder": "Paycom Center",
    "Orlando Magic": "Amway Center",
    "Philadelphia 76ers": "Wells Fargo Center",
    "Phoenix Suns": "Footprint Center",
    "Portland Trail Blazers": "Moda Center",
    "Sacramento Kings": "Golden 1 Center",
    "San Antonio Spurs": "AT&T Center",
    "Toronto Raptors": "Scotiabank Arena",
    "Utah Jazz": "Vivint Arena",
    "Washington Wizards": "Capital One Arena"
}

# Arena time zones
ARENA_TIMEZONES = {
    "State Farm Arena": "Eastern",
    "TD Garden": "Eastern",
    "Barclays Center": "Eastern",
    "Spectrum Center": "Eastern",
    "United Center": "Central",
    "Rocket Mortgage FieldHouse": "Eastern",
    "American Airlines Center": "Central",
    "Ball Arena": "Mountain",
    "Little Caesars Arena": "Eastern",
    "Chase Center": "Pacific",
    "Toyota Center": "Central",
    "Gainbridge Fieldhouse": "Eastern",
    "Crypto.com Arena": "Pacific",
    "FedExForum": "Central",
    "Kaseya Center": "Eastern",
    "Fiserv Forum": "Central",
    "Target Center": "Central",
    "Smoothie King Center": "Central",
    "Madison Square Garden": "Eastern",
    "Paycom Center": "Central",
    "Amway Center": "Eastern",
    "Wells Fargo Center": "Eastern",
    "Footprint Center": "Mountain",
    "Moda Center": "Pacific",
    "Golden 1 Center": "Pacific",
    "AT&T Center": "Central",
    "Scotiabank Arena": "Eastern",
    "Vivint Arena": "Mountain",
    "Capital One Arena": "Eastern"
}

def load_and_process_nba_data(file_path='Actual_nba_start_times.csv'):
    """
    Load and process NBA start times data from actual NBA CSV file
    Expected columns: Date, Away_Team, Home_Team, Tip_Off_Time, Arena (optional)
    """
    print("=== LOADING NBA DATA ===")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows of data")
        print(f"Columns: {list(df.columns)}")
        
        # Display first few rows to understand structure
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Try to detect common column patterns and standardize names
        df_columns_lower = [col.lower().strip() for col in df.columns]
        
        # Map common variations to standard names
        column_mapping = {}
        
        # Date columns
        date_variations = ['date', 'game_date', 'gamedate', 'game date']
        for i, col_lower in enumerate(df_columns_lower):
            if any(var in col_lower for var in date_variations):
                column_mapping[df.columns[i]] = 'Date'
                break
        
        # Away team columns
        away_variations = ['away', 'visitor', 'visiting', 'team1', 'away_team', 'awayteam']
        for i, col_lower in enumerate(df_columns_lower):
            if any(var in col_lower for var in away_variations):
                column_mapping[df.columns[i]] = 'Away_Team'
                break
        
        # Home team columns
        home_variations = ['home', 'team2', 'home_team', 'hometeam']
        for i, col_lower in enumerate(df_columns_lower):
            if any(var in col_lower for var in home_variations):
                column_mapping[df.columns[i]] = 'Home_Team'
                break
        
        # Time columns
        time_variations = ['time', 'tip_off', 'tipoff', 'start_time', 'game_time', 'tip off time', 'start time']
        for i, col_lower in enumerate(df_columns_lower):
            if any(var in col_lower for var in time_variations):
                column_mapping[df.columns[i]] = 'Tip_Off_Time'
                break
        
        # Arena columns (optional)
        arena_variations = ['arena', 'venue', 'stadium']
        for i, col_lower in enumerate(df_columns_lower):
            if any(var in col_lower for var in arena_variations):
                column_mapping[df.columns[i]] = 'Arena'
                break
        
        # Apply column mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"Column mapping applied: {column_mapping}")
        
        # Ensure we have required columns
        required_columns = ['Date', 'Away_Team', 'Home_Team', 'Tip_Off_Time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"WARNING: Missing required columns: {missing_columns}")
            print("Please ensure your CSV has columns for: Date, Away Team, Home Team, Tip-off Time")
            
            # Try to infer from available columns
            if len(df.columns) >= 4:
                print("Attempting to map columns by position...")
                df.columns = ['Date', 'Away_Team', 'Home_Team', 'Tip_Off_Time'] + list(df.columns[4:])
        
        return df
        
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found!")
        print("Please ensure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None

def determine_arena_from_home_team(home_team):
    """
    Get arena and timezone information from home team
    """
    arena = NBA_ARENAS.get(home_team, "Unknown Arena")
    timezone = ARENA_TIMEZONES.get(arena, "Unknown")
    return arena, timezone

def create_features_and_target(df):
    """
    Create features and target variable from NBA data, including arena information
    """
    print("\n=== CREATING FEATURES AND TARGET ===")
    
    if df is None:
        return None, None
    
    # Create a copy to work with
    data = df.copy()
    
    # Clean team names (remove extra spaces, fix common variations)
    if 'Away_Team' in data.columns:
        data['Away_Team'] = data['Away_Team'].astype(str).str.strip()
    if 'Home_Team' in data.columns:
        data['Home_Team'] = data['Home_Team'].astype(str).str.strip()
    
    # Parse the date
    print("Processing dates...")
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek  # Monday=0, Sunday=6
    data['DayOfYear'] = data['Date'].dt.dayofyear
    
    # Parse tip-off time and convert to minutes after midnight
    print("Processing tip-off times...")
    
    def parse_time_to_minutes(time_str):
        try:
            if pd.isna(time_str):
                return np.nan
            
            # Remove extra spaces and convert to string
            time_str = str(time_str).strip()
            
            # Handle different time formats
            # Try common formats: "7:30 PM", "19:30", "7:30PM", "730 PM", etc.
            formats_to_try = [
                '%I:%M %p',      # 7:30 PM
                '%I:%M%p',       # 7:30PM
                '%I%p',          # 7PM
                '%H:%M',         # 19:30 (24-hour)
                '%H:%M:%S',      # 19:30:00
                '%I %p',         # 7 PM
            ]
            
            for fmt in formats_to_try:
                try:
                    time_obj = pd.to_datetime(time_str, format=fmt, errors='raise')
                    minutes = time_obj.hour * 60 + time_obj.minute
                    return minutes
                except:
                    continue
            
            # If no format worked, try pandas' flexible parsing
            try:
                time_obj = pd.to_datetime(time_str, errors='coerce')
                if not pd.isna(time_obj):
                    minutes = time_obj.hour * 60 + time_obj.minute
                    return minutes
            except:
                pass
            
            return np.nan
            
        except:
            return np.nan
    
    data['Tip_Off_Minutes'] = data['Tip_Off_Time'].apply(parse_time_to_minutes)
    
    # Process arena information
    print("Processing arena information...")
    
    # If arena column exists in data, use it; otherwise, infer from home team
    if 'Arena' in data.columns and not data['Arena'].isna().all():
        # Use provided arena information
        data['Arena'] = data['Arena'].fillna('Unknown Arena')
        # Map timezone from arena
        data['Arena_Timezone'] = data['Arena'].apply(
            lambda arena: ARENA_TIMEZONES.get(arena, "Unknown")
        )
    else:
        # Infer arena from home team
        arena_info = data['Home_Team'].apply(determine_arena_from_home_team)
        data['Arena'] = [info[0] for info in arena_info]
        data['Arena_Timezone'] = [info[1] for info in arena_info]
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    encoders = {}
    
    # Prepare categorical columns
    categorical_columns = {
        'Away_Team': 'away_team',
        'Home_Team': 'home_team',
        'Arena': 'arena',
        'Arena_Timezone': 'timezone'
    }
    
    for col_name, encoder_key in categorical_columns.items():
        if col_name in data.columns:
            # Fill NaN values
            data[col_name] = data[col_name].fillna('Unknown')
            
            # Create and fit encoder
            le = LabelEncoder()
            data[f'{col_name}_Encoded'] = le.fit_transform(data[col_name])
            encoders[encoder_key] = le
    
    # Create additional features
    print("Creating additional features...")
    
    # Weekend indicator
    data['Is_Weekend'] = (data['DayOfWeek'] >= 5).astype(int)  # Saturday=5, Sunday=6
    
    # Season features (NBA season typically starts in October)
    data['Is_Season_Start'] = ((data['Month'] >= 10) | (data['Month'] <= 2)).astype(int)
    
    # Holiday proximity (simplified)
    data['Is_Holiday_Season'] = ((data['Month'] == 12) | 
                                ((data['Month'] == 1) & (data['Day'] <= 7))).astype(int)
    
    # Create interaction features (safely)
    if 'Away_Team_Encoded' in data.columns and 'Home_Team_Encoded' in data.columns:
        data['Team_Matchup'] = data['Away_Team_Encoded'] * 100 + data['Home_Team_Encoded']
    
    if 'Arena_Encoded' in data.columns:
        data['Arena_Day_Interaction'] = data['Arena_Encoded'] * 10 + data['DayOfWeek']
        data['Arena_Month_Interaction'] = data['Arena_Encoded'] * 100 + data['Month']
    
    # Time zone numeric encoding
    timezone_numeric = {
        'Eastern': 1, 'Central': 2, 'Mountain': 3, 'Pacific': 4, 'Unknown': 0
    }
    data['Timezone_Numeric'] = data['Arena_Timezone'].map(timezone_numeric).fillna(0)
    
    # Game type features (based on day of week and time patterns)
    data['Is_Prime_Time'] = ((data['DayOfWeek'] >= 4) & (data['DayOfWeek'] <= 6)).astype(int)  # Fri-Sun
    data['Is_Matinee'] = ((data['DayOfWeek'] == 6) | (data['DayOfWeek'] == 0)).astype(int)    # Sun, Mon
    
    print(f"Created features including arena data. Data shape: {data.shape}")
    print(f"Unique arenas: {data['Arena'].nunique()}")
    print(f"Valid tip-off times: {data['Tip_Off_Minutes'].notna().sum()}/{len(data)}")
    
    # Show some statistics
    if data['Tip_Off_Minutes'].notna().any():
        valid_times = data['Tip_Off_Minutes'].dropna()
        print(f"Tip-off time range: {valid_times.min():.0f}-{valid_times.max():.0f} minutes")
        print(f"Average tip-off: {valid_times.mean():.1f} minutes ({valid_times.mean()/60:.1f} hours)")
    
    return data, encoders

def diagnose_and_clean_data(X, y):
    """
    Diagnose and clean data for XGBoost training
    """
    print("\n=== DIAGNOSTIC RESULTS ===")
    
    # Check target variable
    print(f"Target variable (Tip_Off_Minutes) info:")
    print(f"  - Shape: {y.shape}")
    print(f"  - Data type: {y.dtype}")
    print(f"  - NaN values: {y.isna().sum()}")
    print(f"  - Infinite values: {np.isinf(y).sum() if len(y) > 0 else 0}")
    
    if not y.empty and y.notna().any():
        valid_y = y.dropna()
        if len(valid_y) > 0:
            print(f"  - Value range: {valid_y.min():.0f} to {valid_y.max():.0f} minutes")
            print(f"  - Mean: {valid_y.mean():.2f} minutes ({valid_y.mean()/60:.1f} hours)")
            print(f"  - Median: {valid_y.median():.2f} minutes ({valid_y.median()/60:.1f} hours)")
    
    # Check features
    print(f"\nFeatures (X) info:")
    print(f"  - Shape: {X.shape}")
    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"  - Columns with NaN values:")
        for col, count in nan_cols.items():
            print(f"    {col}: {count}")
    else:
        print("  - No NaN values in features")
    
    # Clean the data
    print("\n=== CLEANING DATA ===")
    
    # Remove rows where target is NaN or infinite
    mask = ~(y.isna() | np.isinf(y))
    X_clean = X[mask].copy()
    y_clean = y[mask].copy()
    
    removed_rows = len(y) - len(y_clean)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with invalid target values")
    
    # Handle NaN values in features
    numeric_features = X_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_features) > 0:
        # Fill NaN with median for numeric features
        for col in numeric_features:
            if X_clean[col].isna().sum() > 0:
                median_val = X_clean[col].median()
                X_clean[col] = X_clean[col].fillna(median_val)
                print(f"Filled {col} NaN values with median: {median_val}")
        
        # Handle infinite values in features
        inf_mask = np.isinf(X_clean[numeric_features]).any(axis=1)
        if inf_mask.sum() > 0:
            print(f"Removing {inf_mask.sum()} rows with infinite feature values")
            X_clean = X_clean[~inf_mask]
            y_clean = y_clean[~inf_mask]
    
    print(f"Final cleaned data shape: X={X_clean.shape}, y={y_clean.shape}")
    
    return X_clean, y_clean

def train_xgboost_model(X, y, test_size=0.2, random_state=42):
    """
    Complete pipeline to train XGBoost model for NBA tip-off prediction
    """
    # Clean the data
    X_clean, y_clean = diagnose_and_clean_data(X, y)
    
    if len(X_clean) == 0:
        print("ERROR: No valid data remaining after cleaning!")
        return None, None, None, None, None, None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )
    
    # Final check before training
    print(f"\n=== FINAL TRAINING DATA CHECK ===")
    print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"NaN in y_train: {y_train.isna().sum()}")
    print(f"NaN in X_train: {X_train.isna().sum().sum()}")
    print(f"Infinite in y_train: {np.isinf(y_train).sum()}")
    
    # Train the model
    print(f"\n=== TRAINING XGBOOST MODEL ===")
    
    # XGBoost parameters optimized for this use case
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    try:
        model.fit(X_train, y_train)
        print("Model trained successfully!")
    except Exception as e:
        print(f"ERROR training model: {e}")
        return None, None, None, None, None, None
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Evaluate model performance
    print(f"\n=== MODEL PERFORMANCE ===")
    
    # Training metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    
    # Test metrics
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"Training Performance:")
    print(f"  - MSE: {train_mse:.2f}")
    print(f"  - MAE: {train_mae:.2f} minutes ({train_mae:.1f} minutes)")
    print(f"  - R²: {train_r2:.4f}")
    
    print(f"Test Performance:")
    print(f"  - MSE: {test_mse:.2f}")
    print(f"  - MAE: {test_mae:.2f} minutes ({test_mae:.1f} minutes)")
    print(f"  - R²: {test_r2:.4f}")
    
    # Feature importance
    print(f"\n=== TOP 15 FEATURE IMPORTANCE ===")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15))
    
    return model, X_train, X_test, y_train, y_test, test_predictions

def predict_future_game(model, encoders, date, away_team, home_team, feature_columns):
    """
    Predict tip-off time for a future game
    """
    try:
        # Parse the date
        game_date = pd.to_datetime(date, errors='coerce')
        if pd.isna(game_date):
            return None, "Invalid date format", None, None
        
        # Get arena information from home team
        arena, timezone = determine_arena_from_home_team(home_team)
        
        # Create feature dictionary
        features = {}
        features['Year'] = game_date.year
        features['Month'] = game_date.month
        features['Day'] = game_date.day
        features['DayOfWeek'] = game_date.dayofweek
        features['DayOfYear'] = game_date.dayofyear
        features['Is_Weekend'] = 1 if game_date.dayofweek >= 5 else 0
        features['Is_Season_Start'] = 1 if (game_date.month >= 10 or game_date.month <= 2) else 0
        features['Is_Holiday_Season'] = 1 if (game_date.month == 12 or (game_date.month == 1 and game_date.day <= 7)) else 0
        features['Is_Prime_Time'] = 1 if (game_date.dayofweek >= 4 and game_date.dayofweek <= 6) else 0
        features['Is_Matinee'] = 1 if (game_date.dayofweek == 6 or game_date.dayofweek == 0) else 0
        
        # Handle team and arena encoding
        try:
            away_team_encoded = encoders['away_team'].transform([away_team])[0]
        except (ValueError, KeyError):
            away_team_encoded = 0
            
        try:
            home_team_encoded = encoders['home_team'].transform([home_team])[0]
        except (ValueError, KeyError):
            home_team_encoded = 0
            
        try:
            arena_encoded = encoders['arena'].transform([arena])[0]
        except (ValueError, KeyError):
            arena_encoded = 0
            
        try:
            timezone_encoded = encoders['timezone'].transform([timezone])[0]
        except (ValueError, KeyError):
            timezone_encoded = 0
        
        # Add encoded features
        features['Away_Team_Encoded'] = away_team_encoded
        features['Home_Team_Encoded'] = home_team_encoded
        features['Arena_Encoded'] = arena_encoded
        features['Arena_Timezone_Encoded'] = timezone_encoded
        
        # Add interaction features (if they exist in the model)
        if 'Team_Matchup' in feature_columns:
            features['Team_Matchup'] = away_team_encoded * 100 + home_team_encoded
        if 'Arena_Day_Interaction' in feature_columns:
            features['Arena_Day_Interaction'] = arena_encoded * 10 + game_date.dayofweek
        if 'Arena_Month_Interaction' in feature_columns:
            features['Arena_Month_Interaction'] = arena_encoded * 100 + game_date.month
        
        # Time zone numeric
        timezone_numeric = {'Eastern': 1, 'Central': 2, 'Mountain': 3, 'Pacific': 4, 'Unknown': 0}
        features['Timezone_Numeric'] = timezone_numeric.get(timezone, 0)
        
        # Create DataFrame with the correct feature order
        feature_df = pd.DataFrame([features])
        
        # Ensure all expected columns are present
        for col in feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        feature_df = feature_df[feature_columns]
        
        # Make prediction
        predicted_minutes = model.predict(feature_df)[0]
        
        # Convert back to time format
        hours = int(predicted_minutes // 60)
        minutes = int(predicted_minutes % 60)
        
        # Convert to 12-hour format
        if hours == 0:
            time_str = f"12:{minutes:02d} AM"
        elif hours < 12:
            time_str = f"{hours}:{minutes:02d} AM"
        elif hours == 12:
            time_str = f"12:{minutes:02d} PM"
        else:
            time_str = f"{hours-12}:{minutes:02d} PM"
            
        return predicted_minutes, time_str, arena, timezone
        
    except Exception as e:
        return None, f"Prediction error: {e}", None, None

def create_sample_csv():
    """
    Create a sample CSV file with the expected format
    """
    print("\n=== CREATING SAMPLE CSV ===")
    
    sample_data = {
        'Date': [
            '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19',
            '2024-01-20', '2024-01-21', '2024-01-22', '2024-01-23', '2024-01-24'
        ],
        'Away_Team': [
            'Boston Celtics', 'Los Angeles Lakers', 'Golden State Warriors', 
            'Miami Heat', 'Denver Nuggets', 'Phoenix Suns', 'Milwaukee Bucks',
            'Philadelphia 76ers', 'Dallas Mavericks', 'Brooklyn Nets'
        ],
        'Home_Team': [
            'New York Knicks', 'Chicago Bulls', 'Portland Trail Blazers',
            'Orlando Magic', 'Utah Jazz', 'Sacramento Kings', 'Atlanta Hawks',
            'Washington Wizards', 'San Antonio Spurs', 'Detroit Pistons'
        ],
        'Tip_Off_Time': [
            '7:30 PM', '8:00 PM', '10:00 PM', '7:00 PM', '9:00 PM',
            '6:00 PM', '7:30 PM', '7:00 PM', '8:30 PM', '7:00 PM'
        ]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_nba_start_times.csv', index=False)
    print("Sample CSV created as 'sample_nba_start_times.csv'")
    print("Columns: Date, Away_Team, Home_Team, Tip_Off_Time")
    return sample_df

def main():
    """
    Main function demonstrating the complete workflow
    """
    print("=== NBA TIP-OFF TIME PREDICTION (UPDATED VERSION) ===\n")
    
    # Try to load the actual data
    df = load_and_process_nba_data('Actual_nba_start_times.csv')
    
    # If file doesn't exist, create a sample and use it
    if df is None:
        print("Creating sample data for demonstration...")
        df = create_sample_csv()
    
    # Process the data
    processed_df, encoders = create_features_and_target(df)
    
    if processed_df is None:
        print("ERROR: Could not process data")
        return None, None, None
    
    # Define feature columns (based on what we created)
    feature_columns = []
    
    # Add basic date features
    basic_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 
                     'Is_Weekend', 'Is_Season_Start', 'Is_Holiday_Season',
                     'Is_Prime_Time', 'Is_Matinee', 'Timezone_Numeric']
    
    # Add encoded features that exist
    encoded_features = ['Away_Team_Encoded', 'Home_Team_Encoded', 
                       'Arena_Encoded', 'Arena_Timezone_Encoded']
    
    # Add interaction features that exist
    interaction_features = ['Team_Matchup', 'Arena_Day_Interaction', 'Arena_Month_Interaction']
    
    # Combine all features that actually exist in the data
    all_possible_features = basic_features + encoded_features + interaction_features
    feature_columns = [col for col in all_possible_features if col in processed_df.columns]
    
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    X = processed_df[feature_columns]
    y = processed_df['Tip_Off_Minutes']
    
    # Train