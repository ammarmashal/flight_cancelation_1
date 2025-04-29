import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Preprocessing functions
def process_missing_values(df, imputer_report=None):
    if imputer_report is None:
        imputer_report = {
            'dropped': [],
            'filled_0': [],
            'filled_N': [],
            'geo_imputed': {},
            'text_imputed': [],
            'single_impute': {},
            'complete_columns': []
        }

    df = df.drop(columns=['IATA_CODE_x'])
    imputer_report['dropped'].append('IATA_CODE_x')

    num_cols = df.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputer.fit(df[num_cols])

    delay_cols = ['AIR_SYSTEM_DELAY', 'WEATHER_DELAY', 'LATE_AIRCRAFT_DELAY', 
                    'AIRLINE_DELAY', 'SECURITY_DELAY']
    for col in delay_cols:
        df[col] = df[col].fillna(0)
    imputer_report['filled_0'].extend(delay_cols)

    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].fillna('N')
    imputer_report['filled_N'].append('CANCELLATION_REASON')

    geo_cols = {
        'DEST_LAT': df['DEST_LAT'].median(),
        'DEST_LON': df['DEST_LON'].mean(),
        'ORIGIN_LAT': df['ORIGIN_LAT'].median(),
        'ORIGIN_LON': df['ORIGIN_LON'].mean()
    }
    for col, val in geo_cols.items():
        df[col] = df[col].fillna(val)
    imputer_report['geo_imputed'] = geo_cols

    text_cols = ['IATA_CODE_y', 'IATA_CODE', 'DESTINATION_CITY', 'DESTINATION_STATE',
                'ORIGIN_CITY', 'ORIGIN_STATE', 'ORIGIN_AIRPORT_NORMALIZED',
                'DESTINATION_AIRPORT_NORMALIZED', 'TAIL_NUMBER']
    for col in text_cols:
        df[col] = df[col].fillna('UNKNOWN')
    imputer_report['text_imputed'] = text_cols

    scheduled_time_median = df['SCHEDULED_TIME'].median()
    df['SCHEDULED_TIME'] = df['SCHEDULED_TIME'].fillna(scheduled_time_median)
    imputer_report['single_impute']['SCHEDULED_TIME'] = scheduled_time_median

    complete_cols = ['MONTH', 'DESTINATION_AIRPORT_CODE', 'ORIGIN_AIRPORT_CODE',
                   'DAY', 'Flight_Status', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT',
                   'FLIGHT_NUMBER', 'AIRLINE', 'DISTANCE']
    imputer_report['complete_columns'] = complete_cols

    return df, imputer_report, imputer

def create_airport_clusters(df, n_clusters=100):
    origins = df[['ORIGIN_AIRPORT_NORMALIZED', 'ORIGIN_LAT', 'ORIGIN_LON']].drop_duplicates()
    destinations = df[['DESTINATION_AIRPORT_NORMALIZED', 'DEST_LAT', 'DEST_LON']].drop_duplicates()
    destinations.columns = ['ORIGIN_AIRPORT_NORMALIZED', 'ORIGIN_LAT', 'ORIGIN_LON']
    all_airports = pd.concat([origins, destinations]).drop_duplicates('ORIGIN_AIRPORT_NORMALIZED')

    coords = all_airports[['ORIGIN_LAT', 'ORIGIN_LON']].values
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    model.fit(coords)
    return model

def calculate_airport_importance(df):
    origin_counts = df['ORIGIN_AIRPORT_NORMALIZED'].value_counts()
    dest_counts = df['DESTINATION_AIRPORT_NORMALIZED'].value_counts()
    total_counts = origin_counts.add(dest_counts, fill_value=0)
    return {
        'large': set(total_counts.nlargest(20).index),
        'medium': set(total_counts.nlargest(100).index) - set(total_counts.nlargest(20).index),
        'small': set(total_counts.index) - set(total_counts.nlargest(100).index)
    }

def add_airport_features(df, cluster_model, importance_map):
    df['ORIGIN_CLUSTER'] = cluster_model.predict(df[['ORIGIN_LAT', 'ORIGIN_LON']].values)
    df['DEST_CLUSTER'] = cluster_model.predict(df[['DEST_LAT', 'DEST_LON']].values)

    df['ORIGIN_IMPORTANCE'] = df['ORIGIN_AIRPORT_NORMALIZED'].apply(
        lambda x: 'large' if x in importance_map['large'] else 
                  'medium' if x in importance_map['medium'] else 'small')
    df['DEST_IMPORTANCE'] = df['DESTINATION_AIRPORT_NORMALIZED'].apply(
        lambda x: 'large' if x in importance_map['large'] else 
                  'medium' if x in importance_map['medium'] else 'small')

    df['DISTANCE_KM'] = df.apply(lambda row: geodesic(
        (row['ORIGIN_LAT'], row['ORIGIN_LON']),
        (row['DEST_LAT'], row['DEST_LON'])).km, axis=1)

    df['ORIGIN_STATE'] = df['ORIGIN_STATE'].str.upper()
    df['DESTINATION_STATE'] = df['DESTINATION_STATE'].str.upper()

    df['SAME_STATE'] = (df['ORIGIN_STATE'] == df['DESTINATION_STATE']).astype(int)

    return df

def calculate_airline_stats(df):
    stats = df.groupby('AIRLINE')['Flight_Status'].agg(['mean', 'count'])
    stats.columns = ['airline_cancellation_rate', 'airline_flight_count']
    return stats

def add_airline_features(df, airline_stats):
    df = df.merge(airline_stats, how='left', left_on='AIRLINE', right_index=True)
    df['airline_cancellation_rate'].fillna(airline_stats['airline_cancellation_rate'].mean(), inplace=True)
    df['airline_flight_count'].fillna(1, inplace=True)
    return df

def add_date_features(df):
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']].assign(YEAR=2020))
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    df['SEASON'] = df['MONTH'].apply(lambda x: (x % 12 + 3) // 3)
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
    df.drop(columns=['DATE'], inplace=True)
    return df

def select_and_encode_features(df):
    features = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'SEASON', 'IS_WEEKEND',
        'SCHEDULED_TIME', 'DISTANCE_KM',
        'AIRLINE', 'airline_cancellation_rate', 'airline_flight_count',
        'ORIGIN_CLUSTER', 'DEST_CLUSTER', 
        'ORIGIN_IMPORTANCE', 'DEST_IMPORTANCE',
        'SAME_STATE'
    ]
    df = df[features]

    categoricals = ['AIRLINE', 'ORIGIN_IMPORTANCE', 'DEST_IMPORTANCE', 'SEASON']
    df = pd.get_dummies(df, columns=categoricals, drop_first=True)
    return df
# Pipeline function
def preprocess_pipeline(input_csv_path, output_file_path):
    df = pd.read_csv(input_csv_path)
    df, _, _ = process_missing_values(df)
    cluster_model = create_airport_clusters(df)
    importance_map = calculate_airport_importance(df)
    df = add_airport_features(df, cluster_model, importance_map)
    airline_stats = calculate_airline_stats(df)
    df = add_airline_features(df, airline_stats)
    if 'YEAR' not in df.columns:
        df['YEAR'] = 2020
    df = add_date_features(df)
    df_encoded = select_and_encode_features(df)

    df_encoded['Flight_Status'] = df['Flight_Status'].values  # Add label back
    df_encoded.to_csv(output_file_path, index=False)
    print(f"Preprocessing complete. Output saved to: {output_file_path}")
