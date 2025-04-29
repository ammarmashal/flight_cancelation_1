from src.data_cleaning import load_raw_data, map_airport_codes, clean_and_merge_data, enrich_with_geodata, save_cleaned_data
from src.prepare_data import (
    process_missing_values, 
    create_airport_clusters, calculate_airport_importance,
    add_airport_features, calculate_airline_stats,
    add_airline_features, add_date_features, 
    select_and_encode_features
)
from src.modeling import train_model, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def main():
    # Load and clean data
    flights, airlines, airports, a_codes, n_codes = load_raw_data()
    airport_code_mapping = map_airport_codes(airports, a_codes, n_codes)
    cleaned_flights = clean_and_merge_data(flights, airlines, airport_code_mapping)
    df = enrich_with_geodata(cleaned_flights, airports)
    save_cleaned_data(df)

    # Add dummy year if missing
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
    # Split data before encoding and label separation
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Flight_Status'], random_state=42)
    # Drop labels and encode features

    y_train = train_df['Flight_Status']
    X_train = train_df.drop(columns=['Flight_Status'])

    y_test = test_df['Flight_Status']
    X_test = test_df.drop(columns=['Flight_Status'])

    # Train and evaluate
    model = train_model(X_train, y_train, X_test, y_test)
    print("\nTrain Evaluation:")
    evaluate_model(model, X_train, y_train)
    print("\nTest Evaluation:")
    evaluate_model(model, X_test, y_test)

    # Save model
    model.save_model('flight_cancellation_model.txt')

if __name__ == "__main__":
    main()
