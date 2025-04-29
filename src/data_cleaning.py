import pandas as pd

def load_raw_data():
    """
    Load all raw data files from the data directory.
    """
    flights = pd.read_csv("data/flights.csv")
    airlines = pd.read_csv("data/airlines.csv")
    airports = pd.read_csv("data/airports.csv")
    a_codes = pd.read_excel("data/A_Codes.xlsx")
    n_codes = pd.read_excel("data/N_Codes.xlsx")
    return flights, airlines, airports, a_codes, n_codes


def map_airport_codes(airports, a_codes, n_codes):
    """
    Combine all airport code sources and unify airport names.
    """
    a_codes.columns = ['IATA_CODE', 'AIRPORT']
    n_codes.columns = ['IATA_CODE', 'AIRPORT']

    all_airports = pd.concat([airports[['IATA_CODE', 'AIRPORT']], a_codes, n_codes], ignore_index=True)
    all_airports['AIRPORT_GROUP_NAME'] = all_airports['AIRPORT'].str.strip().str.lower()

    airport_code_mapping = all_airports[['IATA_CODE', 'AIRPORT_GROUP_NAME']].drop_duplicates()
    return airport_code_mapping


def clean_and_merge_data(flights, airlines, airport_code_mapping):
    """
    Clean the flights dataset and merge with airlines and unified airport names.
    """

    # Columns we want to keep from flights
    columns_to_keep = [
        'MONTH', 'DAY', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
        'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_TIME',
        'Flight_Status', 'CANCELLATION_REASON',
        'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DISTANCE'
    ]
    flights = flights[columns_to_keep]

    # Merge with airline names
    flights = flights.merge(airlines, how='left', on='AIRLINE')

    # Create mapping dictionary from IATA_CODE to normalized name
    airport_dict = airport_code_mapping.set_index('IATA_CODE')['AIRPORT_GROUP_NAME'].to_dict()

    # 🟢 Keep original codes for geodata merging
    flights['ORIGIN_AIRPORT_CODE'] = flights['ORIGIN_AIRPORT']
    flights['DESTINATION_AIRPORT_CODE'] = flights['DESTINATION_AIRPORT']

    # Replace airport codes with normalized names
    flights['ORIGIN_AIRPORT_NORMALIZED'] = flights['ORIGIN_AIRPORT'].map(airport_dict)
    flights['DESTINATION_AIRPORT_NORMALIZED'] = flights['DESTINATION_AIRPORT'].map(airport_dict)

    return flights



def enrich_with_geodata(flights, airports):
    """
    Merge geographic and city/state information for origin and destination airports.
    """
    geo_info = airports[['IATA_CODE', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']]

    # Merge for origin airport using original IATA code
    flights = flights.merge(
        geo_info.rename(columns={
            'CITY': 'ORIGIN_CITY',
            'STATE': 'ORIGIN_STATE',
            'LATITUDE': 'ORIGIN_LAT',
            'LONGITUDE': 'ORIGIN_LON'
        }),
        how='left',
        left_on='ORIGIN_AIRPORT_CODE',
        right_on='IATA_CODE'
    )

    # Merge for destination airport
    flights = flights.merge(
        geo_info.rename(columns={
            'CITY': 'DESTINATION_CITY',
            'STATE': 'DESTINATION_STATE',
            'LATITUDE': 'DEST_LAT',
            'LONGITUDE': 'DEST_LON'
        }),
        how='left',
        left_on='DESTINATION_AIRPORT_CODE',
        right_on='IATA_CODE'
    )


    return flights



def save_cleaned_data(df, path="output/flights_cleaned.csv"):
    """
    Save the cleaned flights dataset to a CSV file.
    """
    df.to_csv(path, index=False)
    print(f"✅ Cleaned data saved to {path}")
