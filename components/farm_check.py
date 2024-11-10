# components/farm_check.py

import pandas as pd

# Load the crop recommendation data
df = pd.read_csv('crop_recommendation.csv')
num_col = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def get_crop_parameters(crop_name):
    crop_name = crop_name.lower()
    crop_info = df[df['label'].str.lower() == crop_name]
    
    if not crop_info.empty:
        parameters = crop_info.iloc[0][num_col].to_dict()
        return parameters
    else:
        return None

def get_suitable_locations(crop_name, humidity_range, rainfall_range):
    # Sample data for locations (can be replaced with an actual data source)
    locations_data = [
        {'name': 'Nagpur', 'humidity': 85, 'rainfall': 150},
        {'name': 'Pune', 'humidity': 60, 'rainfall': 100},
        {'name': 'Hyderabad', 'humidity': 90, 'rainfall': 120},
        {'name': 'Mumbai', 'humidity': 95, 'rainfall': 200},
        {'name': 'Delhi', 'humidity': 70, 'rainfall': 80},
        # ... more location data ...
    ]
    
    suitable_locations = [
        location['name']
        for location in locations_data
        if humidity_range[0] <= location['humidity'] <= humidity_range[1] and
           rainfall_range[0] <= location['rainfall'] <= rainfall_range[1]
    ]

    return suitable_locations
