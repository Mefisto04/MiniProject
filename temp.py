import random
import pandas as pd
from datetime import datetime, timedelta

# Helper functions
def generate_random_lat_long():
    # Coordinates within India roughly: Latitude 8.0 to 37.0, Longitude 68.0 to 97.0
    latitude = random.uniform(8.0, 37.0)
    longitude = random.uniform(68.0, 97.0)
    return longitude, latitude

def generate_random_date():
    # Random date in 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    return random_date

def get_season(date):
    # Assign season based on month
    month = date.month
    if 3 <= month <= 6: # March to June
        return "Zaid"
    elif 7 <= month <= 10: # July to October
        return "Kharif"
    else: # November to February
        return "Rabi"

def generate_soil_type():
    soil_types = ["Alluvial", "Black", "Red", "Laterite", "Arid", "Saline", "Peaty"]
    return random.choice(soil_types)

def generate_crop(season, soil_type):
    # Common crops based on season and soil suitability
    crop_options = {
        "Rabi": ["Wheat", "Barley", "Mustard", "Peas", "Lentils"],
        "Kharif": ["Rice", "Millet", "Cotton", "Sugarcane", "Maize"],
        "Zaid": ["Watermelon", "Cucumber", "Vegetables", "Fodder", "Spices"]
    }
    if soil_type in ["Saline", "Arid"]:
        # More salt-tolerant crops
        return "Spices"
    return random.choice(crop_options[season])

# Generate dataset
data = []
for _ in range(500):
    longitude, latitude = generate_random_lat_long()
    date = generate_random_date()
    season = get_season(date)
    farm_size_hectares = random.uniform(0.5, 20.0)
    nitrogen = random.uniform(20, 150)
    phosphorus = random.uniform(5, 60)
    potassium = random.uniform(5, 120)
    soil_type = generate_soil_type()
    cultivated_crop = generate_crop(season, soil_type)

    data.append([longitude, latitude, date.strftime("%Y-%m-%d"), season, farm_size_hectares,
                 nitrogen, phosphorus, potassium, soil_type, cultivated_crop])

# Create a DataFrame
df = pd.DataFrame(data, columns=[
    "Longitude", "Latitude", "Date", "Season", "Farm_Size_Hectares",
    "N", "P", "K", "Soil_Type", "Cultivated_Crop"
])

# Display the first few rows
df.head()
