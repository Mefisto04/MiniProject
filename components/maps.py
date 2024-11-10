import folium
import streamlit as st
from folium.plugins import MarkerCluster
import pandas as pd
from streamlit_folium import folium_static

crop_data = pd.read_csv('crop_farm.csv')

def create_crop_map():
    crop_map = folium.Map(location=[20.5937, 78.9629], zoom_start=7)
    marker_cluster = MarkerCluster().add_to(crop_map)

    for _, row in crop_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Crop: {row['Cultivated_Crop']}, N: {row['N']}, P: {row['P']}, K: {row['K']}, Soil Type: {row['Soil_Type']}",
            icon=folium.Icon(color='green')
        ).add_to(marker_cluster)

    folium_static(crop_map)



def crop_map():
    st.title("Crop Distribution Map")
    create_crop_map()

def soil_map():
    st.title("Soil Distribution Map")
    # You can create a similar function for the soil map
    st.write("Soil map functionality can be implemented here.")
