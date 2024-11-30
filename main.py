import streamlit as st
from components.crop_prediction import crop_prediction
from components.data_visualization import data_visualization
from components.maps import crop_map, soil_map
from components.navigation import navigate_to
from components.farm_check import get_crop_parameters, get_suitable_locations

if 'page' not in st.session_state:
    st.session_state.page = 'analysis'

def main():
    if st.session_state.page == 'home':
        crop_prediction()
        
    elif st.session_state.page == 'analysis':
        data_visualization()
    elif st.session_state.page == 'crop':
        crop_map()
    elif st.session_state.page == 'soil':
        soil_map()

    elif st.session_state.page == 'farm_check':
        st.title("Farm Check")

        crop_name = st.text_input("Enter Crop Name:")
        if st.button("Check"):
            crop_params = get_crop_parameters(crop_name)

            if crop_params:
                st.write(f"Parameters for {crop_name}:")
                for param, value in crop_params.items():
                    st.write(f"{param}: {value}")

                humidity = crop_params.get("humidity")
                rainfall = crop_params.get("rainfall")

                if humidity is not None and rainfall is not None:
                    humidity_range = (humidity - 10, humidity + 10) 
                    rainfall_range = (rainfall - 50, rainfall + 50)  
                    
                    st.subheader(f"Humidity Range for {crop_name}: {humidity_range[0]} - {humidity_range[1]}")
                    st.subheader(f"Rainfall Range for {crop_name}: {rainfall_range[0]} - {rainfall_range[1]}")
                    
                    suitable_locations = get_suitable_locations(crop_name, humidity_range, rainfall_range)
                    if suitable_locations:
                        st.write(f"Suitable locations for {crop_name}:")
                        for location in suitable_locations:
                            st.write(f"- {location}")
                    else:
                        st.write(f"No suitable locations found for {crop_name}.")
                else:
                    st.write("Humidity or Rainfall data is not available for this crop.")
            else:
                st.write(f"Crop {crop_name} not found in the dataset.")

    if st.button("Go to Analysis"):
        navigate_to('analysis')
    if st.button("Go to Map"):
        navigate_to('crop')
    if st.button("Go to Farm Check"):
        navigate_to('farm_check')
    if st.button("Back to Home"):
        navigate_to('home')

if __name__ == "__main__":
    main()
