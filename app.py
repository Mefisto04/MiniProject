# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Load the datasets
# df = pd.read_csv('crop_recommendation.csv')
# schemes_df = pd.read_csv('govscheme.csv')
# farm_data = pd.read_csv('farm_data.csv')

# # Preprocess the data
# num_col = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
# target_col = 'label'
# X = df[num_col]
# y = df[target_col]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocessor = ColumnTransformer(
#     transformers=[('num', StandardScaler(), num_col)]
# )

# # Train the model
# model = RandomForestClassifier()
# pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
# pipe.fit(X_train, y_train)

# # Clean the schemes data
# schemes_df.replace('N/A', np.nan, inplace=True)
# schemes_df = schemes_df[
#     (schemes_df['Implementation_End_Year'].str.lower() == 'ongoing') &
#     (schemes_df['Farmer_Eligibility'].notna()) &
#     (schemes_df['Target_Crops'].notna())
# ]

# # Function to suggest schemes based on the predicted crop
# def suggest_schemes(predicted_crop, region):
#     predicted_crop = str(predicted_crop).lower()
#     crop_categories = {
#         'fruits': ['banana', 'apple', 'mango', 'grapes', 'orange'],
#         'cereals': ['rice', 'wheat', 'maize'],
#         'beans': ['lentil', 'chickpea', 'black gram', 'kidneybeans'],
#         'nuts': ['peanut', 'almond', 'cashew']
#     }

#     target_crops = schemes_df['Target_Crops'].fillna('').str.lower()

#     # Check for specific crop first
#     schemes = schemes_df[target_crops.str.contains(predicted_crop, case=False, na=False)]
#     if not schemes.empty:
#         return schemes['Scheme_Name'].tolist()

#     # Check for broader categories
#     for category, crops in crop_categories.items():
#         if predicted_crop in crops:
#             crop_pattern = '|'.join(crops)
#             schemes = schemes_df[target_crops.str.contains(crop_pattern, case=False, na=False)]
#             break

#     # Check for "Beans" if specific crop or broader categories did not return any schemes
#     if predicted_crop == 'kidneybeans':
#         beans_schemes = schemes_df[target_crops.str.contains('beans', case=False, na=False)]
#         if not beans_schemes.empty:
#             return beans_schemes['Scheme_Name'].tolist()

#     # If no schemes found, check for "All crops"
#     all_crops_schemes = schemes_df[target_crops.str.contains('all crops', case=False, na=False)]
#     if not all_crops_schemes.empty:
#         return all_crops_schemes['Scheme_Name'].tolist()

#     return []  # Return empty list if no schemes are found

# # Function for plotting graphs
# def plot_graph(x_param, y_param, graph_type):
#     plt.figure(figsize=(10, 6))

#     if graph_type == 'Scatter':
#         for label in df['label'].unique():
#             subset = df[df['label'] == label]
#             plt.scatter(subset[x_param], subset[y_param], label=label, alpha=0.6)
#     elif graph_type == 'Line':
#         for label in df['label'].unique():
#             subset = df[df['label'] == label]
#             plt.plot(subset[x_param], subset[y_param], label=label, marker='o', alpha=0.6)
#     elif graph_type == 'Bar':
#         for label in df['label'].unique():
#             subset = df[df['label'] == label]
#             plt.bar(subset[x_param], subset[y_param], label=label, alpha=0.6)

#     plt.title(f'{graph_type} plot between {x_param} and {y_param}')
#     plt.xlabel(x_param)
#     plt.ylabel(y_param)
#     plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()
#     st.pyplot(plt)

# # Navigation simulation using session state
# if 'page' not in st.session_state:
#     st.session_state.page = 'home'

# def navigate_to(page):
#     st.session_state.page = page

# # Main app
# if st.session_state.page == 'home':
#     st.title("Crop Prediction and Scheme Recommendation")

#     st.header("Enter the following details:")

#     # Get user input
#     N = st.number_input("Nitrogen content (N)", min_value=0, max_value=100, value=50)
#     P = st.number_input("Phosphorus content (P)", min_value=0, max_value=100, value=50)
#     K = st.number_input("Potassium content (K)", min_value=0, max_value=100, value=50)
#     temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
#     humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
#     ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=7.0)
#     rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

#     # Make prediction when the user clicks the button
#     if st.button("Predict Crop"):
#         input_data = pd.DataFrame({
#             'N': [N], 'P': [P], 'K': [K],
#             'temperature': [temperature], 'humidity': [humidity],
#             'ph': [ph], 'rainfall': [rainfall]
#         })

#         # Predict the crop
#         predicted_crop = pipe.predict(input_data)[0]
#         st.write(f"Predicted Crop: **{predicted_crop}**")

#         # Suggest schemes based on the predicted crop
#         schemes = suggest_schemes(predicted_crop, 'All India')
#         if schemes:
#             st.write("Suggested Schemes:")
#             for scheme in schemes:
#                 st.write(f"- {scheme}")
#         else:
#             st.write("No schemes found for the predicted crop.")

#     # Button to navigate to Analysis
#     if st.button("Go to Analysis"):
#         navigate_to('analysis')

# elif st.session_state.page == 'analysis':
#     st.title("Crop Data Analysis")

#     # Dropdowns for graph parameters
#     parameters = list(df.columns[:-1])
#     graph_types = ['Scatter', 'Line', 'Bar']
#     x_param = st.selectbox('X Parameter', parameters)
#     y_param = st.selectbox('Y Parameter', parameters)
#     graph_type = st.selectbox('Graph Type', graph_types)

#     # Plot the graph when the user clicks the button
#     if st.button("Plot Graph"):
#         plot_graph(x_param, y_param, graph_type)

#     # Button to go back to Home
#     if st.button("Back to Home"):
#         navigate_to('home')




# import streamlit as st
# import pandas as pd
# import numpy as np
# import folium
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# from folium.plugins import MarkerCluster
# from streamlit_folium import folium_static


# df = pd.read_csv('crop_recommendation.csv')
# schemes_df = pd.read_csv('govschemeupdated.csv')
# farm_data = pd.read_csv('crop_farm.csv')
# crop_data = pd.read_csv('crop_farm.csv')  

# num_col = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
# target_col = 'label'
# X = df[num_col]
# y = df[target_col]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocessor = ColumnTransformer(
#     transformers=[('num', StandardScaler(), num_col)]
# )

# model = RandomForestClassifier()
# pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
# pipe.fit(X_train, y_train)

# schemes_df.replace('N/A', np.nan, inplace=True)
# schemes_df = schemes_df[
#     (schemes_df['Implementation_End_Year'].str.lower() == 'ongoing') &
#     (schemes_df['Farmer_Eligibility'].notna()) &
#     (schemes_df['Target_Crops'].notna())
# ]

# def suggest_schemes(predicted_crop, region):
#     predicted_crop = str(predicted_crop).lower()
#     crop_categories = {
#         'fruits': ['banana', 'apple', 'mango', 'grapes', 'orange'],
#         'cereals': ['rice', 'wheat', 'maize'],
#         'beans': ['lentil', 'chickpea', 'black gram', 'kidneybeans'],
#         'nuts': ['peanut', 'almond', 'cashew']
#     }

#     target_crops = schemes_df['Target_Crops'].fillna('').str.lower()

#     schemes = schemes_df[target_crops.str.contains(predicted_crop, case=False, na=False)]
#     if not schemes.empty:
#         return schemes['Scheme_Name'].tolist()

#     for category, crops in crop_categories.items():
#         if predicted_crop in crops:
#             crop_pattern = '|'.join(crops)
#             schemes = schemes_df[target_crops.str.contains(crop_pattern, case=False, na=False)]
#             break

#     if predicted_crop == 'kidneybeans':
#         beans_schemes = schemes_df[target_crops.str.contains('beans', case=False, na=False)]
#         if not beans_schemes.empty:
#             return beans_schemes['Scheme_Name'].tolist()

#     all_crops_schemes = schemes_df[target_crops.str.contains('all crops', case=False, na=False)]
#     if not all_crops_schemes.empty:
#         return all_crops_schemes['Scheme_Name'].tolist()

#     return []  

# def plot_graph(x_param, y_param, graph_type):
#     plt.figure(figsize=(10, 6))

#     if graph_type == 'Scatter':
#         for label in df['label'].unique():
#             subset = df[df['label'] == label]
#             plt.scatter(subset[x_param], subset[y_param], label=label, alpha=0.6)
#     elif graph_type == 'Line':
#         for label in df['label'].unique():
#             subset = df[df['label'] == label]
#             plt.plot(subset[x_param], subset[y_param], label=label, marker='o', alpha=0.6)
#     elif graph_type == 'Bar':
#         for label in df['label'].unique():
#             subset = df[df['label'] == label]
#             plt.bar(subset[x_param], subset[y_param], label=label, alpha=0.6)

#     plt.title(f'{graph_type} plot between {x_param} and {y_param}')
#     plt.xlabel(x_param)
#     plt.ylabel(y_param)
#     plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()
#     st.pyplot(plt)

# if 'page' not in st.session_state:
#     st.session_state.page = 'home'

# def navigate_to(page):
#     st.session_state.page = page

# def create_crop_map():
#     crop_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  
#     marker_cluster = MarkerCluster().add_to(crop_map)

#     for _, row in crop_data.iterrows():
#         folium.Marker(
#             location=[row['Latitude'], row['Longitude']],
#             popup=f"Crop: {row['Cultivated_Crop']}, N: {row['N']}, P: {row['P']}, K: {row['K']}, Soil Type: {row['Soil_Type']}",
#             icon=folium.Icon(color='green')
#         ).add_to(marker_cluster)

#     return crop_map

# def create_soil_map():
#     soil_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Centered on India
#     marker_cluster = MarkerCluster().add_to(soil_map)

#     for _, row in crop_data.iterrows():
#         folium.Marker(
#             location=[row['Latitude'], row['Longitude']],
#             popup=f"N: {row['N']}, P: {row['P']}, K: {row['K']}, Soil Type: {row['Soil_Type']}",
#             icon=folium.Icon(color='blue')
#         ).add_to(marker_cluster)

#     return soil_map

# # Main app
# if st.session_state.page == 'home':
#     st.title("Crop Prediction and Scheme Recommendation")

#     st.header("Enter the following details:")

#     # Get user input
#     N = st.number_input("Nitrogen content (N)", min_value=0, max_value=100, value=68)
#     P = st.number_input("Phosphorus content (P)", min_value=0, max_value=100, value=58)
#     K = st.number_input("Potassium content (K)", min_value=0, max_value=100, value=38)
#     temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=23.22)
#     humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=83.03)
#     ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=6.3)
#     rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=221.20)

#     # Make prediction when the user clicks the button
#     if st.button("Predict Crop"):
#         input_data = pd.DataFrame({
#             'N': [N], 'P': [P], 'K': [K],
#             'temperature': [temperature], 'humidity': [humidity],
#             'ph': [ph], 'rainfall': [rainfall]
#         })

#         # Predict the crop
#         predicted_crop = pipe.predict(input_data)[0]
#         st.write(f"Predicted Crop: **{predicted_crop}**")

#         # Suggest schemes based on the predicted crop
#         schemes = suggest_schemes(predicted_crop, 'All India')
#         if schemes:
#             st.write("Suggested Schemes:")
#             for scheme in schemes:
#                 st.write(f"- {scheme}")
#         else:
#             st.write("No schemes found for the predicted crop.")

#     # Button to navigate to Analysis
#     if st.button("Go to Analysis"):
#         navigate_to('analysis')

#     # Button to navigate to Crop
#     if st.button("Go to Map"):
#         navigate_to('crop')

# elif st.session_state.page == 'analysis':
#     st.title("Crop Data Analysis")

#     # Dropdowns for graph parameters
#     parameters = list(df.columns[:-1])
#     graph_types = ['Scatter', 'Line', 'Bar']
#     x_param = st.selectbox('X Parameter', parameters)
#     y_param = st.selectbox('Y Parameter', parameters)
#     graph_type = st.selectbox('Graph Type', graph_types)

#     # Plot the graph when the user clicks the button
#     if st.button("Plot Graph"):
#         plot_graph(x_param, y_param, graph_type)

#     # Button to go back to Home
#     if st.button("Go back to Home"):
#         navigate_to('home')

# elif st.session_state.page == 'crop':
#     st.title("Crop and Soil Distribution Maps")

#     # Create maps
#     st.header("Crop Distribution Map")
#     crop_map = create_crop_map()
#     folium_static(crop_map)

#     st.header("Soil Data Map")
#     soil_map = create_soil_map()
#     folium_static(soil_map)

#     # Button to go back to Home
#     if st.button("Go back to Home"):
#         navigate_to('home')



import streamlit as st
import pandas as pd
import numpy as np
import folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static


# Load data
df = pd.read_csv('crop_recommendation.csv')
schemes_df = pd.read_csv('govschemeupdated.csv')
farm_data = pd.read_csv('crop_farm.csv')
crop_data = pd.read_csv('crop_farm.csv')  
msp = pd.read_csv('msp_2024.csv')

# Prepare data for model
num_col = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target_col = 'label'
X = df[num_col]
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_col)]
)

model = RandomForestClassifier()
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
pipe.fit(X_train, y_train)

# Filter schemes for ongoing projects
schemes_df.replace('N/A', np.nan, inplace=True)
schemes_df = schemes_df[
    (schemes_df['Implementation_End_Year'].str.lower() == 'ongoing') &
    (schemes_df['Farmer_Eligibility'].notna()) &
    (schemes_df['Target_Crops'].notna())
]

def suggest_schemes(predicted_crop, region):
    predicted_crop = str(predicted_crop).lower()
    crop_categories = {
        'fruits': ['banana', 'apple', 'mango', 'grapes', 'orange'],
        'cereals': ['rice', 'wheat', 'maize'],
        'beans': ['lentil', 'chickpea', 'black gram', 'kidneybeans'],
        'nuts': ['peanut', 'almond', 'cashew']
    }

    target_crops = schemes_df['Target_Crops'].fillna('').str.lower()
    schemes = schemes_df[target_crops.str.contains(predicted_crop, case=False, na=False)]
    
    if not schemes.empty:
        return schemes['Scheme_Name'].tolist()

    for category, crops in crop_categories.items():
        if predicted_crop in crops:
            crop_pattern = '|'.join(crops)
            schemes = schemes_df[target_crops.str.contains(crop_pattern, case=False, na=False)]
            break

    if predicted_crop == 'kidneybeans':
        beans_schemes = schemes_df[target_crops.str.contains('beans', case=False, na=False)]
        if not beans_schemes.empty:
            return beans_schemes['Scheme_Name'].tolist()

    all_crops_schemes = schemes_df[target_crops.str.contains('all crops', case=False, na=False)]
    if not all_crops_schemes.empty:
        return all_crops_schemes['Scheme_Name'].tolist()

    return []  

def plot_graph(x_param, y_param, graph_type):
    plt.figure(figsize=(10, 6))

    if graph_type == 'Scatter':
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.scatter(subset[x_param], subset[y_param], label=label, alpha=0.6)
    elif graph_type == 'Line':
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.plot(subset[x_param], subset[y_param], label=label, marker='o', alpha=0.6)
    elif graph_type == 'Bar':
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.bar(subset[x_param], subset[y_param], label=label, alpha=0.6)

    plt.title(f'{graph_type} plot between {x_param} and {y_param}')
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

if 'page' not in st.session_state:
    st.session_state.page = 'analysis'

def navigate_to(page):
    st.session_state.page = page

def create_crop_map():
    crop_map = folium.Map(location=[20.5937, 78.9629], zoom_start=8)  
    marker_cluster = MarkerCluster().add_to(crop_map)

    for _, row in crop_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Crop: {row['Cultivated_Crop']}, N: {row['N']}, P: {row['P']}, K: {row['K']}, Soil Type: {row['Soil_Type']}",
            icon=folium.Icon(color='green')
        ).add_to(marker_cluster)

    return crop_map

def create_soil_map():
    soil_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Centered on India
    marker_cluster = MarkerCluster().add_to(soil_map)

    for _, row in crop_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"N: {row['N']}, P: {row['P']}, K: {row['K']}, Soil Type: {row['Soil_Type']}",
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)

    return soil_map

def get_crop_parameters(crop_name):
    crop_name = crop_name.lower()
    crop_info = df[df['label'].str.lower() == crop_name]
    
    if not crop_info.empty:
        parameters = crop_info.iloc[0][num_col].to_dict()
        return parameters
    else:
        return None

def get_suitable_locations(crop_name, humidity_range, rainfall_range):
    # Sample data for locations (you can replace this with your actual data source)
    locations_data = [
    {'name': 'Nagpur', 'humidity': 85, 'rainfall': 150},
    {'name': 'Pune', 'humidity': 60, 'rainfall': 100},
    {'name': 'Hyderabad', 'humidity': 90, 'rainfall': 120},
    {'name': 'Mumbai', 'humidity': 95, 'rainfall': 200},
    {'name': 'Delhi', 'humidity': 70, 'rainfall': 80},
    {'name': 'Bangalore', 'humidity': 65, 'rainfall': 130},
    {'name': 'Chennai', 'humidity': 80, 'rainfall': 160},
    {'name': 'Kolkata', 'humidity': 78, 'rainfall': 170},
    {'name': 'Ahmedabad', 'humidity': 55, 'rainfall': 90},
    {'name': 'Jaipur', 'humidity': 45, 'rainfall': 70},
    {'name': 'Lucknow', 'humidity': 65, 'rainfall': 110},
    {'name': 'Bhopal', 'humidity': 58, 'rainfall': 95},
    {'name': 'Indore', 'humidity': 52, 'rainfall': 85},
    {'name': 'Surat', 'humidity': 72, 'rainfall': 120},
    {'name': 'Visakhapatnam', 'humidity': 75, 'rainfall': 140},
    {'name': 'Patna', 'humidity': 68, 'rainfall': 105},
    {'name': 'Vadodara', 'humidity': 62, 'rainfall': 95},
    {'name': 'Guwahati', 'humidity': 78, 'rainfall': 180},
    {'name': 'Coimbatore', 'humidity': 70, 'rainfall': 95},
    {'name': 'Kochi', 'humidity': 85, 'rainfall': 250},
    {'name': 'Thiruvananthapuram', 'humidity': 80, 'rainfall': 180},
    {'name': 'Bhubaneswar', 'humidity': 75, 'rainfall': 150},
    {'name': 'Raipur', 'humidity': 62, 'rainfall': 130},
    {'name': 'Chandigarh', 'humidity': 55, 'rainfall': 110},
    {'name': 'Ranchi', 'humidity': 65, 'rainfall': 140},
    {'name': 'Agra', 'humidity': 58, 'rainfall': 85},
    {'name': 'Varanasi', 'humidity': 70, 'rainfall': 100},
    {'name': 'Amritsar', 'humidity': 60, 'rainfall': 70},
    {'name': 'Jodhpur', 'humidity': 40, 'rainfall': 35},
    {'name': 'Dehradun', 'humidity': 72, 'rainfall': 200},
]
    
    suitable_locations = []

    for location in locations_data:
        # Check if the location's humidity and rainfall fall within the specified ranges
        if (humidity_range[0] <= location['humidity'] <= humidity_range[1] and
            rainfall_range[0] <= location['rainfall'] <= rainfall_range[1]):
            suitable_locations.append(location['name'])

    return suitable_locations


# Main app
if st.session_state.page == 'home':
    st.title("Crop Prediction and Scheme Recommendation")

    st.header("Enter the following details:")

    # Get user input for crop prediction
    N = st.number_input("Nitrogen content (N)", min_value=0, max_value=100, value=68)
    P = st.number_input("Phosphorus content (P)", min_value=0, max_value=100, value=58)
    K = st.number_input("Potassium content (K)", min_value=0, max_value=100, value=38)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=23.22)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=83.03)
    ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=6.3)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=221.20)

    if st.button("Predict Crop"):
        input_data = pd.DataFrame({
            'N': [N], 'P': [P], 'K': [K],
            'temperature': [temperature], 'humidity': [humidity],
            'ph': [ph], 'rainfall': [rainfall]
        })

        # Predict the top 3 crops
        predicted_probabilities = pipe.predict_proba(input_data)[0]
        top_3_indices = predicted_probabilities.argsort()[-3:][::-1]
        top_3_crops = [pipe.classes_[index] for index in top_3_indices]

        st.write("Predicted Crops:")

        # Display each crop in a card format
        for i, crop in enumerate(top_3_crops, start=1):
            # Get suggested schemes for the crop
            schemes = suggest_schemes(crop, 'All India')
            
            # Format schemes as bullet points
            scheme_list = ""
            if schemes:
                scheme_list = "<ul>" + "".join(f"<li>{scheme}</li>" for scheme in schemes) + "</ul>"
            else:
                scheme_list = "<p>No schemes found.</p>"

            # Create the card using markdown
            card_html = f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 10px;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
            ">
                <h3 style="color: #5e0797;">{i}. {crop}</h3>
                <strong>Suggested Schemes:</strong>
                {scheme_list}
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)


                
        # if not found_scheme:
        #     st.write("No schemes found for the predicted crops.")


    # Button to navigate to Analysis
    if st.button("Go to Analysis"):
        navigate_to('analysis')

    # Button to navigate to Crop
    if st.button("Go to Map"):
        navigate_to('crop')

    # Button to navigate to Farm Check
    if st.button("Go to Farm Check"):
        navigate_to('farm_check')

elif st.session_state.page == 'analysis':
    st.title("Crop Data Analysis")

    # Dropdowns for graph parameters
    parameters = list(df.columns[:-1])
    graph_types = ['Scatter', 'Line', 'Bar']
    x_param = st.selectbox('X Parameter', parameters)
    y_param = st.selectbox('Y Parameter', parameters)
    graph_type = st.selectbox('Graph Type', graph_types)

    # Plot graph based on selections
    plot_graph(x_param, y_param, graph_type)

    # Button to navigate back to Home
    if st.button("Back to Home"):
        navigate_to('home')

    # Button to navigate to Crop
    if st.button("Go to Map"):
        navigate_to('crop')

    # Button to navigate to Farm Check
    if st.button("Go to Farm Check"):
        navigate_to('farm_check')

elif st.session_state.page == 'crop':
    st.title("Crop Distribution Map")

    crop_map = create_crop_map()
    folium_static(crop_map)

    # Button to navigate back to Home
    if st.button("Back to Home"):
        navigate_to('home')

    # Button to navigate to Analysis
    if st.button("Go to Analysis"):
        navigate_to('analysis')

    # Button to navigate to Farm Check
    if st.button("Go to Farm Check"):
        navigate_to('farm_check')

elif st.session_state.page == 'farm_check':
    st.title("Farm Check")

    crop_name = st.text_input("Enter Crop Name:")
    if st.button("Check"):
        crop_params = get_crop_parameters(crop_name)

        if crop_params:
            st.write(f"Parameters for {crop_name}:")
            for param, value in crop_params.items():
                st.write(f"{param}: {value}")

            # Extract humidity and rainfall from crop parameters
            humidity = crop_params.get("humidity")
            rainfall = crop_params.get("rainfall")

            # Define ranges for humidity and rainfall
            if humidity is not None and rainfall is not None:
                humidity_range = (humidity - 10, humidity + 10)  # Example range
                rainfall_range = (rainfall - 50, rainfall + 50)  # Example range
                
                st.write(f"Humidity Range for {crop_name}: {humidity_range[0]} - {humidity_range[1]}")
                st.write(f"Rainfall Range for {crop_name}: {rainfall_range[0]} - {rainfall_range[1]}")
                
                # Get suitable locations based on ranges
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

    # Button to navigate back to Home
    if st.button("Back to Home"):
        navigate_to('home')

    # Button to navigate to Analysis
    if st.button("Go to Analysis"):
        navigate_to('analysis')

    # Button to navigate to Crop
    if st.button("Go to Map"):
        navigate_to('crop')
