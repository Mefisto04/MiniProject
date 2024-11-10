import pandas as pd
import numpy as np

schemes_df = pd.read_csv('govschemeupdated.csv')

schemes_df.replace('N/A', np.nan, inplace=True)
schemes_df = schemes_df[
    (schemes_df['Implementation_End_Year'].str.lower() == 'ongoing') &
    (schemes_df['Farmer_Eligibility'].notna()) &
    (schemes_df['Target_Crops'].notna())
]

def suggest_schemes(predicted_crop):
    schemes = schemes_df[schemes_df['Target_Crops'].str.contains(predicted_crop, case=False, na=False)]
    return schemes['Scheme_Name'].tolist()
