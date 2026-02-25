# backend/ml/rainfall.py
import pandas as pd
from datetime import datetime

df = pd.read_csv('..\data\Sub_Division_IMD_2017.csv')  # keep your path

month_map = {1:'JAN', 2:'FEB', 3:'MAR', 4:'APR', 5:'MAY', 6:'JUN',
             7:'JUL', 8:'AUG', 9:'SEP', 10:'OCT', 11:'NOV', 12:'DEC'}

season_months = {
    'Winter': [12, 1, 2],
    'Pre-Monsoon': [3, 4, 5],
    'Monsoon': [6, 7, 8, 9],
    'Post-Monsoon': [10, 11]
}

def predict_rainfall(region: str):
    df_region = df[df['SUBDIVISION'] == region].copy()
    if df_region.empty:
        return {"error": f"No data found for region '{region}'"}

    current_month = datetime.now().month
    if current_month in [2,5,9,11]:
        current_month += 1 
    for season, months in season_months.items():
        if current_month in months:
            current_season = season
            break

    season_columns = [month_map[m] for m in season_months[current_season]]
    df_region['SeasonTotal'] = df_region[season_columns].sum(axis=1)
    average_rainfall = df_region['SeasonTotal'].mean()

    return {
        "region": region,
        "season": current_season,
        "average_rainfall_mm": round(average_rainfall, 2)
    }
