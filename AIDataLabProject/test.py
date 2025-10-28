import numpy as np 
import pandas as pd
import geopandas as gpd
import folium
atlbusiness_data = pd.read_excel("data/Atlanta_Business_License_Records_2025.xlsx")
bus_ridership_data = pd.read_excel("data/MARTA_Bus_Ridership_2023_20250912.xlsx")
train_ridership_data = pd.read_excel("data/MARTA_Train_Ridership_2023_20250917.xlsx")

stops_gdf = gpd.read_file("data/MARTA_Stops_Shapefiles/MARTA_Stops.shp")
#crs = 4326
citylimit_gdf = gpd.read_file("data/ATL_CityLimit_Shapefiles/Official_Atlanta_City_Limits_-_Open_Data.shp")
#original crs is 6447
citylimit_gdf = citylimit_gdf.to_crs(epsg = 4326)

basemap = folium.Map(location = [33.753746, -84.386330], zoom_start=12)

#adding city limits to basemap

