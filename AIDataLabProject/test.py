import numpy as np 
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point

#specific route/stop data from MARTA website
stops = pd.read_csv("data/MARTA_GTFS/stops.txt")
routes = pd.read_csv("data/MARTA_GTFS/routes.txt")
trips = pd.read_csv("data/MARTA_GTFS/trips.txt")
stop_times = pd.read_csv("data/MARTA_GTFS/stop_times.txt")


atlbusiness_data = pd.read_excel("data/Atlanta_Business_License_Records_2025.xlsx")
bus_ridership_data = pd.read_excel("data/MARTA_Bus_Ridership_2023_20250912.xlsx")
train_ridership_data = pd.read_excel("data/MARTA_Train_Ridership_2023_20250917.xlsx")

# make atl business_data into a geopandas file
atlbusiness_data["geometry"] = gpd.points_from_xy(atlbusiness_data["longitude"], atlbusiness_data["latitude"])
business_df_geo = gpd.GeoDataFrame(atlbusiness_data, crs='epsg:4326', geometry="geometry")
stops_gdf = gpd.read_file("data/MARTA_Stops_Shapefiles/MARTA_Stops.shp")

citylimit_gdf = gpd.read_file("data/ATL_CityLimit_Shapefiles/Official_Atlanta_City_Limits_-_Open_Data.shp")
# original crs is 6447
citylimit_gdf = citylimit_gdf.to_crs(epsg=4326)
for col in citylimit_gdf.columns:
    if pd.api.types.is_datetime64_any_dtype(citylimit_gdf[col]):
        citylimit_gdf[col] = citylimit_gdf[col].astype(str)

basemap = folium.Map(location=[33.753746, -84.386330], zoom_start=12)
#adding city limits to basemap
folium.GeoJson(
    citylimit_gdf,
    name='Atlanta City Limits'
).add_to(basemap)


#convert everything to EPSG:26916 for accurate distance measurements
business_df_geo = business_df_geo.to_crs(epsg=26916)
stops_gdf = stops_gdf.to_crs(epsg=26916)
citylimit_gdf_utm = citylimit_gdf.to_crs(epsg=26916)


#spatial join to get only businesses and stops within Atlanta
city_union = citylimit_gdf_utm.union_all()
business_in_atlanta = business_df_geo[business_df_geo.within(city_union)]
stops_in_atlanta = stops_gdf[stops_gdf.within(city_union)]



#do another spacial join to determine distances between respective businesses and their closest stop
nearest = gpd.sjoin_nearest(
    business_in_atlanta, 
    stops_in_atlanta[["geometry"]], 
    how="left", 
    distance_col="dist_to_stop_m"
)

#determine mean differences
mean_dist = (
    nearest.groupby("disinvested_neighborhood")["dist_to_stop_m"]
    .mean()
    .reset_index()
)

print(mean_dist)
print(business_in_atlanta.crs) 
print(stops_in_atlanta.crs)

#are types of businesses more prevalent in disinvested neighborhoods

#check peak ridership dates, what insentivised these , where are the buses nearest 
#create shape file that is a circle around different arenas
#check how much major sporting events/concerts affect peak ridership

#check top reviewed locations on google maps using web scraping,


#naics code analysis

top_types = (
    business_df_geo
    .groupby(["disinvested_neighborhood", "naics_name"])
    .size()
    .reset_index(name="count")
)

#calculate percentages for comparison
top_types["percent"] = (
    top_types.groupby("disinvested_neighborhood")["count"]
    .transform(lambda x: x / x.sum() * 100)
)

#get the top 10 industries for each neighborhood type
top_types.sort_values(["disinvested_neighborhood", "count"], ascending=[True, False]).groupby("disinvested_neighborhood").head(10)


#section for timeseries on stops near mercedes benz stadium
gtfs_stops_gdf = gpd.GeoDataFrame(
    stops,
    geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
    crs="EPSG:4326"
)

gtfs_stops_gdf = gtfs_stops_gdf.to_crs(epsg=26916)

stadium_point = gpd.GeoSeries(
    [Point(-84.4006, 33.7573)],
    crs="EPSG:4326"
)
stadium_point_calc = stadium_point.to_crs(epsg=26916)
stadium_buffer_calc = stadium_point_calc.buffer(402.336) #creates a 402 meter circle around the stadium (0.25 miles)
stadium_circle_gdf = gpd.GeoDataFrame(geometry=stadium_buffer_calc, crs="epsg:26916")

stadium_union = stadium_circle_gdf.union_all()
stops_within_radius = stops_gdf[stops_gdf.within(stadium_union)]

near_stop_ids = stops_within_radius['stop_code'].astype(str).unique()
stop_times['stop_id'] = stop_times['stop_id'].astype(str)
trips_for_stops = stop_times[stop_times.stop_id.isin(near_stop_ids)]

trip_ids = trips_for_stops['trip_id'].unique()
routes_for_trips = trips[trips.trip_id.isin(trip_ids)]
route_ids = routes_for_trips['route_id'].unique()

routes_near_stadium = routes[routes.route_id.isin(route_ids)]
print(routes_near_stadium[['route_id', 'route_short_name', 'route_long_name']])
bus_routes_only = routes_near_stadium[
    routes_near_stadium['route_short_name'].str.isnumeric()
]

near_route_numbers = bus_routes_only['route_short_name'].astype(int).unique()

#filter ridership data to just these bus routes
filtered_ridership = bus_ridership_data[
    bus_ridership_data['Route'].isin(near_route_numbers)
]

#filter again to only include 2024
ridership_2024 = filtered_ridership[
    (filtered_ridership['Date'] >= '2024-01-01') & 
    (filtered_ridership['Date'] <= '2024-12-31')
]
#time series plot
time_series_weekly = ridership_2024.groupby(pd.Grouper(key='Date', freq='W'))['Total trips'].sum().reset_index() # Aggregate weekly for smoother plot


plt.figure(figsize=(12,5))
sns.lineplot(data=time_series_weekly, x='Date', y='Total trips')
plt.title("Weekly MARTA Bus Ridership for Routes Serving Mercedes-Benz Stadium (2024)")
plt.xlabel("Week")
plt.ylabel("Total Trips")
plt.show()

plt.figure(figsize=(12,5))
sns.lineplot(data=ridership_2024, x='Date', y='Total trips')
plt.title("Daily MARTA Bus Ridership for Routes Serving Mercedes-Benz Stadium (2024)")
plt.xlabel("Day")
plt.ylabel("Total Trips")
plt.show()