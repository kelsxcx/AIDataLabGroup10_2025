import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from folium import FeatureGroup
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from shapely.geometry import LineString
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.spatial import KDTree
import numpy as np

import matplotlib.colors as mcolors
import osmnx as ox  #pip install osmnx
import networkx as nx


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
        
stadium_point_gpd = gpd.GeoSeries([Point(-84.4006, 33.7573)], crs="EPSG:4326")
stadium_point = stadium_point_gpd.iloc[0]  # This gets the Point object
stadium_lat = stadium_point.y
stadium_lon = stadium_point.x

basemap = folium.Map(
    location=[stadium_lat, stadium_lon],
    zoom_start=13,
    tiles="CartoDB Positron"  # <— fixes the grey/blue missing basemap issue
)
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
city_union = citylimit_gdf_utm.geometry.union_all()
business_in_atlanta = gpd.sjoin(
    business_df_geo,
    citylimit_gdf_utm[['geometry']],
    how='inner',
    predicate='within'
)
#drop index_right column created by sjoin
business_in_atlanta.drop(columns=['index_right'], inplace=True)

stops_in_atlanta = gpd.sjoin(
    stops_gdf,
    citylimit_gdf_utm[['geometry']],
    how='inner',
    predicate='within'
)
# drop index_right column
stops_in_atlanta.drop(columns=['index_right'], inplace=True)

#do another spacial join to determine distances between respective businesses and their closest stop
nearest = gpd.sjoin_nearest(
    business_in_atlanta, 
    stops_in_atlanta[["geometry"]], 
    how="left", 
    distance_col="dist_to_stop_m",
    max_distance=5000,  # optional, speeds up nearest search
    rsuffix='stop'
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

#get stops near Mercedes-Benz Stadium

stadium_point_utm = stadium_point_gpd.to_crs(epsg=26916)
stadium_buffer = stadium_point_utm.buffer(402.336)  #402 meters (~0.25 miles)
stadium_circle_gdf = gpd.GeoDataFrame(geometry=stadium_buffer, crs="EPSG:26916")
stadium_union = stadium_circle_gdf.geometry.union_all()

stops_within_radius = stops_gdf[stops_gdf.within(stadium_union)]
near_stop_ids = stops_within_radius['stop_code'].astype(str).unique()

stop_times['stop_id'] = stop_times['stop_id'].astype(str)
trips_for_stops = stop_times[stop_times['stop_id'].isin(near_stop_ids)]
trip_ids = trips_for_stops['trip_id'].unique()

routes_for_trips = trips[trips['trip_id'].isin(trip_ids)]
route_ids = routes_for_trips['route_id'].unique()

routes_near_stadium = routes[routes['route_id'].isin(route_ids)]
bus_routes_only = routes_near_stadium[routes_near_stadium['route_short_name'].str.isnumeric()]
near_route_numbers = bus_routes_only['route_short_name'].astype(int).unique()

#filter bus ridership
filtered_ridership = bus_ridership_data[bus_ridership_data['Route'].isin(near_route_numbers)]
filtered_ridership['Date'] = pd.to_datetime(filtered_ridership['Date'])

ridership_2023 = filtered_ridership[
    (filtered_ridership['Date'] >= '2023-01-01') &
    (filtered_ridership['Date'] <= '2023-12-31')
]

#time series plot
time_series_weekly = (
    ridership_2023
    .groupby(pd.Grouper(key='Date', freq='W'))['Total trips']
    .sum()
    .reset_index()
)

peak_days = ridership_2023[ridership_2023['Total trips'] > 1500]
print(peak_days[['Date', 'Total trips']].sort_values('Total trips', ascending=False))


plt.figure(figsize=(12,5))
sns.lineplot(data=time_series_weekly, x='Date', y='Total trips')
plt.title("Weekly MARTA Bus Ridership for Routes Serving Mercedes-Benz Stadium (2023)")
plt.xlabel("Week")
plt.ylabel("Total Trips")
plt.show()

plt.figure(figsize=(12,5))
sns.lineplot(data=ridership_2023, x='Date', y='Total trips')
plt.title("Daily MARTA Bus Ridership for Routes Serving Mercedes-Benz Stadium (2023)")
plt.xlabel("Day")
plt.ylabel("Total Trips")
plt.show()

#joining mobility score data to business data
mobility_gdf = gpd.read_file(r"data\ONE_ATL\ONE ATL Mobility Scores by NSA.shp")
mobility_gdf = mobility_gdf.to_crs(epsg=26916)
#drop index_right if it exists
if 'index_right' in mobility_gdf.columns:
    mobility_gdf.drop(columns='index_right', inplace=True)

business_with_mobility = gpd.sjoin(
    nearest, 
    mobility_gdf[["Economic M", "geometry"]],
    how="left",
    predicate="within"
)

#clean mobility score data
business_with_mobility["Economic M"] = pd.to_numeric(business_with_mobility["Economic M"], errors='coerce')
business_with_mobility = business_with_mobility.dropna(subset=["Economic M"])

#binning mobility scores
bins = [0, 35, 45.4, 55, 100]
labels = ["Very Low", "Low", "Disinvested cutoff", "High"]
business_with_mobility["mobility_bin"] = pd.cut(business_with_mobility["Economic M"], bins=bins, labels=labels)


plt.figure(figsize=(10,6))
sns.boxplot(
    data=business_with_mobility,
    x="mobility_bin",
    y="dist_to_stop_m",
    palette="Set2"
)
plt.xlabel("Mobility Score Bin")
plt.ylabel("Distance to Nearest MARTA Stop (m)")
plt.title("Distance to Transit by Mobility Score Bin")
plt.show()

#transit access is not the issue here

#had to manually input each date and event

events_2023 = [
    {"Date": "2023-04-28", "Event": "Taylor Swift", "Event Type": "Concert"},
    {"Date": "2023-04-29", "Event": "Taylor Swift", "Event Type": "Concert"},
    {"Date": "2023-04-30", "Event": "Taylor Swift", "Event Type": "Concert"},
    {"Date": "2023-08-05", "Event": "Falcon's Game", "Event Type": "Sports Event"},
    {"Date": "2023-08-11", "Event": "Beyoncé", "Event Type": "Concert"},
    {"Date": "2023-08-12", "Event": "Beyoncé", "Event Type": "Concert"},
    {"Date": "2023-08-14", "Event": "Beyoncé", "Event Type": "Concert"},
    {"Date": "2023-09-10", "Event": "Falcon's Game", "Event Type": "Sports Event"},
    {"Date": "2023-09-21", "Event": "Karol G", "Event Type": "Concert"},
    {"Date": "2023-10-20", "Event": "ATLive", "Event Type": "Concert"},
    {"Date": "2023-10-21", "Event": "ATLive", "Event Type": "Concert"},
    {"Date": "2023-11-26", "Event": "Falcon's Game", "Event Type": "Sports Event"},
    {"Date": "2023-12-15", "Event": "HBCU Band of the Year", "Event Type": "Concert"},
]

events_df = pd.DataFrame(events_2023)

events_df['Date'] = pd.to_datetime(events_df['Date'])

print(events_df)
ridership_2023['Date'] = pd.to_datetime(ridership_2023['Date'])
merged = pd.merge(ridership_2023, events_df, on='Date', how='left')

train_stops_near_stadium = (
    stops_within_radius['stop_name']
    .str.upper()                          # make uppercase
    .str.replace(r"\s*STATION$", "", regex=True)  # remove "STATION" at end
    .str.replace(r"\(.*\)", "", regex=True)      # remove parentheses content
    .str.strip()                           # remove extra whitespace
)

# Normalize the station names in train ridership
train_ridership_data['station_normalized'] = (
    train_ridership_data['Station']
    .str.upper()
    .str.replace(r"\s*STATION$", "", regex=True)
    .str.strip()
)

# Filter for stations near stadium
train_ridership_near_stadium = train_ridership_data[
    train_ridership_data['station_normalized'].isin(train_stops_near_stadium)
].copy()

train_ridership_2023 = train_ridership_near_stadium[
    (train_ridership_near_stadium['Date'] >= '2023-01-01') &
    (train_ridership_near_stadium['Date'] <= '2023-12-31')
].copy()

daily_train_ridership_2023 = (
    train_ridership_2023
    .groupby('Date')['Total trips']
    .sum()
    .reset_index()
)

plt.figure(figsize=(12,5))
sns.lineplot(data=daily_train_ridership_2023, x='Date', y='Total trips', marker='o')
for _, row in events_df.iterrows():
    plt.axvline(row['Date'], color='red', linestyle='--', alpha=0.5)
plt.title("Daily MARTA Train Ridership Near Mercedes-Benz Stadium (2023)")
plt.xlabel("Date")
plt.ylabel("Total Trips")
plt.show()



ridership_compare = pd.merge(
    daily_train_ridership_2023,
    ridership_2023,
    on="Date",
    how="outer"
).sort_values("Date")

plt.figure(figsize=(14,6))
sns.lineplot(data=ridership_compare, x='Date', y='Total trips_x', label="Train Ridership", marker='o')
sns.lineplot(data=ridership_compare, x='Date', y='Total trips_y', label="Bus Ridership", marker='o')

# Add event vertical lines
for _, row in events_df.iterrows():
    plt.axvline(row['Date'], color='red', linestyle='--', alpha=0.3)

plt.title("Train vs Bus Ridership Near Mercedes-Benz Stadium (2023)")
plt.xlabel("Date")
plt.ylabel("Daily Ridership")
plt.legend()
plt.tight_layout()
plt.show()


#computing average distance between train and disenvested businesses
train_stops = stops_in_atlanta[
    stops_in_atlanta["stop_name"].str.endswith("STATION")
].copy()
train_stops = train_stops.to_crs(epsg=26916)
nearest_train = gpd.sjoin_nearest(
    business_in_atlanta,
    train_stops[["geometry"]],
    how="left",
    distance_col="dist_to_train_m"
)
avg_train_distance = nearest_train["dist_to_train_m"].mean()

train_dist_by_area = (
    nearest_train.groupby("disinvested_neighborhood")["dist_to_train_m"]
    .mean()
    .reset_index()
)

print(train_dist_by_area)

#violin plot visualization of average distance to nearest train stop
plt.figure(figsize=(10,6))
sns.violinplot(
    data=nearest_train,
    x="disinvested_neighborhood",
    y="dist_to_train_m",
    palette="Set2"
)
plt.xlabel("Disinvested Neighborhood")
plt.ylabel("Distance to Nearest Train Stop (m)")
plt.title("Distribution of Distance to Train Stops")
plt.show()


business_in_atlanta = business_in_atlanta.reset_index().rename(columns={'index':'biz_id'})
nearest = nearest.reset_index().rename(columns={'index':'biz_id'})
nearest_train = nearest_train.reset_index().rename(columns={'index':'biz_id'})
business_with_mobility = business_with_mobility.reset_index().rename(columns={'index':'biz_id'})

full_df = business_with_mobility.merge(
    nearest_train[['biz_id', 'dist_to_train_m']],
    on='biz_id',
    how='left'
)
full_df['dist_to_stadium_m'] = full_df.geometry.apply(lambda x: x.distance(stadium_point_utm.iloc[0]))

corr_df = full_df[[
    "Economic M",
    "dist_to_stop_m",
    "dist_to_train_m",
    "dist_to_stadium_m",
    "disinvested_neighborhood"
]].copy()

corr_df["disinvested_neighborhood"] = corr_df["disinvested_neighborhood"].astype(int)

#pearson correlation heatmap matrix
plt.figure(figsize=(8,6))
sns.heatmap(
    corr_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix — Transit, Mobility, & Disinvestment")
plt.show()

#Because rail access is spatially concentrated near the stadium (as reflected in the moderate correlation 
# between distance to train and distance to stadium), relying on rail-based mobility alone would concentrate 
# World Cup economic spillovers near downtown rather than dispersing them across Atlanta’s neighborhoods.


#regression

reg_df = full_df[[
    "Economic M",
    "dist_to_stop_m",
    "dist_to_train_m",
    "dist_to_stadium_m",
    "disinvested_neighborhood"
]].dropna()

reg_df["disinvested_neighborhood"] = reg_df["disinvested_neighborhood"].astype(int)

X = reg_df[["dist_to_stop_m","dist_to_train_m","dist_to_stadium_m","disinvested_neighborhood"]]
y = reg_df["Economic M"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

#VIF calculation makes sure there
# is not strong multicollinearity between the chosen variables
vif_df = pd.DataFrame()
vif_df["variable"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_df)



#spatial analysis 
coords = np.column_stack([full_df.geometry.x, full_df.geometry.y])
tree = KDTree(coords)

threshold = 1500  # adjust as desired
neighbors = tree.query_ball_tree(tree, r=threshold)

parent = list(range(len(full_df)))

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(i, j):
    pi, pj = find(i), find(j)
    if pi != pj:
        parent[pi] = pj

for i, nbrs in enumerate(neighbors):
    for j in nbrs:
        union(i, j)

component_ids = [find(i) for i in range(len(full_df))]
full_df['component_id'] = component_ids

# --- Step 3: Convert to lat/lon for Folium ---
full_df_latlon = full_df.to_crs(epsg=4326)

# --- Step 4: Assign colors to components ---
num_colors = 20
np.random.seed(42)
colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
color_list = [colors[c % num_colors] for c in component_ids]
hex_colors = [mcolors.to_hex(c) for c in color_list]

# --- Step 5: Add each business as a circle marker ---
for idx, row in full_df_latlon.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=3,
        color=hex_colors[idx],
        fill=True,
        fill_opacity=0.7
    ).add_to(basemap)

# --- Step 6: Display map ---
basemap



dis_df = full_df_latlon[full_df_latlon["disinvested_neighborhood"] == True]
all_df = full_df_latlon

#stadium_lat = float(stadium_point.y.iloc[0])
#stadium_lon = float(stadium_point.x.iloc[0])

m = folium.Map(location=[stadium_lat, stadium_lon], zoom_start=13, tiles="CartoDB Positron")

folium.Marker(
    location=[stadium_lat, stadium_lon],
    popup="Mercedes-Benz Stadium",
    icon=folium.Icon(color="red", icon="star", prefix="fa")
).add_to(m)

heat_data = [
    [float(row.geometry.y), float(row.geometry.x)]
    for _, row in dis_df.iterrows()
]

heat_layer = HeatMap(
    heat_data,
    name="Disinvested Business Heatmap",
    radius=15,
    blur=12,
    min_opacity=0.4
)
m.add_child(heat_layer)

marker_layer = FeatureGroup(name="Disinvested Businesses (points)")

for _, row in dis_df.iterrows():
    folium.CircleMarker(
        location=[float(row.geometry.y), float(row.geometry.x)],
        radius=4,
        color="yellow",
        fill=True,
        fill_opacity=0.8,
        popup=f"{row.get('business_name','Business')}<br>Lat: {row.geometry.y:.6f}, Lon: {row.geometry.x:.6f}"
    ).add_to(marker_layer)

m.add_child(marker_layer)

all_layer = FeatureGroup(name="All Businesses (points)")
for _, row in all_df.iterrows():
    folium.CircleMarker(
        location=[float(row.geometry.y), float(row.geometry.x)],
        radius=2,
        color="blue",
        fill=True,
        fill_opacity=0.5
    ).add_to(all_layer)

m.add_child(all_layer)
folium.LayerControl().add_to(m)

m

m.save("atlanta_disinvested_businesses_map.html")

#i want one stop that reaches Business Lat: 33.786596, Lon: -84.491483

#one that reaches Business Lat: 33.658172, Lon: -84.387688

#and one that reaches Business Lat: 33.698288, Lon: -84.483900


#find the fastest driving routes from stadium_point to denser areas

ox.settings.use_cache = True
ox.settings.log_console = True

stadium_lat = float(stadium_point.y)
stadium_lon = float(stadium_point.x)
stadium_xy = (stadium_lat, stadium_lon)

business_coords = [
    (33.786596, -84.491483),
    (33.658172, -84.387688),
    (33.698288, -84.483900)
]

G = ox.graph_from_point((stadium_lat, stadium_lon), dist=15000, network_type='drive')
origin_node = ox.nearest_nodes(G, stadium_lon, stadium_lat)
target_nodes = [
    ox.nearest_nodes(G, lon, lat)
    for (lat, lon) in business_coords
]

computed_routes = []
for target in target_nodes:
    route = nx.shortest_path(G, origin_node, target, weight='length')
    computed_routes.append(route)

# Plot routes on the network (single call)
fig, ax = ox.plot_graph_routes(
    G,
    computed_routes,
    route_colors=['r', 'g', 'b'][:len(computed_routes)],
    route_linewidth=3,
    node_size=0
)


#now we are going to place hypothetical stops along these routes
# --- Create a function to create stops along routes with proper distance calculations ---
def create_stops_along_route(route_nodes, G, spacing_meters=500):
    """
    Create stops along a route at regular intervals in meters.
    
    Parameters:
    - route_nodes: List of node IDs
    - G: NetworkX graph
    - spacing_meters: Distance between stops in meters
    
    Returns:
    - List of (lat, lon) tuples for stops
    """
    stops = []
    
    # Get coordinates for all nodes in the route
    coords = []
    for node in route_nodes:
        node_data = G.nodes[node]
        coords.append((node_data['x'], node_data['y']))  # (lon, lat)
    
    # Create LineString
    line = LineString(coords)
    
    # Convert to UTM for accurate distance measurements
    line_gdf = gpd.GeoDataFrame(
        geometry=[line],
        crs="EPSG:4326"
    )
    
    # Convert to UTM (zone 16N for Atlanta)
    line_utm = line_gdf.to_crs(epsg=26916)
    line_utm_geom = line_utm.iloc[0].geometry
    
    # Get length in meters
    length_meters = line_utm_geom.length
    
    # Calculate number of stops
    num_stops = int(length_meters / spacing_meters)
    
    # Create stops along the UTM line
    for i in range(num_stops + 1):
        distance_along = i * spacing_meters
        point_utm = line_utm_geom.interpolate(distance_along)
        
        # Convert back to lat/lon
        point_gdf = gpd.GeoDataFrame(
            geometry=[point_utm],
            crs="EPSG:26916"
        ).to_crs(epsg=4326)
        
        point = point_gdf.iloc[0].geometry
        stops.append((point.y, point.x))  # (lat, lon)
    
    return stops

# --- Now create hypothetical stops using the corrected function ---
print("\n" + "="*60)
print("CREATING HYPOTHETICAL STOPS ALONG ROUTES")
print("="*60)

hypothetical_stops = []

for i, route in enumerate(computed_routes):
    print(f"Creating stops along route {i+1}...")
    route_stops = create_stops_along_route(route, G, spacing_meters=500)
    hypothetical_stops.extend(route_stops)
    print(f"  Added {len(route_stops)} stops for route {i+1}")

print(f"\nTotal hypothetical stops created: {len(hypothetical_stops)}")

# Visualize all stops on the map
print("\nAdding stops to map...")
for lat, lon in hypothetical_stops:
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color="orange",
        fill=True,
        fill_opacity=0.8,
        popup=f"Hypothetical Stop<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
    ).add_to(basemap)

# --- Analyze disinvested businesses near these stops ---
print("\n" + "="*60)
print("ANALYZING BUSINESSES NEAR HYPOTHETICAL STOPS")
print("="*60)

# Create GeoDataFrame for hypothetical stops
hyp_stops_gdf = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in hypothetical_stops],
    crs="EPSG:4326"
).to_crs(epsg=26916)

# Check how many disinvested businesses we have
print(f"\nChecking business data...")
print(f"Total businesses in dataset: {len(full_df)}")
print(f"Businesses with 'disinvested_neighborhood' = True: {len(full_df[full_df['disinvested_neighborhood'] == True])}")
print(f"Unique values in disinvested_neighborhood: {full_df['disinvested_neighborhood'].unique()}")

# Prepare business data
if len(full_df[full_df['disinvested_neighborhood'] == True]) == 0:
    print("\nWARNING: No businesses marked as 'disinvested_neighborhood' = True.")
    print("Using alternative approach: selecting businesses in low mobility areas...")
    
    # Calculate threshold for low mobility (bottom 25%)
    if len(full_df) > 0:
        low_mobility_threshold = full_df['Economic M'].quantile(0.25)
        print(f"Low mobility threshold (25th percentile): {low_mobility_threshold}")
        dis_businesses_gdf = full_df[full_df['Economic M'] <= low_mobility_threshold].copy()
        print(f"Found {len(dis_businesses_gdf)} businesses in low mobility areas")
    else:
        print("ERROR: No business data available.")
        dis_businesses_gdf = gpd.GeoDataFrame()
else:
    dis_businesses_gdf = full_df[full_df['disinvested_neighborhood'] == True].copy()
    print(f"Using {len(dis_businesses_gdf)} disinvested businesses")

if len(dis_businesses_gdf) > 0:
    # Convert to UTM
    dis_businesses_gdf = dis_businesses_gdf.to_crs(epsg=26916)
    
    # Clean up
    dis_businesses_gdf = dis_businesses_gdf.drop(columns=['index_right'], errors='ignore')
    dis_businesses_gdf = dis_businesses_gdf.reset_index(drop=True)
    hyp_stops_gdf = hyp_stops_gdf.reset_index(drop=True)
    
    # Create buffers around stops (400m walking distance)
    buffer_dist = 400
    print(f"\nCreating {buffer_dist}m buffers around each stop...")
    hyp_stops_gdf['geometry_buffer'] = hyp_stops_gdf.geometry.buffer(buffer_dist)
    buffered_stops = hyp_stops_gdf.set_geometry('geometry_buffer')
    
    # Perform spatial join
    print("Performing spatial join...")
    joined = gpd.sjoin(
        dis_businesses_gdf, 
        buffered_stops, 
        how='left', 
        predicate='within',
        lsuffix='_business',
        rsuffix='_stop'
    )
    
    # Count businesses per stop
    if 'index_stop' in joined.columns:
        business_counts = joined.groupby('index_stop').size()
    else:
        business_counts = pd.Series([0] * len(hyp_stops_gdf), index=hyp_stops_gdf.index)
    
    business_counts = business_counts.reindex(hyp_stops_gdf.index, fill_value=0)
    hyp_stops_gdf['businesses_nearby'] = business_counts.values
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total hypothetical stops: {len(hyp_stops_gdf)}")
    print(f"Stops serving at least 1 business: {(hyp_stops_gdf['businesses_nearby'] > 0).sum()}")
    print(f"Total businesses served: {hyp_stops_gdf['businesses_nearby'].sum()}")
    print(f"Average businesses per stop: {hyp_stops_gdf['businesses_nearby'].mean():.2f}")
    print(f"Maximum businesses at one stop: {hyp_stops_gdf['businesses_nearby'].max()}")
    
    # Show top stops
    if hyp_stops_gdf['businesses_nearby'].max() > 0:
        print("\nTop 10 stops serving businesses:")
        top_stops = hyp_stops_gdf.nlargest(10, 'businesses_nearby')
        
        # Convert to lat/lon for display
        top_stops_latlon = top_stops.to_crs(epsg=4326)
        
        for idx, row in top_stops_latlon.iterrows():
            print(f"  {idx+1}. ({row.geometry.y:.6f}, {row.geometry.x:.6f}): "
                  f"{int(row['businesses_nearby'])} businesses")
    else:
        print("\nNo businesses found within 400m of any stops.")
        print("Try increasing buffer distance or check if routes pass through business areas.")
    
    # Add color-coded stops to map
    hyp_stops_latlon = hyp_stops_gdf.to_crs(epsg=4326)
    max_businesses = max(1, hyp_stops_latlon['businesses_nearby'].max())
    
    print(f"\nAdding color-coded stops to map...")
    for idx, row in hyp_stops_latlon.iterrows():
        business_count = row['businesses_nearby']
        
        if business_count > 0:
            # Color scale from green (few) to red (many)
            color_intensity = min(1.0, business_count / max_businesses)
            red = int(255 * color_intensity)
            green = int(255 * (1 - color_intensity))
            blue = 0
            color = f'#{red:02x}{green:02x}{blue:02x}'
            size = 4 + min(8, business_count * 2)
            
            popup_text = f"<b>Hypothetical Stop</b><br>Nearby businesses: {business_count}<br>Lat: {row.geometry.y:.6f}<br>Lon: {row.geometry.x:.6f}"
        else:
            color = 'gray'
            size = 3
            popup_text = f"<b>Hypothetical Stop</b><br>No nearby businesses<br>Lat: {row.geometry.y:.6f}<br>Lon: {row.geometry.x:.6f}"
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=size,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=popup_text
        ).add_to(basemap)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 200px; height: 160px; 
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white;
         padding: 10px;">
         <b>Stop Legend</b><br>
         <i class="fa fa-circle" style="color:red"></i> Many businesses<br>
         <i class="fa fa-circle" style="color:orange"></i> Some businesses<br>
         <i class="fa fa-circle" style="color:green"></i> Few businesses<br>
         <i class="fa fa-circle" style="color:gray"></i> No businesses<br>
         <i class="fa fa-star" style="color:red"></i> Stadium<br>
         <i class="fa fa-circle" style="color:orange"></i> Original stops
    </div>
    '''
    basemap.get_root().html.add_child(folium.Element(legend_html))
    
    print("\nMap updated with color-coded stops:")
    print("  Red: Stops with many businesses")
    print("  Green: Stops with few businesses")
    print("  Gray: Stops with no businesses")
    print("  Orange dots: Original stop locations")
else:
    print("\nERROR: No business data available for analysis.")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

# Save the updated map
print("\nSaving map to 'hypothetical_stops_analysis.html'...")
basemap.save("hypothetical_stops_analysis.html")
print("Map saved successfully!")

# Display the map
basemap


# --- Create map centered on stadium ---
m = folium.Map(location=[stadium_lat, stadium_lon], zoom_start=13, tiles="CartoDB Positron")

# Stadium marker
folium.Marker(
    location=[stadium_lat, stadium_lon],
    popup="Mercedes-Benz Stadium",
    icon=folium.Icon(color="red", icon="star", prefix="fa")
).add_to(m)

# --- Disinvested business heatmap ---
dis_df = full_df_latlon[full_df_latlon["disinvested_neighborhood"] == True]
heat_data = [[row.geometry.y, row.geometry.x] for _, row in dis_df.iterrows()]
HeatMap(heat_data, radius=15, blur=12, min_opacity=0.4, name="Disinvested Business Heatmap").add_to(m)

# --- All businesses layer ---
all_layer = FeatureGroup(name="All Businesses")
for _, row in full_df_latlon.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,
        color="blue",
        fill=True,
        fill_opacity=0.5
    ).add_to(all_layer)
m.add_child(all_layer)

# --- Disinvested businesses layer ---
dis_layer = FeatureGroup(name="Disinvested Businesses")
for _, row in dis_df.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color="yellow",
        fill=True,
        fill_opacity=0.8,
        popup=f"{row.get('business_name','Business')}<br>Lat: {row.geometry.y:.6f}, Lon: {row.geometry.x:.6f}"
    ).add_to(dis_layer)
m.add_child(dis_layer)

# --- Real MARTA routes using stops_gdf ---
real_routes_layer = FeatureGroup(name="Real MARTA Routes")

# Ensure stop_id is string
stops_gdf['stop_id'] = stops_gdf['stop_id'].astype(str)

# Extract lat/lon from geometry if not already done
if 'stop_lat' not in stops_gdf.columns or 'stop_lon' not in stops_gdf.columns:
    stops_gdf['stop_lat'] = stops_gdf.geometry.y
    stops_gdf['stop_lon'] = stops_gdf.geometry.x

stop_cols = ['stop_id', 'stop_lat', 'stop_lon']

for route_id in routes_near_stadium['route_id']:
    # Get one representative trip
    route_trips = trips[trips['route_id'] == route_id]['trip_id'].unique()
    if len(route_trips) == 0:
        continue
    trip_id = route_trips[0]

    # Get stops for this trip
    route_stop_seq = stop_times[stop_times['trip_id'] == trip_id].sort_values('stop_sequence')

    # Merge with stops_gdf
    route_joined = route_stop_seq.merge(stops_gdf[stop_cols], on='stop_id', how='left')

    # Drop missing coordinates
    route_joined = route_joined.dropna(subset=['stop_lat','stop_lon'])

    # Convert to list of (lat, lon)
    route_coords = list(zip(route_joined['stop_lat'], route_joined['stop_lon']))

    if len(route_coords) > 1:
        folium.PolyLine(
            locations=route_coords,
            color='blue',
            weight=3,
            opacity=0.7,
            popup=f"Route {route_id}"
        ).add_to(real_routes_layer)

m.add_child(real_routes_layer)

# --- Hypothetical stops layer ---
hypothetical_layer = FeatureGroup(name="Hypothetical Stops")
for lat, lon in hypothetical_stops:
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color="orange",
        fill=True,
        fill_opacity=0.8,
        popup=f"Hypothetical Stop<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
    ).add_to(hypothetical_layer)

m.add_child(hypothetical_layer)

# --- Layer control ---
folium.LayerControl().add_to(m)

# --- Save map ---
m.save("atlanta_real_and_hypothetical_routes.html")
print("Map saved as 'atlanta_real_and_hypothetical_routes.html'")
