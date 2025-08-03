import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import KMeans

# Load the Dataset
df = pd.read_csv("TURNO_DATA.csv")

# Q1: Residence Location (nighttime GPS)
night_data = df[df['hr'].between(0, 4)].dropna(subset=['avg_lat', 'avg_long'])
night_data['lat_rounded'] = night_data['avg_lat'].round(3)
night_data['long_rounded'] = night_data['avg_long'].round(3)
residence_locations = (
    night_data.groupby(['vin', 'lat_rounded', 'long_rounded'])
    .size().reset_index(name='frequency')
    .sort_values(['vin', 'frequency'], ascending=[True, False])
    .drop_duplicates('vin')
    .rename(columns={'lat_rounded': 'res_lat', 'long_rounded': 'res_long'})
)

# Q2: Charging Location (battery % increase > 10)
df = df.rename(columns={'yearr': 'year', 'mmm': 'month', 'ddd': 'day'})
df['half_hour_num'] = df['half_hour'].map({'h1': 0, 'h2': 1})
df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day']]) + \
                  pd.to_timedelta(df['hr'], unit='h') + \
                  pd.to_timedelta(df['half_hour_num'] * 30, unit='m')
df = df.sort_values(['vin', 'timestamp'])
df['bat_diff'] = df.groupby('vin')['avg_bat_charge'].diff()

charging_events = df[df['bat_diff'] > 10].dropna(subset=['avg_lat', 'avg_long'])
charging_events['lat_rounded'] = charging_events['avg_lat'].round(3)
charging_events['long_rounded'] = charging_events['avg_long'].round(3)
charging_locations = (
    charging_events.groupby(['vin', 'lat_rounded', 'long_rounded'])
    .size().reset_index(name='frequency')
    .sort_values(['vin', 'frequency'], ascending=[True, False])
    .drop_duplicates('vin')
    .rename(columns={'lat_rounded': 'charge_lat', 'long_rounded': 'charge_long'})
)

# Q3: Drivers at Default Risk (low daily km)
df['date'] = df['timestamp'].dt.date
day_data = df[df['hr'].between(6, 22)].dropna(subset=['avg_lat', 'avg_long'])
day_data = day_data[
    (day_data['avg_lat'].between(-90, 90)) & (day_data['avg_long'].between(-180, 180))
]

def compute_daily_distance(group):
    group = group.sort_values('timestamp')
    coords = list(zip(group['avg_lat'], group['avg_long']))
    distances = [geodesic(coords[i], coords[i + 1]).km for i in range(len(coords) - 1)]
    return pd.Series({'total_distance_km': sum(distances)})

distance_summary = (
    day_data.groupby(['vin', 'date'])
    .apply(compute_daily_distance)
    .reset_index()
)

vin_distance_avg = (
    distance_summary.groupby('vin')['total_distance_km']
    .mean().reset_index(name='avg_daily_distance_km')
)
vin_distance_avg['low_earning_risk'] = vin_distance_avg['avg_daily_distance_km'] < 10

# Q4: Cluster high/low activity zones (KMeans)
location_data = df.dropna(subset=['avg_lat', 'avg_long'])
location_data = location_data[
    (location_data['avg_lat'].between(-90, 90)) & (location_data['avg_long'].between(-180, 180))
]
sampled_data = location_data[['avg_lat', 'avg_long']].sample(n=20000, random_state=42)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
sampled_data['cluster'] = kmeans.fit_predict(sampled_data[['avg_lat', 'avg_long']])
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['center_lat', 'center_long'])
cluster_centers['cluster'] = cluster_centers.index
cluster_sizes = sampled_data['cluster'].value_counts().reset_index()
cluster_sizes.columns = ['cluster', 'count']
cluster_summary = pd.merge(cluster_centers, cluster_sizes, on='cluster')

# Save to Excel
output_file = "Shrivastava-Devesh-AUTO.xlsx"
with pd.ExcelWriter(output_file) as writer:
    residence_locations.to_excel(writer, sheet_name="Q1_Residence", index=False)
    charging_locations.to_excel(writer, sheet_name="Q2_Charging", index=False)
    vin_distance_avg.to_excel(writer, sheet_name="Q3_DefaultRisk", index=False)
    cluster_summary.to_excel(writer, sheet_name="Q4_Clusters", index=False)

print(f"✅ Results saved to {output_file}")
