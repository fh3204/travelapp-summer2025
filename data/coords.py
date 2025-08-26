import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

# Read dataset
df = pd.read_csv("attractions.csv")

# Initialize geocoder
geolocator = Nominatim(user_agent="itinerary-mapper")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Create storage for coordinates
unique_names = df['attraction_name'].unique()
coords = {}

# Geocode each unique attraction name
print("Geocoding attractions (may take a few minutes)...")
for name in tqdm(unique_names):
    location = geocode(f"{name}, Dubai")
    if location:
        coords[name] = (location.latitude, location.longitude)
    else:
        coords[name] = (None, None)

# Map coordinates back to the dataframe
df['lat'] = df['attraction_name'].map(lambda name: coords[name][0])
df['lng'] = df['attraction_name'].map(lambda name: coords[name][1])

# Save to new CSV
df.to_csv("attractions_with_coords.csv", index=False)
print("done")
