import json
from pathlib import Path
from openrouteservice import Client
from openrouteservice.exceptions import ApiError

#  API key
client = Client(key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjUzYTJlYTRjNzRlNjQ3N2JhOGMwNzdiYjFlOWIyYTc1IiwiaCI6Im11cm11cjY0In0=")

# Path to cache file
CACHE_FILE = Path("travel_time_cache.json")

# Load cache from file if it exists
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, "r") as f:
            travel_time_cache = json.load(f)
    except json.JSONDecodeError:
        travel_time_cache = {}
else:
    travel_time_cache = {}

def save_cache():
    """Save the current cache to file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(travel_time_cache, f)

def get_travel_time(coord1, coord2, radius=None):
    """
    Returns travel time in minutes or "-" if failed.
    coord1, coord2 = (lat, lon) tuples
    """
    # Cache key 
    key = str((
        round(coord1[0], 6), round(coord1[1], 6),
        round(coord2[0], 6), round(coord2[1], 6),
        radius
    ))

    # If we already have it, return from cache
    if key in travel_time_cache:
        return travel_time_cache[key]

    c1 = (coord1[1], coord1[0])
    c2 = (coord2[1], coord2[0])

    try:
        kwargs = {
            'coordinates': [c1, c2],
            'profile': 'driving-car',
            'format': 'geojson'
        }
        if radius:
            kwargs['radiuses'] = [radius, radius]

        route = client.directions(**kwargs)
        duration_sec = route['features'][0]['properties']['segments'][0]['duration']
        minutes = round(duration_sec / 60, 1)

        travel_time_cache[key] = minutes
        save_cache()  # Save after getting a new value
        return minutes

    except ApiError:
        travel_time_cache[key] = "-"
        save_cache()
        return "-"
