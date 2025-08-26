from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime, timedelta
from travel_time import get_travel_time

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

# Load data and model
data = pd.read_csv('data/attractions.csv')
model = joblib.load('model/rf_model.pkl')
original_data = data.copy()


# Preprocess for prediction
data_encoded = data.drop(columns=['attraction_name'])
data_encoded = pd.get_dummies(data_encoded)
model_features = model.feature_names_in_
data_encoded = data_encoded.reindex(columns=model_features, fill_value=0)
predictions = model.predict(data_encoded)
original_data['score'] = predictions


def make_feature_vector(group_type, budget, interests):
    feature = {
        'cost_level_low': 0,
        'cost_level_medium': 0,
        'cost_level_high': 0,
        'age_range_kids_no': 0,
        'age_range_kids_yes': 0,
        'travel_type_solo': 0,
        'travel_type_couple': 0,
        'travel_type_family': 0,
        'travel_type_friends': 0
    }

    if budget < 1500:
        feature['cost_level_low'] = 1
    elif budget < 5000:
        feature['cost_level_medium'] = 1
    else:
        feature['cost_level_high'] = 1

    if group_type == 'family':
        feature['age_range_kids_yes'] = 1
    else:
        feature['age_range_kids_no'] = 1

    if f'travel_type_{group_type}' in feature:
        feature[f'travel_type_{group_type}'] = 1

    for col in model_features:
        if col.startswith('attraction_type_'):
            keyword = col.split('attraction_type_')[1].lower()
            feature[col] = int(any(keyword in i.lower() for i in interests))

    return pd.DataFrame([feature]).reindex(columns=model_features, fill_value=0)


def create_itinerary(filtered_df, trip_duration, start_date_str, start_time_str):
    itinerary = []
    used_places = set()
    start_date = datetime.strptime(start_date_str, "%d-%m-%Y")

    # Drop rows with missing lat/lng
    filtered_df = filtered_df.dropna(subset=['latitude', 'longitude'])

    # Separate Hatta attractions
    hatta_df = filtered_df[filtered_df['attraction_name'].str.startswith("Hatta")].copy()
    non_hatta_df = filtered_df[~filtered_df['attraction_name'].str.startswith("Hatta")].copy()
    hatta_used = False

    for day_num in range(trip_duration):
        today_date = (start_date + timedelta(days=day_num)).strftime("%d-%m-%Y")
        day_activities = []
        current_time = datetime.strptime(start_time_str, "%I:%M %p")
        prev_lat, prev_lng = None, None

        if not hatta_used and not hatta_df.empty:
            # Schedule all Hatta attractions for this day
            daily_choices = hatta_df
            hatta_used = True
        else:
            remaining_places = non_hatta_df[~non_hatta_df['attraction_name'].isin(used_places)]
            if remaining_places.empty:
                break
            daily_choices = remaining_places.sample(n=min(3, len(remaining_places)), replace=False)

        used_places.update(daily_choices['attraction_name'].values)

        for _, row in daily_choices.iterrows():
            # Activity start time
            activity_time_str = current_time.strftime("%I:%M %p")
            activity_duration_hours = float(row.get('duration_hours', 2) or 2)

            # Default travel time
            travel_time_min = 0
            if prev_lat is not None and prev_lng is not None:
                travel_time_min = get_travel_time((prev_lat, prev_lng), (row['latitude'], row['longitude']))
                if travel_time_min in ("-", None):
                    travel_time_min = get_travel_time(
                        (prev_lat, prev_lng),
                        (row['latitude'], row['longitude']),
                        radius=3000
                    )
                travel_time_min = travel_time_min if travel_time_min not in (None, "-") else 0

                # Add travel time to arrival
                current_time += timedelta(minutes=travel_time_min)
                activity_time_str = current_time.strftime("%I:%M %p")

            # Add activity to list
            day_activities.append({
                'name': row['attraction_name'],
                'time': activity_time_str,
                'description': row['attraction_name'],
                'lat': float(row['latitude']),
                'lng': float(row['longitude']),
                'travel_time': (str(travel_time_min) + " mins.") if travel_time_min else "-",
                'cost': row.get('cost', "-")
            })

            # Update time for next activity
            current_time += timedelta(hours=activity_duration_hours)
            prev_lat, prev_lng = row['latitude'], row['longitude']

        itinerary.append({
            'day': f"Day {day_num + 1}",
            'date': today_date,
            'activities': day_activities
        })

    return itinerary


@app.route('/api/generate-itinerary', methods=['POST'])
def generate_itinerary():
    payload = request.get_json()
    group_type = payload['group_type']
    budget = payload['budget']
    interests = payload['interests']
    duration = payload['trip_duration']
    destination = payload['destination']
    start_date = payload['start_date']
    end_date = payload['end_date']
    start_time = payload['start_time']

    # Create user vector (future extensibility)
    _ = make_feature_vector(group_type, budget, interests)

    # Get top matching attractions
    top_df = original_data[original_data['score'] == 1].copy()
    if len(top_df) < duration * 3:
        top_df = original_data.sort_values(by='score', ascending=False).head(duration * 3 + 5)

    # Attractions for itinerary generation
    top_itinerary_df = (
        top_df[['attraction_name', 'attraction_type', 'popularity_score', 'duration_hours', 'latitude', 'longitude', 'cost']]
        .copy()
        .sort_values(by='popularity_score', ascending=False)
        .head(duration * 3)
    )

    # Top 4 highlights for display
    top_recommendations = (
        top_itinerary_df
        .head(4)
        .rename(columns={
            'attraction_name': 'name',
            'attraction_type': 'category',
            'popularity_score': 'score',
            'duration_hours': 'average_hours',
            'cost': 'cost'
        })
    )

    # Create itinerary using full set
    itinerary = create_itinerary(
        top_itinerary_df.rename(columns={
            'attraction_name': 'name',
            'attraction_type': 'category',
            'popularity_score': 'score',
            'duration_hours': 'average_hours'
        }).rename(columns={
            'name': 'attraction_name',
            'category': 'attraction_type',
            'score': 'popularity_score',
            'average_hours': 'duration_hours'
        }),
        duration,
        start_date,
        start_time
    )

    return jsonify({
        'data': {
            'destination': destination,
            'group_type': group_type,
            'budget': budget,
            'start_date': start_date,
            'end_date': end_date,
            'trip_duration': duration,
            'interests': interests,
            'recommendations': top_recommendations.to_dict(orient='records'),
            'daily_plan': itinerary
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=8000)
