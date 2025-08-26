import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load model
model = joblib.load('model/rf_model.pkl')

# Load dataset
df = pd.read_csv('data/attractions.csv')

# Save original names before dropping/encoding
original_df = df.copy()

# Drop attraction_name for modeling
df = df.drop(columns=['attraction_name'])

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['attraction_type', 'cost_level', 'age_range_kids', 'travel_type'])

# Align dataset to training format
model_features = model.feature_names_in_
df = df.reindex(columns=model_features, fill_value=0)

# Predict recommendations
predictions = model.predict(df)
original_df['predicted'] = predictions

# Filter only recommended attractions
recommended = original_df[original_df['predicted'] == 1].copy()


# Shuffle and group by attraction type
recommended = recommended.sample(frac=1, random_state=42)  # Shuffle

# Pick 3 unique attractions per day (max 7 days)
def select_diverse(df, days=3):
    selected = []
    used_types = set()
    for _, row in df.iterrows():
        if len(selected) >= days:
            break
        if row['attraction_type'] not in used_types:
            selected.append(row)
            used_types.add(row['attraction_type'])
    return pd.DataFrame(selected)

# For now just return 3 varied recommendations
output = select_diverse(recommended, days=3)

# Convert to JSON for Flask
def get_recommendations():
    return output[['attraction_name', 'attraction_type', 'duration_hours', 'popularity_score']].to_dict(orient='records')
