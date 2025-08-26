import pandas as pd

# Load your dataset
df = pd.read_csv("attractions.csv")

# Updated mapping with 'free'
cost_mapping = {
    'free': 0,     # no entry fee
    'low': 25,     # small ticket fee
    'medium': 75,  # mid-range attractions
    'high': 150    # expensive attractions
}

# Create a new 'cost' column
df['cost'] = df['cost_level'].map(cost_mapping)

# Save it back to CSV
df.to_csv("attractions.csv", index=False)

print("done")
