import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Load dataset
df = pd.read_csv("data/attractions.csv")

# Drop non-informative columns
df = df.drop(columns=["attraction_name", "day_number", "age_range_kids"])

# Fill NaNs
df = df.fillna("")

# Convert attraction_type to binary columns
def split_and_strip(x):
    return [item.strip() for item in str(x).split(",") if item.strip()]

mlb_type = MultiLabelBinarizer()
type_encoded = mlb_type.fit_transform(df['attraction_type'].apply(split_and_strip))
df_type = pd.DataFrame(type_encoded, columns=[f"type_{c}" for c in mlb_type.classes_])
df = pd.concat([df.drop(columns=["attraction_type"]), df_type], axis=1)

# Encode cost_level and travel_type
df = pd.get_dummies(df, columns=["cost_level", "travel_type"])

# Create is_indoor if not present
if 'is_indoor' not in df.columns and 'is_outdoor' in df.columns:
    df['is_indoor'] = 1 - df['is_outdoor']

# Separate features and target
X = df.drop(columns=["is_recommended"])
y = df["is_recommended"]

# Split into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training label distribution:")
print(y_train.value_counts())
print("Test label distribution:")
print(y_test.value_counts())

# Train model with class weights
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation accuracy:", round(cv_scores.mean(), 3))

# Save model
joblib.dump(model, "model/rf_model.pkl")
