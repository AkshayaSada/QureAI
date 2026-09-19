import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
file_path = "datasets/disease_prediction/Training.csv"
data = pd.read_csv(file_path)

# Remove unwanted CSV index columns
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Separate symptoms and disease
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Convert disease names into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing portions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create the machine learning model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train the model
print("\nTraining QureAI model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("       QUREAI MODEL")
print("==============================")
print("Model accuracy:", round(accuracy * 100, 2), "%")

# Save the trained model
joblib.dump(model, "backend/qureai_model.pkl")

# Save the label encoder
joblib.dump(label_encoder, "backend/label_encoder.pkl")

# Save the symptom names
joblib.dump(list(X.columns), "backend/symptoms.pkl")

print("\nModel saved successfully!")
print("Files created:")
print("1. backend/qureai_model.pkl")
print("2. backend/label_encoder.pkl")
print("3. backend/symptoms.pkl")
