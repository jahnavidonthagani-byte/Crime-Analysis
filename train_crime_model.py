import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -----------------------------
# 1️⃣ Load your dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\dkavy\OneDrive\Desktop\Logistic Regression\knowledge\crime_dataset_india.csv")

# -----------------------------
# 2️⃣ Create the 'highCrime' target column
# -----------------------------
# If the dataset has ViolentCrimesPerPop, use it; otherwise, use Crime Code as a proxy
if 'ViolentCrimesPerPop' in df.columns:
    df['highCrime'] = np.where(df['ViolentCrimesPerPop'] > 0.1, 1, 0)
else:
    df['highCrime'] = np.where(df['Crime Code'] > df['Crime Code'].median(), 1, 0)

# -----------------------------
# 3️⃣ Encode categorical columns
# -----------------------------
categorical_cols = ['Victim Gender','Weapon Used','Crime Domain','Case Closed','City']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -----------------------------
# 4️⃣ Define features (X) and target (y)
# -----------------------------
numeric_cols = ['Report Number','Crime Code','Victim Age','Police Deployed']
feature_columns = numeric_cols + [col + '_encoded' for col in categorical_cols]

X = df[feature_columns]
y = df['highCrime']

# -----------------------------
# 5️⃣ Split data into Train and Test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 6️⃣ Train the Random Forest model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 7️⃣ Predict and evaluate accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model training completed successfully!\n")
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8️⃣ Save model, encoders, and feature info
# -----------------------------
with open("crime_predictor.pkl", "wb") as f:
    pickle.dump((model, encoders, feature_columns), f)

# -----------------------------
# 9️⃣ Display feature importance
# -----------------------------
importance = model.feature_importances_
print("\n🔍 Feature Importance:")
for i, col in enumerate(feature_columns):
    print(f"{col}: {importance[i]:.4f}")
