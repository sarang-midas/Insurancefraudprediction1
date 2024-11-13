import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Step 1: Load the data
data = pd.read_csv('./data.csv')

# Step 2: Data py
# Drop unnecessary columns
data = data.drop(['policy_number', 'insured_zip', 'incident_location', '_c39'], axis=1, errors='ignore')

# Convert date columns to datetime if needed
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'], errors='coerce')
data['incident_date'] = pd.to_datetime(data['incident_date'], errors='coerce')

# Fill missing values for categorical and numerical columns
data.fillna(method='ffill', inplace=True)

# Step 3: Preprocessing
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Step 4: Define models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "SVC": SVC()
}

# Evaluate each model
best_score = 0
best_model = None

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    avg_score = scores.mean()
    print(f"{name} Accuracy: {avg_score:.2f}")

    if avg_score > best_score:
        best_score = avg_score
        best_model = pipeline

# Train the best model on the full training set and save it
best_model.fit(X_train, y_train)
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model saved with accuracy: {best_score:.2f}")
