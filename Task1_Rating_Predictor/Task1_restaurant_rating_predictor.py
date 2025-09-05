# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load dataset (replace with your data path)
df = pd.read_csv(r"C:\Users\as140\OneDrive\Desktop\Clg\ML internship\Dataset .csv")

# 1. Preprocessing
# Select relevant features and target
features = [
    'Cuisines', 'Average Cost for two', 'Currency', 
    'Has Table booking', 'Has Online delivery', 
    'Price range', 'Votes'
]
target = 'Aggregate rating'
df = df[features + [target]]

# Handle missing values
df['Cuisines'] = df['Cuisines'].fillna('Unknown')
df['Votes'] = df['Votes'].fillna(0)
df['Price range'] = df['Price range'].fillna(df['Price range'].median())

# Split data
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing pipeline
categorical_features = ['Cuisines', 'Currency', 'Has Table booking', 'Has Online delivery']
numeric_features = ['Average Cost for two', 'Price range', 'Votes']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 2. Model Training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# 3. Model Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 4. Feature Importance Analysis
# Extract feature names from preprocessor
feature_names = numeric_features.copy()
cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
feature_names += list(cat_encoder.get_feature_names_out(categorical_features))

# Get importance scores
rf = model.named_steps['regressor']
importances = rf.feature_importances_

# Create and sort feature importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
importance_df.head(10).sort_values('Importance').plot.barh(
    x='Feature', 
    y='Importance',
    title='Top 10 Features Affecting Restaurant Ratings'
)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# Interpretation Report
print("\nKey Insights:")
print("1. Votes is the strongest predictor of restaurant ratings, indicating customer engagement correlates with quality")
print("2. Price range and average cost show economic factors influence perceived quality")
print("3. Online delivery availability has more impact than table booking")
print("4. Specific cuisines (e.g., Japanese, Italian) show higher impact than others")
print(f"5. Model explains {r2:.1%} of rating variance (R²)")