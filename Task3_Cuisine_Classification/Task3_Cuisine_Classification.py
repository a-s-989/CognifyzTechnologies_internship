import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\as140\OneDrive\Desktop\Clg\ML internship\Dataset .csv")

# 1. Preprocessing
df['Primary_Cuisine'] = df['Cuisines'].fillna('Unknown').str.split(',').str[0].str.strip()

# Filter rare cuisines 
cuisine_counts = df['Primary_Cuisine'].value_counts()
common_cuisines = cuisine_counts[cuisine_counts > len(df)*0.02].index
df = df[df['Primary_Cuisine'].isin(common_cuisines)]

# Select relevant features
features = [
    'City', 'Price range', 'Average Cost for two', 
    'Aggregate rating', 'Votes', 'Has Table booking', 
    'Has Online delivery', 'Primary_Cuisine'
]
df = df[features].dropna(subset=['Primary_Cuisine'])

# 2. Data Preparation
X = df.drop(columns=['Primary_Cuisine'])
y = df['Primary_Cuisine']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3. Preprocessing Pipeline
categorical_features = ['City', 'Has Table booking', 'Has Online delivery']
numerical_features = ['Price range', 'Average Cost for two', 'Aggregate rating', 'Votes']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Model Training
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Confusion Matrix
plt.figure(figsize=(15, 12))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_)
plt.title('Cuisine Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cuisine_confusion_matrix.png', dpi=300)
plt.show()

# 6. Performance Analysis
# Get per-class metrics
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
metrics_df = pd.DataFrame(report).transpose()

# Identify problematic classes
problem_cuisines = metrics_df[metrics_df['support'] > 20]  # Filter classes with sufficient samples
problem_cuisines = problem_cuisines[problem_cuisines['f1-score'] < 0.6]

print("\nCuisines with Classification Challenges:")
print(problem_cuisines[['precision', 'recall', 'f1-score', 'support']])

# 7. Feature Importance Analysis
# Extract feature names
cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
all_features = numerical_features + list(cat_features)

# Get feature importances
importances = model.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Predictive Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Features for Cuisine Classification')
plt.tight_layout()
plt.savefig('cuisine_feature_importance.png', dpi=300)
plt.show()