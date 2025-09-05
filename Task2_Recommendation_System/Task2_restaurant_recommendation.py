import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset 
# Assuming columns: ['Restaurant Name', 'Cuisines', 'Price range', 'City', 'Aggregate rating', 'Votes']
df = pd.read_csv(r"C:\Users\as140\OneDrive\Desktop\Clg\ML internship\Dataset .csv")

# 1. Preprocessing
def preprocess_data(df):
    # Handle missing values
    df['Cuisines'] = df['Cuisines'].fillna('Unknown').str.split(', ')
    df['Price range'] = df['Price range'].fillna(2).astype(int)  # Median price
    
    # Create binary features
    df['Popular'] = (df['Votes'] > df['Votes'].median()).astype(int)
    df['Highly Rated'] = (df['Aggregate rating'] > 4.0).astype(int)
    
    return df

df = preprocess_data(df)

# 2. Feature Engineering
mlb = MultiLabelBinarizer()
cuisine_features = mlb.fit_transform(df['Cuisines'])
cuisine_df = pd.DataFrame(cuisine_features, columns=mlb.classes_)

# Create feature matrix
feature_matrix = pd.DataFrame({
    'Price Range': df['Price range'],
    'Popular': df['Popular'],
    'Highly Rated': df['Highly Rated']
})

feature_matrix = pd.concat([feature_matrix, cuisine_df], axis=1)

# 3. Recommendation System
def recommend_restaurants(user_preferences, df, feature_matrix, top_n=5):
    """
    user_preferences: dict with keys:
        - 'cuisines': list of preferred cuisines
        - 'max_price': integer (1-4)
        - 'min_rating': float (0-5)
        - 'popular': bool
    """
    # Create user profile vector
    user_vector = np.zeros(len(feature_matrix.columns))
    price_idx = feature_matrix.columns.get_loc('Price Range')
    
    # Set price preference (lower price is better)
    user_vector[price_idx] = 5 - user_preferences.get('max_price', 4)
    
    # Set binary features
    if user_preferences.get('popular', False):
        user_vector[feature_matrix.columns.get_loc('Popular')] = 1
    if user_preferences.get('min_rating', 0) > 4.0:
        user_vector[feature_matrix.columns.get_loc('Highly Rated')] = 1
    
    # Set cuisine preferences
    for cuisine in user_preferences.get('cuisines', []):
        if cuisine in feature_matrix.columns:
            user_vector[feature_matrix.columns.get_loc(cuisine)] = 1
    
    # Calculate cosine similarity
    similarities = cosine_similarity([user_vector], feature_matrix)[0]
    
    # Filter by minimum rating
    min_rating = user_preferences.get('min_rating', 0)
    valid_indices = df[df['Aggregate rating'] >= min_rating].index
    
    # Get top recommendations
    df['Similarity'] = similarities
    recommendations = df.loc[valid_indices].sort_values('Similarity', ascending=False).head(top_n)
    
    return recommendations[['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating', 'City']]

# 4. Test the System
if __name__ == "__main__":
    # Sample user preferences
    user_prefs = {
        'cuisines': ['Italian', 'Pizza'],
        'max_price': 3,
        'min_rating': 4.2,
        'popular': True
    }
    
    # Get recommendations
    recommendations = recommend_restaurants(user_prefs, df, feature_matrix)
    
    print("Top Restaurant Recommendations:")
    print(recommendations)