#import libraries

from flask import Flask, render_template, request, jsonify
import pandas as pd
import ast
from textblob import TextBlob
from fuzzywuzzy import process
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Create the web app
app = Flask(__name__)

print("Starting restaurant recommender...")

# Load the restaurant data
df = pd.read_csv('cleaned_dataset.csv')
df_full = df.copy()
print(f"Loaded {len(df_full)} restaurants")

# Show some info about the data
print(f"Cities: {len(df_full['City'].dropna().unique())}")
print(f"Sample cities: {sorted(df_full['City'].dropna().unique().tolist()[:5])}")

def calculate_sentiment(review_list):
    """
    Figure out if reviews are positive or negative
    Returns a score from -1 (bad) to 1 (good)
    """
    if pd.isnull(review_list) or not review_list or review_list == '[]':
        return 0  # No reviews = neutral
    
    try:
        # Convert string to list if needed
        if isinstance(review_list, str) and review_list.startswith('['):
            reviews = ast.literal_eval(review_list)
        else:
            reviews = [str(review_list)]
        
        sentiments = []
        for review in reviews:
            if review and isinstance(review, str) and len(str(review).strip()) > 0:
                try:
                    # Get sentiment score
                    sentiment_score = TextBlob(str(review)).sentiment.polarity
                    sentiments.append(sentiment_score)
                except:
                    continue
        
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except:
        return 0

# Add sentiment scores to all restaurants
print("Analyzing review sentiments...")
df_full['Sentiment_Score'] = df_full['Reviews'].apply(calculate_sentiment)
print("Done with sentiment analysis")

def convert_cuisine_string_to_list(cuisine_str):
    """Convert cuisine string to list of cuisines"""
    if pd.isnull(cuisine_str) or not cuisine_str or cuisine_str == '[]':
        return []
    
    try:
        cuisines = ast.literal_eval(cuisine_str)
        return [c.strip() for c in cuisines]
    except:
        return []

# Prepare cuisine data
df_full['Cuisine_Style_List'] = df_full['Cuisine Style'].apply(convert_cuisine_string_to_list)

print("Creating features for machine learning...")

# Create features for the algorithms
cuisine_dummies = df_full['Cuisine_Style_List'].str.join('|').str.get_dummies(sep='|')

# Normalize ratings and sentiment to 0-1 scale
df_full['Rating_norm'] = (df_full['Rating'] - df_full['Rating'].min()) / (df_full['Rating'].max() - df_full['Rating'].min())
df_full['Sentiment_norm'] = (df_full['Sentiment_Score'] - df_full['Sentiment_Score'].min()) / (df_full['Sentiment_Score'].max() - df_full['Sentiment_Score'].min())

# Fill missing values
df_full['Rating_norm'] = df_full['Rating_norm'].fillna(0.5)
df_full['Sentiment_norm'] = df_full['Sentiment_norm'].fillna(0.5)

# Combine all features
features = pd.concat([cuisine_dummies, df_full[['Rating_norm', 'Sentiment_norm']]], axis=1)
print(f"Created {features.shape[1]} features")

# Set up K-Means clustering with 7 groups
print("Setting up K-Means clustering...")
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_full['Cluster'] = kmeans.fit_predict(features)
print("Clustering done")

print("All algorithms ready!")
print(f"Using {len(df_full)} restaurants with {features.shape[1]} features")

def find_closest_restaurant(input_name):
    """Find the best matching restaurant name, even with typos"""
    if not input_name or not isinstance(input_name, str) or len(input_name.strip()) < 2:
        return None
        
    all_names = df_full['Name'].dropna().tolist()
    try:
        best_match, score = process.extractOne(input_name, all_names)
        print(f"Found match: '{input_name}' -> '{best_match}' ({score}%)")
        return best_match if best_match and score >= 50 else None
    except Exception as e:
        print(f"Matching error: {e}")
        return None

def recommend_content_based(name, top_n=10):
    """
    Algorithm 1: Find similar restaurants based on features
    Uses cosine similarity to compare restaurants
    """
    closest_name = find_closest_restaurant(name)
    if closest_name is None:
        return None, None, "No matching restaurant found"
    
    try:
        # Get the input restaurant
        input_idx = df_full[df_full['Name'] == closest_name].index[0]
        input_features = features.iloc[input_idx].values.reshape(1, -1)
        
        # Calculate similarity with all other restaurants
        similarities = []
        for idx in range(len(df_full)):
            if idx != input_idx:
                other_features = features.iloc[idx].values.reshape(1, -1)
                dot_product = np.dot(input_features, other_features.T)[0][0]
                input_norm = np.linalg.norm(input_features)
                other_norm = np.linalg.norm(other_features)
                
                if input_norm > 0 and other_norm > 0:
                    similarity = dot_product / (input_norm * other_norm)
                else:
                    similarity = 0
                
                similarities.append((idx, similarity))
        
        # Get most similar restaurants
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, sim in similarities[:top_n]]
        
        return closest_name, df_full.iloc[top_indices], None
    except Exception as e:
        print(f"Content-based error: {e}")
        return None, None, f"Error: {str(e)}"

def recommend_kmeans(name, top_n=10):
    """
    Algorithm 2: Find restaurants in the same cluster
    Uses the 7 clusters from K-means
    """
    closest_name = find_closest_restaurant(name)
    if closest_name is None:
        return None, None, "No matching restaurant found"
    
    try:
        # Get cluster of input restaurant
        input_restaurant = df_full[df_full['Name'] == closest_name].iloc[0]
        input_cluster = input_restaurant['Cluster']
        
        # Find other restaurants in same cluster
        cluster_restaurants = df_full[
            (df_full['Cluster'] == input_cluster) & 
            (df_full['Name'] != closest_name)
        ]
        
        if len(cluster_restaurants) == 0:
            return closest_name, cluster_restaurants, f"No other restaurants in cluster {input_cluster}"
        
        # Return highest rated ones
        return closest_name, cluster_restaurants.nlargest(top_n, 'Rating'), None
    except Exception as e:
        print(f"K-means error: {e}")
        return None, None, f"Error: {str(e)}"

def recommend_popular_based(city=None, cuisine=None, price=None, min_rating=None, top_n=10):
    """
    Algorithm 3: Recommend popular restaurants
    Uses weighted score: 50% rating + 30% sentiment
    """
    filtered_df = df_full.copy()
    
    # Apply filters
    if city:
        filtered_df = filtered_df[filtered_df['City'] == city]
    if cuisine:
        filtered_df = filtered_df[filtered_df['Cuisine_Style_List'].apply(
            lambda x: cuisine in x if x else False)]
    if price:
        price_map = {'Budget': '$', 'Moderate': '$$ - $$$', 'Luxury': '$$$$'}
        filtered_df = filtered_df[filtered_df['Price Range'] == price_map[price]]
    if min_rating:
        filtered_df = filtered_df[filtered_df['Rating'] >= float(min_rating)]
    
    # Calculate popularity score
    filtered_df['Popularity_Score'] = (
        0.5 * filtered_df['Rating_norm'] +
        0.3 * filtered_df['Sentiment_norm']+
                          0.2 *filtered_df['Number of Reviews']
    )
    
    # Return most popular
    return filtered_df.nlargest(top_n, 'Popularity_Score')

def filter_restaurants(city=None, cuisine=None, price=None, min_rating=None, top_n=20):
    """
    Algorithm 4: Simple filter search
    Basic filtering without complex algorithms
    """
    filtered_df = df_full.copy()
    
    # Apply basic filters
    if city:
        filtered_df = filtered_df[filtered_df['City'] == city]
    if cuisine:
        filtered_df = filtered_df[filtered_df['Cuisine_Style_List'].apply(
            lambda x: cuisine in x if x else False)]
    if price:
        price_map = {'Budget': '$', 'Moderate': '$$ - $$$', 'Luxury': '$$$$'}
        filtered_df = filtered_df[filtered_df['Price Range'] == price_map[price]]
    if min_rating:
        filtered_df = filtered_df[filtered_df['Rating'] >= float(min_rating)]
    
    return filtered_df.head(top_n)

@app.route('/')
def index():
    """Main page - show the web interface"""
    try:
        # Get cities and cuisines for dropdowns
        cities = sorted(df_full['City'].dropna().unique().tolist())
        
        all_cuisines_list = []
        for cuisine_list in df_full['Cuisine_Style_List'].dropna():
            if isinstance(cuisine_list, list):
                all_cuisines_list.extend(cuisine_list)
        
        all_cuisines = sorted(set(all_cuisines_list))
        
        print(f"Loaded {len(cities)} cities and {len(all_cuisines)} cuisines")
        
        return render_template('index.html', 
                             cities=cities, 
                             cuisines=all_cuisines,
                             total_restaurants=len(df_full))
    except Exception as e:
        print(f"Error loading page: {e}")
        return render_template('index.html', 
                             cities=[], 
                             cuisines=[],
                             total_restaurants=0)

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for restaurant recommendations"""
    data = request.get_json()
    algorithm = data.get('algorithm', 'content')
    restaurant_name = data.get('restaurant_name', '')
    city = data.get('city', '')
    cuisine = data.get('cuisine', '')
    price = data.get('price', '')
    min_rating = data.get('min_rating', '')
    
    try:
        # Algorithm 1: Content-based
        if algorithm == 'content':
            if not restaurant_name:
                return jsonify({'error': 'Please enter a restaurant name'})
            
            closest_name, results, error_msg = recommend_content_based(restaurant_name)
            if error_msg:
                return jsonify({'error': error_msg})
            if results is None or len(results) == 0:
                return jsonify({'error': f'No recommendations found for "{restaurant_name}"'})
            
            return jsonify({
                'algorithm': 'Content-Based Filtering',
                'closest_match': closest_name,
                'recommendations': results.to_dict('records')
            })
            
        # Algorithm 2: Popularity-based
        elif algorithm == 'popular':
            results = recommend_popular_based(city, cuisine, price, min_rating)
            return jsonify({
                'algorithm': 'Popularity-Based Filtering',
                'recommendations': results.to_dict('records')
            })
            
        # Algorithm 3: K-means clustering
        elif algorithm == 'kmeans':
            if not restaurant_name:
                return jsonify({'error': 'Please enter a restaurant name'})
            
            closest_name, results, error_msg = recommend_kmeans(restaurant_name)
            if error_msg:
                return jsonify({'error': error_msg})
            if results is None or len(results) == 0:
                return jsonify({'error': f'No recommendations found for "{restaurant_name}"'})
            
            input_cluster = df_full[df_full['Name'] == closest_name].iloc[0]['Cluster']
            return jsonify({
                'algorithm': 'K-Means Clustering',
                'closest_match': closest_name,
                'cluster': f'Cluster {input_cluster}',
                'recommendations': results.to_dict('records')
            })
            
        # Algorithm 4: Filter search
        elif algorithm == 'filter':
            results = filter_restaurants(city, cuisine, price, min_rating)
            return jsonify({
                'algorithm': 'Filter Search',
                'recommendations': results.to_dict('records')
            })
            
        else:
            return jsonify({'error': 'Invalid algorithm'})
            
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'})

# Start the web server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

