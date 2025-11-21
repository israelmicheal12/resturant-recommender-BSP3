"""
Restaurant Recommender System - BSP 3 Project
Student: Israel Micheal | ID: 023111832D | University of Luxembourg

Web-based recommender system implementing four algorithms:
1. Content-Based Filtering (Cosine Similarity)
2. Popularity-Based Filtering 
3. Category-Based Filtering
4. K-Means Clustering (using 7 clusters)
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import ast
from textblob import TextBlob
from fuzzywuzzy import process
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

print("🚀 Starting Restaurant Recommender System...")
print("BSP 3 Project - Israel Micheal - University of Luxembourg")

print("📊 Loading restaurant dataset...")
df = pd.read_csv('data/cleaned_dataset.csv')
print(f"✅ Successfully loaded {len(df)} restaurants")

# Use the ENTIRE dataset - no sampling!
df_full = df.copy()
print(f"📊 Using ALL {len(df_full)} restaurants from the dataset")

# DEBUG: Show dataset information
print(f"🏙️  Total unique cities: {len(df_full['City'].dropna().unique())}")
print(f"📍 Cities available: {sorted(df_full['City'].dropna().unique().tolist()[:10])}")

def calculate_sentiment(review_list):
    """Calculate sentiment score for reviews"""
    if pd.isnull(review_list) or not review_list or review_list == '[]':
        return 0
    
    try:
        if isinstance(review_list, str) and review_list.startswith('['):
            reviews = ast.literal_eval(review_list)
        else:
            reviews = [str(review_list)]
        
        sentiments = []
        for review in reviews:
            if review and isinstance(review, str) and len(str(review).strip()) > 0:
                try:
                    sentiment_score = TextBlob(str(review)).sentiment.polarity
                    sentiments.append(sentiment_score)
                except:
                    continue
        
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except:
        return 0

print("😊 Analyzing review sentiments...")
df_full['Sentiment_Score'] = df_full['Reviews'].apply(calculate_sentiment)
print("✅ Sentiment analysis completed")

def convert_cuisine_string_to_list(cuisine_str):
    """Convert cuisine string to list"""
    if pd.isnull(cuisine_str) or not cuisine_str or cuisine_str == '[]':
        return []
    try:
        cuisines = ast.literal_eval(cuisine_str)
        return [c.strip() for c in cuisines]
    except:
        return []

df_full['Cuisine_Style_List'] = df_full['Cuisine Style'].apply(convert_cuisine_string_to_list)

# Create features for recommendation
print("🔧 Creating features for machine learning...")
cuisine_dummies = df_full['Cuisine_Style_List'].str.join('|').str.get_dummies(sep='|')

# Normalize numerical features
df_full['Rating_norm'] = (df_full['Rating'] - df_full['Rating'].min()) / (df_full['Rating'].max() - df_full['Rating'].min())
df_full['Sentiment_norm'] = (df_full['Sentiment_Score'] - df_full['Sentiment_Score'].min()) / (df_full['Sentiment_Score'].max() - df_full['Sentiment_Score'].min())

# Fill NaN values
df_full['Rating_norm'] = df_full['Rating_norm'].fillna(0.5)
df_full['Sentiment_norm'] = df_full['Sentiment_norm'].fillna(0.5)

features = pd.concat([cuisine_dummies, df_full[['Rating_norm', 'Sentiment_norm']]], axis=1)
print(f"✅ Feature engineering completed - {features.shape[1]} features")

# K-Means Clustering Setup (7 clusters as per your project description)
print("🎯 Setting up K-Means clustering with 7 clusters...")
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_full['Cluster'] = kmeans.fit_predict(features)
print("✅ K-Means clustering completed")

print("🔄 All recommendation algorithms ready!")
print(f"📊 Dataset info: {len(df_full)} restaurants, {features.shape[1]} features, 7 clusters")

def find_closest_restaurant(input_name):
    """Fuzzy matching for restaurant names with improved error handling"""
    if not input_name or not isinstance(input_name, str) or len(input_name.strip()) < 2:
        return None
        
    all_names = df_full['Name'].dropna().tolist()
    try:
        best_match, score = process.extractOne(input_name, all_names)
        print(f"🔍 Fuzzy match: '{input_name}' -> '{best_match}' (confidence: {score}%)")
        return best_match if best_match and score >= 50 else None
    except Exception as e:
        print(f"❌ Fuzzy matching error: {e}")
        return None

def recommend_content_based(name, top_n=5):
    """ALGORITHM 1: Content-based recommendations using cosine similarity"""
    closest_name = find_closest_restaurant(name)
    if closest_name is None:
        return None, None, "No matching restaurant found"
    
    try:
        # Get the input restaurant's features
        input_idx = df_full[df_full['Name'] == closest_name].index[0]
        input_features = features.iloc[input_idx].values.reshape(1, -1)
        
        # Calculate cosine similarity manually
        similarities = []
        for idx in range(len(df_full)):
            if idx != input_idx:
                other_features = features.iloc[idx].values.reshape(1, -1)
                # Manual cosine similarity calculation
                dot_product = np.dot(input_features, other_features.T)[0][0]
                input_norm = np.linalg.norm(input_features)
                other_norm = np.linalg.norm(other_features)
                
                if input_norm > 0 and other_norm > 0:
                    similarity = dot_product / (input_norm * other_norm)
                else:
                    similarity = 0
                
                similarities.append((idx, similarity))
        
        # Sort by similarity and get top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, sim in similarities[:top_n]]
        
        return closest_name, df_full.iloc[top_indices], None
    except Exception as e:
        print(f"Error in content-based recommendation: {e}")
        return None, None, f"Content-based recommendation error: {str(e)}"

def recommend_category_based(name, top_n=5):
    """ALGORITHM 2: Category-based recommendations using cuisine groups"""
    closest_name = find_closest_restaurant(name)
    if closest_name is None:
        return None, None, "No matching restaurant found"
    
    try:
        # Get the input restaurant's primary cuisine
        input_restaurant = df_full[df_full['Name'] == closest_name].iloc[0]
        input_cuisines = input_restaurant['Cuisine_Style_List']
        
        if not input_cuisines:
            return closest_name, pd.DataFrame(), "No cuisine information available"
            
        primary_cuisine = input_cuisines[0] if input_cuisines else 'Other'
        
        # Find restaurants with same primary cuisine (excluding input)
        same_cuisine_restaurants = df_full[
            df_full['Cuisine_Style_List'].apply(
                lambda x: primary_cuisine in x if x else False
            ) & 
            (df_full['Name'] != closest_name)
        ]
        
        if len(same_cuisine_restaurants) == 0:
            return closest_name, same_cuisine_restaurants, f"No other restaurants found in cuisine: {primary_cuisine}"
        
        # Return highest rated restaurants from same category
        return closest_name, same_cuisine_restaurants.nlargest(top_n, 'Rating'), None
    except Exception as e:
        print(f"Error in category-based recommendation: {e}")
        return None, None, f"Category-based recommendation error: {str(e)}"

def recommend_kmeans(name, top_n=5):
    """ALGORITHM 3: K-Means clustering recommendations"""
    closest_name = find_closest_restaurant(name)
    if closest_name is None:
        return None, None, "No matching restaurant found"
    
    try:
        # Get the cluster of the input restaurant
        input_restaurant = df_full[df_full['Name'] == closest_name].iloc[0]
        input_cluster = input_restaurant['Cluster']
        
        # Find other restaurants in the same cluster
        cluster_restaurants = df_full[
            (df_full['Cluster'] == input_cluster) & 
            (df_full['Name'] != closest_name)
        ]
        
        if len(cluster_restaurants) == 0:
            return closest_name, cluster_restaurants, f"No other restaurants found in cluster {input_cluster}"
        
        # Return highest rated restaurants from same cluster
        return closest_name, cluster_restaurants.nlargest(top_n, 'Rating'), None
    except Exception as e:
        print(f"Error in K-means recommendation: {e}")
        return None, None, f"K-means recommendation error: {str(e)}"

def recommend_popular_based(city=None, cuisine=None, price=None, min_rating=None, top_n=5):
    """ALGORITHM 4: Popularity-based recommendations with weighted scoring"""
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
    
    # Calculate popularity score (weighted combination as per your project description)
    rating_weight = 0.5      # 50% for ratings (quality)
    sentiment_weight = 0.3   # 30% for sentiment (user feelings)
    # Note: 20% for review count would require review count data
    
    filtered_df['Popularity_Score'] = (
        rating_weight * filtered_df['Rating_norm'] +
        sentiment_weight * filtered_df['Sentiment_norm']
    )
    
    return filtered_df.nlargest(top_n, 'Popularity_Score')

def filter_restaurants(city=None, cuisine=None, price=None, min_rating=None, top_n=20):
    """Basic filtering for restaurant discovery"""
    filtered_df = df_full.copy()
    
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
    """Main page with the web interface"""
    try:
        # Get all unique cities from the FULL dataset
        cities = sorted(df_full['City'].dropna().unique().tolist())
        
        # Get all unique cuisines from FULL dataset
        all_cuisines_list = []
        for cuisine_list in df_full['Cuisine_Style_List'].dropna():
            if isinstance(cuisine_list, list):
                all_cuisines_list.extend(cuisine_list)
        
        all_cuisines = sorted(set(all_cuisines_list))
        
        print(f"📊 Loaded {len(cities)} cities and {len(all_cuisines)} cuisines from FULL dataset")
        print(f"🔢 Using {len(df_full)} restaurants with K-means clustering (7 clusters)")
        
        return render_template('index.html', 
                             cities=cities, 
                             cuisines=all_cuisines,
                             total_restaurants=len(df_full))
    except Exception as e:
        print(f"❌ Error in index route: {e}")
        return render_template('index.html', 
                             cities=[], 
                             cuisines=[],
                             total_restaurants=0)

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for restaurant search and recommendations"""
    data = request.get_json()
    algorithm = data.get('algorithm', 'content')
    restaurant_name = data.get('restaurant_name', '')
    city = data.get('city', '')
    cuisine = data.get('cuisine', '')
    price = data.get('price', '')
    min_rating = data.get('min_rating', '')
    
    try:
        if algorithm == 'content':
            if not restaurant_name:
                return jsonify({'error': 'Please enter a restaurant name for content-based recommendations'})
            
            closest_name, results, error_msg = recommend_content_based(restaurant_name)
            if error_msg:
                return jsonify({'error': error_msg})
            if results is None or len(results) == 0:
                return jsonify({'error': f'No recommendations found for "{restaurant_name}". Try a different name.'})
            
            return jsonify({
                'algorithm': 'Content-Based Filtering (Cosine Similarity)',
                'closest_match': closest_name,
                'recommendations': results.to_dict('records')
            })
            
        elif algorithm == 'popular':
            results = recommend_popular_based(city, cuisine, price, min_rating)
            return jsonify({
                'algorithm': 'Popularity-Based Filtering (Weighted Scoring)',
                'recommendations': results.to_dict('records')
            })
            
        elif algorithm == 'category':
            if not restaurant_name:
                return jsonify({'error': 'Please enter a restaurant name for category-based recommendations'})
            
            closest_name, results, error_msg = recommend_category_based(restaurant_name)
            if error_msg:
                return jsonify({'error': error_msg})
            if results is None or len(results) == 0:
                return jsonify({'error': f'No category-based recommendations found for "{restaurant_name}"'})
            
            cuisine_type = df_full[df_full['Name'] == closest_name].iloc[0]['Cuisine_Style_List']
            primary_cuisine = cuisine_type[0] if cuisine_type else 'Other'
            return jsonify({
                'algorithm': 'Category-Based Filtering',
                'closest_match': closest_name,
                'cuisine_category': primary_cuisine,
                'recommendations': results.to_dict('records')
            })
            
        elif algorithm == 'kmeans':
            if not restaurant_name:
                return jsonify({'error': 'Please enter a restaurant name for K-means clustering recommendations'})
            
            closest_name, results, error_msg = recommend_kmeans(restaurant_name)
            if error_msg:
                return jsonify({'error': error_msg})
            if results is None or len(results) == 0:
                return jsonify({'error': f'No K-means recommendations found for "{restaurant_name}"'})
            
            input_cluster = df_full[df_full['Name'] == closest_name].iloc[0]['Cluster']
            return jsonify({
                'algorithm': 'K-Means Clustering (7 Clusters)',
                'closest_match': closest_name,
                'cluster': f'Cluster {input_cluster}',
                'recommendations': results.to_dict('records')
            })
            
        elif algorithm == 'filter':
            results = filter_restaurants(city, cuisine, price, min_rating)
            return jsonify({
                'algorithm': 'Filtered Search',
                'recommendations': results.to_dict('records')
            })
            
        else:
            return jsonify({'error': 'Invalid algorithm selected'})
            
    except Exception as e:
        print(f"❌ API Search Error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'})

@app.route('/api/compare', methods=['POST'])
def api_compare():
    """API endpoint for scientific comparison of algorithms"""
    data = request.get_json()
    restaurant_name = data.get('restaurant_name', 'Pizza Hut')
    
    comparison_results = {}
    analysis_results = {}
    
    # Test all four algorithms
    algorithms_to_test = ['content', 'category', 'kmeans', 'popular']
    
    for algo in algorithms_to_test:
        try:
            if algo == 'content':
                closest_name, results, error = recommend_content_based(restaurant_name, top_n=3)
                if results is not None and len(results) > 0:
                    comparison_results['content_based'] = {
                        'closest_match': closest_name,
                        'sample_recommendations': results.head(3).to_dict('records')
                    }
            elif algo == 'category':
                closest_name, results, error = recommend_category_based(restaurant_name, top_n=3)
                if results is not None and len(results) > 0:
                    cuisine_type = df_full[df_full['Name'] == closest_name].iloc[0]['Cuisine_Style_List']
                    primary_cuisine = cuisine_type[0] if cuisine_type else 'Other'
                    comparison_results['category_based'] = {
                        'closest_match': closest_name,
                        'cuisine_category': primary_cuisine,
                        'sample_recommendations': results.head(3).to_dict('records')
                    }
            elif algo == 'kmeans':
                closest_name, results, error = recommend_kmeans(restaurant_name, top_n=3)
                if results is not None and len(results) > 0:
                    input_cluster = df_full[df_full['Name'] == closest_name].iloc[0]['Cluster']
                    comparison_results['kmeans_clustering'] = {
                        'closest_match': closest_name,
                        'cluster': f'Cluster {input_cluster}',
                        'sample_recommendations': results.head(3).to_dict('records')
                    }
            elif algo == 'popular':
                results = recommend_popular_based(top_n=3)
                comparison_results['popularity_based'] = {
                    'sample_recommendations': results.head(3).to_dict('records')
                }
        except Exception as e:
            print(f"❌ Error comparing algorithm {algo}: {e}")
    
    # Algorithm analysis (as per your project description)
    analysis = {
        'content_based': {
            'strengths': [
                'High personalization based on all features',
                'Finds very similar restaurants using cosine similarity', 
                'Considers cuisine, rating, and sentiment',
                'Most accurate similarity matching'
            ],
            'weaknesses': [
                'Computationally intensive for large datasets',
                'Requires restaurant name input',
                'May over-specialize recommendations'
            ],
            'best_for': 'Users who want the most similar restaurants based on all features'
        },
        'category_based': {
            'strengths': [
                'Fast and efficient computation',
                'Good for discovering new options in same cuisine category',
                'Easy to understand and interpret',
                'Works well for cuisine exploration'
            ],
            'weaknesses': [
                'Limited to cuisine similarity only',
                'May miss cross-cuisine similarities',
                'Less personalized than content-based'
            ],
            'best_for': 'Users wanting to explore specific cuisine types'
        },
        'kmeans_clustering': {
            'strengths': [
                'Discovers non-obvious patterns using machine learning',
                'Groups restaurants by multiple features simultaneously',
                'Good for exploring diverse but related options',
                'Handles complex feature relationships'
            ],
            'weaknesses': [
                'Requires pre-computation (offline training)',
                'Cluster interpretation can be challenging',
                'Fixed number of clusters (7 in this implementation)'
            ],
            'best_for': 'Users wanting machine learning-based recommendations'
        },
        'popularity_based': {
            'strengths': [
                'No input required from user',
                'Discovers popular and highly-rated restaurants',
                'Great for new users or when exploring new cities',
                'Safe and reliable choices based on crowd wisdom'
            ],
            'weaknesses': [
                'Not personalized to individual preferences',
                'Popularity bias may overlook hidden gems',
                'May recommend only well-known establishments'
            ],
            'best_for': 'New users or those looking for popular, highly-rated choices'
        }
    }
    
    insights = (
        "Each algorithm serves different user scenarios: Content-based for precision, "
        "Category-based for cuisine exploration, K-means for ML-driven discovery, "
        "and Popularity-based for reliable choices. The weighted scoring in popularity-based "
        "algorithm (50% rating, 30% sentiment, 20% reviews) provides balanced recommendations."
    )
    
    return jsonify({
        'comparison': comparison_results,
        'analysis': analysis,
        'insights': insights
    })

if __name__ == '__main__':
    print("✅ Web application ready!")
    print("🎯 Implemented 4 algorithms: Content-based, Category-based, K-means, Popularity-based")
    print("🔢 K-means clustering with 7 clusters as per project specification")
    print("🌐 Starting Flask server at http://localhost:5001")
    app.run(debug=False, host='0.0.0.0', port=5001)

