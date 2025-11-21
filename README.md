# Restaurant Recommender System - BSP 3 Project

## 🎓 Project Overview
A sophisticated web-based restaurant recommendation system implementing three different algorithmic approaches with real-time sentiment analysis.

**Student:** Israel Micheal  
**Student ID:** 023111832D  
**University:** University of Luxembourg  
**Course:** BSP 3 Project - Big Data Analytics

## 🚀 Features

### Recommendation Algorithms
1. **Content-Based Filtering** - Finds similar restaurants using cosine similarity on features
2. **Popularity-Based Filtering** - Recommends highly-rated popular restaurants with weighted scoring
3. **K-Means Clustering** - Uses machine learning to group similar restaurants and recommends within clusters

### Advanced Features
- **Real-time Sentiment Analysis** of customer reviews using TextBlob
- **Fuzzy String Matching** for restaurant name search (handles typos automatically)
- **Advanced Filtering** by city, cuisine, price range, and minimum rating
- **Scientific Algorithm Comparison** with strengths/weaknesses analysis
- **Modern Responsive Design** works on desktop and mobile devices

## 🛠️ Technologies Used

### Backend
- **Python Flask** - Web framework
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms
- **TextBlob** - Natural language processing and sentiment analysis
- **fuzzywuzzy** - Fuzzy string matching

### Frontend
- **HTML5** - Page structure
- **CSS3** - Modern styling with gradients and animations
- **JavaScript** - Frontend logic and API communication
- **Responsive Design** - Mobile-friendly interface

### Deployment
- **Render.com** - Cloud hosting platform
- **gunicorn** - Production WSGI server

## 📁 Project Structure
restaurant_recommender_web/
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
├── render.yaml # Deployment configuration
├── data/
│ └── cleaned_dataset.csv # Restaurant dataset
├── templates/
│ └── index.html # Main web interface
├── static/
│ ├── css/
│ │ └── style.css # Styling and responsive design
│ └── js/
│ └── script.js # Frontend functionality
└── README.md # Project documentation



## Running Locally

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd restaurant_recommender_web