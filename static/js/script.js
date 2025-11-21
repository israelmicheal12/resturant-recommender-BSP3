
// Restaurant Recommender System - Frontend JavaScript
// BSP 3 Project - Israel Micheal - University of Luxembourg

// Global variable to track current selected algorithm
let currentAlgorithm = 'content'; // Default to content-based filtering

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateRatingDisplay();
    console.log("✅ Restaurant Recommender System initialized");
    console.log("🎓 BSP 3 Project - Israel Micheal - University of Luxembourg");
});

function initializeEventListeners() {
    // Algorithm selection buttons
    document.querySelectorAll('.algo-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            selectAlgorithm(this.dataset.algorithm);
        });
    });

    // Rating slider
    document.getElementById('ratingSlider').addEventListener('input', updateRatingDisplay);

    // Search button
    document.getElementById('searchBtn').addEventListener('click', performSearch);

    // Compare button
    document.getElementById('compareBtn').addEventListener('click', performComparison);

    // Enter key support in search input
    document.getElementById('restaurantInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
}

function updateRatingDisplay() {
    const slider = document.getElementById('ratingSlider');
    const valueDisplay = document.getElementById('ratingValue');
    valueDisplay.textContent = slider.value;
    console.log(`⭐ Rating filter updated to: ${slider.value}`);
}

function selectAlgorithm(algorithm) {
    currentAlgorithm = algorithm;
    console.log(`🎯 Algorithm selected: ${algorithm}`);
    
    // Update visual state of algorithm buttons
    document.querySelectorAll('.algo-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-algorithm="${algorithm}"]`).classList.add('active');
    
    // Update algorithm information display
    updateAlgorithmInfo();
}

function updateAlgorithmInfo() {
    const infoDiv = document.getElementById('algorithmInfo');
    
    const algorithms = {
        'content': {
            name: 'Content-Based Filtering',
            description: 'Finds restaurants similar to your input using cosine similarity and feature matching. Analyzes cuisine types, ratings, and review sentiments.',
            requirements: 'Requires a restaurant name as input.'
        },
        'popular': {
            name: 'Popularity-Based Filtering',
            description: 'Recommends highly-rated popular restaurants based on weighted scoring (50% rating, 30% sentiment, 20% reviews). Great for discovering reliable choices.',
            requirements: 'Works with or without filters. No restaurant name required.'
        },
        'category': {
            name: 'Category-Based Filtering',
            description: 'Groups restaurants by cuisine categories and recommends from the same culinary group. Fast and efficient for cuisine exploration.',
            requirements: 'Requires a restaurant name as input.'
        },
        'kmeans': {
            name: 'K-Means Clustering',
            description: 'Uses machine learning with 7 clusters to group similar restaurants. Recommends from the same cluster for non-obvious similarities.',
            requirements: 'Requires a restaurant name as input. Uses pre-computed clusters.'
        },
        'filter': {
            name: 'Filter Search',
            description: 'Basic filtering of restaurants based on your criteria. Simple search without complex algorithms.',
            requirements: 'Uses your selected filters for straightforward discovery.'
        }
    };
    
    const algo = algorithms[currentAlgorithm];
    
    infoDiv.innerHTML = `
        <strong>${algo.name}</strong>
        <p>${algo.description}</p>
        <small>💡 ${algo.requirements}</small>
    `;
}

async function performSearch() {
    const resultsContainer = document.getElementById('resultsContainer');
    const restaurantInput = document.getElementById('restaurantInput').value.trim();
    
    // Validate input for algorithms that require restaurant name
    const algorithmsRequiringName = ['content', 'category', 'kmeans'];
    if (algorithmsRequiringName.includes(currentAlgorithm) && !restaurantInput) {
        showError('Please enter a restaurant name for this algorithm');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                algorithm: currentAlgorithm,
                restaurant_name: restaurantInput,
                city: document.getElementById('citySelect').value,
                cuisine: document.getElementById('cuisineSelect').value,
                price: document.getElementById('priceSelect').value,
                min_rating: document.getElementById('ratingSlider').value
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        displayResults(data);
        
    } catch (error) {
        showError('Network error: Could not connect to server. Please check your connection.');
        console.error('Search error:', error);
    }
}

async function performComparison() {
    const comparisonResults = document.getElementById('comparisonResults');
    const restaurantInput = document.getElementById('restaurantInput').value.trim() || 'Pizza Hut';
    
    showLoading('comparisonResults');
    
    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                restaurant_name: restaurantInput
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error, 'comparisonResults');
            return;
        }
        
        displayComparison(data);
        
    } catch (error) {
        showError('Network error: Could not connect to server.', 'comparisonResults');
        console.error('Comparison error:', error);
    }
}

function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    
    let html = `
        <div class="results-header">
            <h3>${data.algorithm}</h3>
            ${data.closest_match ? `<p>Showing recommendations similar to: <strong>${data.closest_match}</strong></p>` : ''}
            ${data.cuisine_category ? `<p>Cuisine Category: <strong>${data.cuisine_category}</strong></p>` : ''}
            ${data.cluster ? `<p>ML Cluster: <strong>${data.cluster}</strong></p>` : ''}
            <p>Found ${data.recommendations.length} restaurants</p>
        </div>
    `;
    
    if (data.recommendations.length === 0) {
        html += '<div class="error">No restaurants found with the current criteria. Try adjusting your filters or using a different restaurant name.</div>';
    } else {
        html += '<div class="restaurant-grid">';
        
        data.recommendations.forEach(restaurant => {
            const sentiment = getSentimentDisplay(restaurant.Sentiment_Score);
            const popularityScore = restaurant.Popularity_Score ? restaurant.Popularity_Score.toFixed(3) : 'N/A';
            
            html += `
                <div class="restaurant-card">
                    <div class="restaurant-name">${restaurant.Name}</div>
                    <div class="restaurant-info">
                        <div>📍 ${restaurant.City || 'Unknown'}</div>
                        <div>⭐ ${restaurant.Rating}/5</div>
                        <div>💰 ${restaurant.Price_Range || 'Not specified'}</div>
                        <div>🍽️ ${restaurant.Cuisine_Style || 'Not specified'}</div>
                        <div class="${sentiment.class}">😊 Sentiment: ${sentiment.text}</div>
                        ${restaurant.Popularity_Score ? `<div>🔥 Popularity Score: ${popularityScore}</div>` : ''}
                        ${restaurant.Cluster !== undefined ? `<div>🎯 Cluster: ${restaurant.Cluster}</div>` : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    resultsContainer.innerHTML = html;
    console.log(`✅ Displayed ${data.recommendations.length} recommendations using ${data.algorithm}`);
}

function displayComparison(data) {
    const comparisonResults = document.getElementById('comparisonResults');
    
    let html = '<h3>🔬 Scientific Comparison of Four Recommendation Approaches</h3>';
    
    // Algorithm comparison cards
    html += '<div class="algorithm-comparison">';
    
    for (const [algoKey, algoData] of Object.entries(data.comparison)) {
        const algoName = getAlgorithmName(algoKey);
        
        html += `
            <div class="comparison-card">
                <h4>${algoName}</h4>
                ${algoData.closest_match ? `<p><strong>Input:</strong> ${algoData.closest_match}</p>` : ''}
                ${algoData.cuisine_category ? `<p><strong>Cuisine:</strong> ${algoData.cuisine_category}</p>` : ''}
                ${algoData.cluster ? `<p><strong>Cluster:</strong> ${algoData.cluster}</p>` : ''}
                <p><strong>Sample Recommendations:</strong></p>
                <ul>
                    ${algoData.sample_recommendations.slice(0, 3).map(r => 
                        `<li>${r.Name} (⭐ ${r.Rating}/5${r.City ? ` - 📍 ${r.City}` : ''})</li>`
                    ).join('')}
                </ul>
            </div>
        `;
    }
    
    html += '</div>';
    
    // Strengths and weaknesses analysis
    html += '<div class="analysis-section">';
    html += '<h4>📊 Algorithm Analysis</h4>';
    
    for (const [algoKey, analysis] of Object.entries(data.analysis)) {
        const algoName = getAlgorithmName(algoKey);
        
        html += `
            <div class="analysis-card">
                <h5>${algoName}</h5>
                <div class="strengths">
                    <strong>✅ Strengths:</strong>
                    <ul>
                        ${analysis.strengths.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>
                <div class="weaknesses">
                    <strong>❌ Weaknesses:</strong>
                    <ul>
                        ${analysis.weaknesses.map(w => `<li>${w}</li>`).join('')}
                    </ul>
                </div>
                <div class="best-for">
                    <strong>🎯 Best For:</strong> ${analysis.best_for}
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    
    // Scientific insights
    html += `
        <div class="insights">
            <h4>💡 Scientific Insights</h4>
            <p>${data.insights}</p>
            <p><strong>Key Finding:</strong> Each algorithm serves different user scenarios - no single approach dominates all use cases.</p>
            <p><strong>Project Implementation:</strong> Successfully implemented 4 algorithms with K-means clustering using 7 clusters as specified in the BSP 3 project requirements.</p>
        </div>
    `;
    
    comparisonResults.innerHTML = html;
    console.log("✅ Displayed comprehensive algorithm comparison");
}

function getAlgorithmName(key) {
    const names = {
        'content_based': 'Content-Based Filtering',
        'popularity_based': 'Popularity-Based Filtering',
        'category_based': 'Category-Based Filtering',
        'kmeans_clustering': 'K-Means Clustering'
    };
    return names[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function getSentimentDisplay(score) {
    if (score > 0.1) return { class: 'sentiment-positive', text: `Positive (${score.toFixed(2)})` };
    if (score < -0.1) return { class: 'sentiment-negative', text: `Negative (${score.toFixed(2)})` };
    return { class: 'sentiment-neutral', text: `Neutral (${score.toFixed(2)})` };
}

function showLoading(containerId = 'resultsContainer') {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="loading">
            <div class="spinner">⏳</div>
            <p>Processing your request...</p>
            <p><small>Analyzing restaurants and generating recommendations</small></p>
        </div>
    `;
}

function showError(message, containerId = 'resultsContainer') {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="error">
            <strong>❌ Recommendation Error</strong>
            <p>${message}</p>
            <small>Try checking your input, adjusting filters, or using a different restaurant name.</small>
        </div>
    `;
    console.error(`❌ Error: ${message}`);
}

// Initialize algorithm info when page loads
updateAlgorithmInfo();
