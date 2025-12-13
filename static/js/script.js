
/**
 * Restaurant Recommender - Frontend JavaScript
 * Handles user interactions and API calls
 */

// Track which algorithm is selected
let currentAlgorithm = 'content';

// Set up everything when page loads
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    updateRatingDisplay();
    console.log("Restaurant recommender ready");
});

// Set up all click and input handlers
function setupEventListeners() {
    // Algorithm buttons
    document.querySelectorAll('.algo-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            selectAlgorithm(this.dataset.algorithm);
        });
    });

    // Rating slider
    document.getElementById('ratingSlider').addEventListener('input', updateRatingDisplay);

    // Search button
    document.getElementById('searchBtn').addEventListener('click', searchRestaurants);

    // Enter key in search box
    document.getElementById('restaurantInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchRestaurants();
        }
    });
}

// Update rating display when slider moves
function updateRatingDisplay() {
    const slider = document.getElementById('ratingSlider');
    const valueDisplay = document.getElementById('ratingValue');
    valueDisplay.textContent = slider.value;
}

// Handle algorithm selection
function selectAlgorithm(algorithm) {
    currentAlgorithm = algorithm;
    console.log(`Selected algorithm: ${algorithm}`);
    
    // Update button styles
    document.querySelectorAll('.algo-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-algorithm="${algorithm}"]`).classList.add('active');
    
    // Update algorithm info
    updateAlgorithmInfo();
}

// Update the algorithm description
function updateAlgorithmInfo() {
    const infoDiv = document.getElementById('algorithmInfo');
    
    const algorithms = {
        'content': {
            name: 'Content-Based Filtering',
            description: 'Finds restaurants similar to your input. Enter a restaurant name below.',
            requirements: 'Needs a restaurant name'
        },
        'popular': {
            name: 'Popularity-Based Filtering',
            description: 'Shows popular restaurants based on ratings and reviews.',
            requirements: 'Works with or without filters'
        },
        'kmeans': {
            name: 'K-Means Clustering',
            description: 'Uses machine learning to group similar restaurants together.',
            requirements: 'Needs a restaurant name'
        },
        'filter': {
            name: 'Filter Search',
            description: 'Simple filtering based on your criteria.',
            requirements: 'Uses your selected filters'
        }
    };
    
    const algo = algorithms[currentAlgorithm];
    
    infoDiv.innerHTML = `
        <strong>${algo.name}</strong>
        <p>${algo.description}</p>
        <small>💡 ${algo.requirements}</small>
    `;
}

// Main search function
async function searchRestaurants() {
    const resultsContainer = document.getElementById('resultsContainer');
    const restaurantInput = document.getElementById('restaurantInput').value.trim();
    
    // Check if algorithm needs restaurant name
    const needsName = ['content', 'kmeans'];
    if (needsName.includes(currentAlgorithm) && !restaurantInput) {
        showError('Please enter a restaurant name for this algorithm');
        return;
    }
    
    // Show loading
    showLoading();
    
    try {
        // Send request to server
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
        showError('Could not connect to server. Please try again.');
        console.error('Search error:', error);
    }
}

// Show results
function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    
    let html = `
        <div class="results-header">
            <h3>${data.algorithm}</h3>
            ${data.closest_match ? `<p>Similar to: <strong>${data.closest_match}</strong></p>` : ''}
            ${data.cluster ? `<p>Group: <strong>${data.cluster}</strong></p>` : ''}
            <p>Found ${data.recommendations.length} restaurants</p>
        </div>
    `;
    
    if (data.recommendations.length === 0) {
        html += '<div class="error">No restaurants found. Try different filters.</div>';
    } else {
        html += '<div class="restaurant-grid">';
        
        data.recommendations.forEach(restaurant => {
            html += `
                <div class="restaurant-card">
                    <div class="restaurant-name">${restaurant.Name}</div>
                    <div class="restaurant-info">
                        <div>📍 ${restaurant.City || 'Unknown'}</div>
                        <div>⭐ ${restaurant.Rating}/5</div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    resultsContainer.innerHTML = html;
    console.log(`Showing ${data.recommendations.length} results`);
}

// Show loading spinner
function showLoading() {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = `
        <div class="loading">
            <div class="spinner">⏳</div>
            <p>Finding restaurants...</p>
        </div>
    `;
}

// Show error message
function showError(message) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = `
        <div class="error">
            <strong>❌ Error</strong>
            <p>${message}</p>
        </div>
    `;
}

// Initialize algorithm info
updateAlgorithmInfo();