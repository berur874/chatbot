# ChatBot.py - LIGHTWEIGHT VERSION (No heavy NLP models)
import os
import re
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from praser import extract_preferences(msg)

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# ✅ CORS - Allow everything
CORS(app, resources={r"/*": {"origins": "*"}})

# Properties database
properties = [
    {"id": 1, "address": "123 Main St, Los Angeles, CA", "price": 450000, "bedrooms": 3, "bathrooms": 2, "sqft": 1800, "year_built": 2010, "neighborhood": "Downtown LA", "description": "Beautiful 3-bedroom home in the heart of LA"},
    {"id": 2, "address": "456 Oak Ave, San Francisco, CA", "price": 850000, "bedrooms": 2, "bathrooms": 1, "sqft": 1200, "year_built": 1995, "neighborhood": "Mission District", "description": "Charming 2-bedroom in trendy SF neighborhood"},
    {"id": 3, "address": "789 Pine Rd, San Diego, CA", "price": 620000, "bedrooms": 4, "bathrooms": 3, "sqft": 2200, "year_built": 2005, "neighborhood": "La Jolla", "description": "Spacious family home near the beach"},
    {"id": 4, "address": "321 Elm St, Sacramento, CA", "price": 380000, "bedrooms": 3, "bathrooms": 2, "sqft": 1600, "year_built": 1975, "neighborhood": "Midtown", "description": "Classic Sacramento home in walkable neighborhood"},
    {"id": 5, "address": "654 Cedar Ln, Fresno, CA", "price": 310000, "bedrooms": 3, "bathrooms": 2, "sqft": 1400, "year_built": 1980, "neighborhood": "Tower District", "description": "Affordable home in growing Fresno area"},
    {"id": 6, "address": "987 Beach Blvd, San Diego, CA", "price": 195000, "bedrooms": 2, "bathrooms": 1, "sqft": 900, "year_built": 1965, "neighborhood": "Ocean Beach", "description": "Cozy beach cottage near the ocean"},
    {"id": 7, "address": "147 Valley Rd, Fresno, CA", "price": 180000, "bedrooms": 2, "bathrooms": 1, "sqft": 850, "year_built": 1970, "neighborhood": "Fig Garden", "description": "Charming starter home in quiet neighborhood"}
]

# Global variables
rf_model = None
vectorizer = None
property_vectors = None
models_loaded = False

def load_models():
    """Load or train models - called on first request"""
    global rf_model, vectorizer, property_vectors, models_loaded
    
    if models_loaded:
        return True
    
    try:
        # Load/train ML model (SMALLER model for memory)
        logger.info("Loading ML model...")
        try:
            rf_model = joblib.load('house_price_model.pkl')
            logger.info("✅ ML model loaded from file")
        except:
            logger.info("Training new ML model...")
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            # Smaller model for memory
            rf_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
            rf_model.fit(X_train, y_train)
            joblib.dump(rf_model, 'house_price_model.pkl')
            logger.info("✅ ML model trained")
        
        # Create simple text search (NO heavy NLP models!)
        logger.info("Creating text search index...")
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        property_texts = [
            f"{p['neighborhood']} {p['description']} {p['address']}"
            for p in properties
        ]
        property_vectors = vectorizer.fit_transform(property_texts)
        logger.info("✅ Search index created")
        
        models_loaded = True
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

def predict_price(prefs):
    """Predict price using ML model"""
    if rf_model is None:
        return 400000
    
    budget = prefs.get('budget') or 500000
    bedrooms = prefs.get('bedrooms') or 3
    
    city_coords = {
        'los angeles': (34.05, -118.24),
        'san francisco': (37.77, -122.42),
        'san diego': (32.72, -117.16),
        'sacramento': (38.58, -121.49),
        'fresno': (36.75, -119.77)
    }
    lat, lon = city_coords.get(prefs.get('city'), (34.05, -118.24))
    
    features = [[
        budget / 100000.0,
        25.0,
        (bedrooms + 2) / 3.0,
        bedrooms / 3.0,
        1200.0,
        3.0,
        lat,
        lon
    ]]
    
    pred = rf_model.predict(features)[0] * 100000
    return float(np.round(pred, -3))

def simple_search(query, top_k=3):
    """Search properties using simple text matching"""
    if vectorizer is None or property_vectors is None:
        return []
    
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, property_vectors)[0]
    indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in indices:
        score = float(similarities[idx])
        if score > 0.01:  # Very low threshold for text match
            prop = properties[idx].copy()
            prop['similarity_score'] = round(score, 4)
            results.append(prop)
    return results

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'online', 'message': 'Chatbot API is running'})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    })

@app.route('/api/chat/semantic', methods=['POST', 'OPTIONS'])
def chat():
    # Handle preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    try:
        # Load models if needed
        if not models_loaded:
            success = load_models()
            if not success:
                return jsonify({'error': 'Models failed to load'}), 503
        
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message required'}), 400
        
        # Process
        prefs = extract_preferences(message)
        price = predict_price(prefs)
        matches = simple_search(message)
        
        # Format response
        response = f"🔍 Found {len(matches)} properties matching: '{message}'\n\n"
        if price:
            response += f"📊 Estimated value: ~${price:,.0f}\n\n"
        
        if matches:
            for i, p in enumerate(matches, 1):
                response += f"{i}. {p['address']}\n"
                response += f"   💰 ${p['price']:,} | 🛏️ {p['bedrooms']} bed | 📐 {p['sqft']} sqft\n"
                response += f"   📝 {p['description']}\n\n"
        else:
            response += "😅 No properties found matching your query."
        
        return jsonify({
            'response': response,
            'query': message,
            'preferences': prefs,
            'predicted_price': price,
            'properties': matches,
            'count': len(matches)
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)