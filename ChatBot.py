# ChatBot.py - LIGHTWEIGHT VERSION (No heavy NLP models)
import os
import re
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from praser import extract_preferences
from properties import properties
from ml_model import ml_model
from search import simple_search

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# ✅ CORS - Allow everything
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
vectorizer = None
property_vectors = None
models_loaded = False

def load_models():
    """Load or train models - called on first request"""
    global vectorizer, property_vectors, models_loaded
    
    if models_loaded:
        return True
    
    try:
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
        logger.error(f"Search index creation failed: {e}")
        return False
    

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
        
        # Build features for ML model
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
        
        features = [
            budget / 100000.0,
            25.0,
            (bedrooms + 2) / 3.0,
            bedrooms / 3.0,
            1200.0,
            3.0,
            lat,
            lon
        ]
        
        # Get price prediction from ml_model
        prediction = ml_model.predict(features)
        price = float(np.round(prediction * 100000, -3))
        
        matches = simple_search(message, vectorizer, property_vectors)
        
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