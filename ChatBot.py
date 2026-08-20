# ChatBot.py - LIGHTWEIGHT VERSION
import os
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer

# Import data & search utility
from properties import get_all_properties
from ml_model import ml_model
from search import search_properties

# Optional: Try importing parser module safely
try:
    from parser import extract_preferences
except ImportError:
    try:
        from praser import extract_preferences
    except ImportError:
        def extract_preferences(text):
            return {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global TF-IDF index states
vectorizer = None
property_vectors = None
properties_cache = []
models_loaded = False


def load_models():
    """Fetch properties from Supabase and build TF-IDF search index."""
    global vectorizer, property_vectors, properties_cache, models_loaded
    
    try:
        logger.info("Fetching properties from Supabase and indexing...")
        properties_cache = get_all_properties()
        
        if not properties_cache:
            logger.warning("No properties retrieved from database.")
            return False

        property_texts = [
            f"{p.get('neighborhood', '')} {p.get('description', '')} {p.get('address', '')}"
            for p in properties_cache
        ]
        
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        property_vectors = vectorizer.fit_transform(property_texts)
        
        logger.info(f"✅ Search index built with {len(properties_cache)} properties")
        models_loaded = True
        return True
    except Exception as e:
        logger.error(f"Failed to load search index: {e}")
        return False


@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'online', 'message': 'NyumbaHub Chatbot API is running'})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'property_count': len(properties_cache)
    })


@app.route('/api/chat/semantic', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    try:
        if not models_loaded or not properties_cache:
            if not load_models():
                return jsonify({'error': 'Failed to initialize search engine or database'}), 503
        
        data = request.json or {}
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message required'}), 400
        
        # Parse preferences
        prefs = extract_preferences(message)
        
        budget = prefs.get('budget') or 15000000
        bedrooms = prefs.get('bedrooms') or 3
        city = (prefs.get('city') or '').lower()

        # Kenyan Location Coordinates Map
        city_coords = {
            'nairobi': (-1.286389, 36.817223),
            'mombasa': (-4.043477, 39.668206),
            'kisumu': (-0.091702, 34.767956),
            'nakuru': (-0.303099, 36.080025),
            'eldoret': (0.514277, 35.269779),
            'kiambu': (-1.1714, 36.8356),
            'naivasha': (-0.7167, 36.4333)
        }
        lat, lon = city_coords.get(city, (-1.286389, 36.817223))

        # ML Features Setup
        features = [
            budget / 1000000.0,
            25.0,
            (bedrooms + 2) / 3.0,
            bedrooms / 3.0,
            1200.0,
            3.0,
            lat,
            lon
        ]
        
        # Predict price estimate
        try:
            prediction = ml_model.predict(features)
            price = float(np.round(prediction * 1000000, -3))
        except Exception as ml_err:
            logger.warning(f"ML Model prediction skipped: {ml_err}")
            price = None

        # Execute Search
        matches = search_properties(
            query=message, 
            top_k=3
        )

        # Build Response
        response = f"🔍 Found {len(matches)} properties matching: '{message}'\n\n"
        if price:
            response += f"📊 Estimated Value: ~KES {price:,.0f}\n\n"
        
        if matches:
            for i, p in enumerate(matches, 1):
                response += f"{i}. {p.get('address', 'N/A')}\n"
                response += f"   💰 KES {p.get('price', 0):,} | 🛏️ {p.get('bedrooms', 0)} bed | 📐 {p.get('sqft', 0)} sqft\n"
                response += f"   📝 {p.get('description', '')}\n\n"
        else:
            response += "😅 No matching properties found for your query."

        return jsonify({
            'response': response,
            'query': message,
            'preferences': prefs,
            'predicted_price': price,
            'properties': matches,
            'count': len(matches)
        })

    except Exception as e:
        logger.error(f"Chat execution error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)