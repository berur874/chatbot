# ml_chatbot_api.py - WITH CORS FIX

import os
import logging
import numpy as np
import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Make sure this is imported
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ CORS CONFIGURATION - FIX
CORS(app, resources={r"/*": {"origins": "*"}})

# For testing, you can also use:
# CORS(app, resources={r"/api/*": {"origins": "*"}})

print("=" * 50)
print("🤖 ML CHATBOT API STARTING...")
print("=" * 50)
print("📡 CORS allowed origins:")
print("   - http://localhost:5173")
print("   - http://127.0.0.1:5173")
print("   - http://localhost:3000")
print("   - https://nyumbahub.vercel.app")
print("=" * 50)

# ==========================================
# GLOBAL VARIABLES
# ==========================================

rf_model = None
embedder = None
property_embeddings = None
models_loaded = False
model_load_error = None

# ==========================================
# PROPERTIES DATABASE
# ==========================================

properties = [
    {
        "id": 1,
        "address": "123 Main St, Los Angeles, CA",
        "price": 450000,
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft": 1800,
        "year_built": 2010,
        "lat": 34.05,
        "lon": -118.24,
        "neighborhood": "Downtown LA",
        "description": "Beautiful 3-bedroom home in the heart of LA",
        "school_rating": 8,
        "walk_score": 85
    },
    {
        "id": 2,
        "address": "456 Oak Ave, San Francisco, CA",
        "price": 850000,
        "bedrooms": 2,
        "bathrooms": 1,
        "sqft": 1200,
        "year_built": 1995,
        "lat": 37.77,
        "lon": -122.42,
        "neighborhood": "Mission District",
        "description": "Charming 2-bedroom in trendy SF neighborhood",
        "school_rating": 9,
        "walk_score": 92
    },
    {
        "id": 3,
        "address": "789 Pine Rd, San Diego, CA",
        "price": 620000,
        "bedrooms": 4,
        "bathrooms": 3,
        "sqft": 2200,
        "year_built": 2005,
        "lat": 32.72,
        "lon": -117.16,
        "neighborhood": "La Jolla",
        "description": "Spacious family home near the beach",
        "school_rating": 10,
        "walk_score": 70
    },
    {
        "id": 4,
        "address": "321 Elm St, Sacramento, CA",
        "price": 380000,
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft": 1600,
        "year_built": 1975,
        "lat": 38.58,
        "lon": -121.49,
        "neighborhood": "Midtown",
        "description": "Classic Sacramento home in walkable neighborhood",
        "school_rating": 6,
        "walk_score": 88
    },
    {
        "id": 5,
        "address": "654 Cedar Ln, Fresno, CA",
        "price": 310000,
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft": 1400,
        "year_built": 1980,
        "lat": 36.75,
        "lon": -119.77,
        "neighborhood": "Tower District",
        "description": "Affordable home in growing Fresno area",
        "school_rating": 5,
        "walk_score": 65
    },
    {
        "id": 6,
        "address": "987 Beach Blvd, San Diego, CA",
        "price": 195000,
        "bedrooms": 2,
        "bathrooms": 1,
        "sqft": 900,
        "year_built": 1965,
        "lat": 32.71,
        "lon": -117.15,
        "neighborhood": "Ocean Beach",
        "description": "Cozy beach cottage near the ocean",
        "school_rating": 7,
        "walk_score": 90
    },
    {
        "id": 7,
        "address": "147 Valley Rd, Fresno, CA",
        "price": 180000,
        "bedrooms": 2,
        "bathrooms": 1,
        "sqft": 850,
        "year_built": 1970,
        "lat": 36.74,
        "lon": -119.76,
        "neighborhood": "Fig Garden",
        "description": "Charming starter home in quiet neighborhood",
        "school_rating": 6,
        "walk_score": 60
    }
]

# ==========================================
# MODEL LOADING FUNCTIONS
# ==========================================

def load_ml_model():
    """Load ML model - called on first request"""
    global rf_model, model_load_error
    
    if rf_model is not None:
        return True
    
    try:
        logger.info("🔄 Loading ML model...")
        
        try:
            rf_model = joblib.load('house_price_model.pkl')
            logger.info("✅ ML model loaded from cache")
            return True
        except:
            logger.info("🔄 No cached model found. Training new one...")
            
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            joblib.dump(rf_model, 'house_price_model.pkl')
            logger.info("✅ ML model trained and saved")
            return True
            
    except Exception as e:
        error_msg = f"Failed to load ML model: {str(e)}"
        logger.error(f"❌ {error_msg}")
        model_load_error = error_msg
        return False

def load_embedder():
    """Load Sentence Transformer - called on first request"""
    global embedder, property_embeddings, model_load_error
    
    if embedder is not None:
        return True
    
    try:
        logger.info("🧠 Loading Sentence Transformer...")
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            error_msg = "sentence-transformers package not installed"
            logger.error(f"❌ {error_msg}")
            model_load_error = error_msg
            return False
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        logger.info("📊 Computing property embeddings...")
        property_texts = [
            f"Property in {p['neighborhood']}, {p['address']}. "
            f"Priced at ${p['price']} with {p['bedrooms']} bedrooms. "
            f"{p['description']}"
            for p in properties
        ]
        property_embeddings = embedder.encode(property_texts, convert_to_numpy=True)
        
        logger.info("✅ Sentence Transformer loaded successfully!")
        return True
        
    except Exception as e:
        error_msg = f"Failed to load embedder: {str(e)}"
        logger.error(f"❌ {error_msg}")
        model_load_error = error_msg
        return False

def ensure_models():
    """Ensure both models are loaded"""
    global models_loaded
    
    if models_loaded:
        return True, None
    
    if not load_ml_model():
        return False, model_load_error
    
    if not load_embedder():
        return False, model_load_error
    
    models_loaded = True
    return True, None

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_preferences(user_message):
    """Extract budget, bedrooms, location from message"""
    preferences = {
        'budget': None,
        'bedrooms': None,
        'city': None
    }
    
    msg = user_message.lower()
    
    budget_match = re.search(r'(?:under|below|less than|budget of|around|max)\s+\$?(\d{1,3}(?:,\d{3})*)\s*(k|thousand|million)?', msg)
    if budget_match:
        val = int(budget_match.group(1).replace(',', ''))
        unit = budget_match.group(2) if len(budget_match.groups()) > 1 else None
        if unit in ['k', 'thousand'] or (val < 1000):
            val *= 1000
        elif unit in ['m', 'million']:
            val *= 1000000
        preferences['budget'] = val
    
    bed_match = re.search(r'(\d+)\s*(?:bed|bdr|bedroom)', msg)
    if bed_match:
        preferences['bedrooms'] = int(bed_match.group(1))
    
    cities = {
        'los angeles': [r'\blos angeles\b', r'\bla\b'],
        'san francisco': [r'\bsan francisco\b', r'\bsf\b'],
        'san diego': [r'\bsan diego\b'],
        'sacramento': [r'\bsacramento\b'],
        'fresno': [r'\bfresno\b']
    }
    for city, patterns in cities.items():
        if any(re.search(p, msg) for p in patterns):
            preferences['city'] = city
            break
    
    return preferences

def predict_price(preferences):
    """Predict price using ML model"""
    try:
        model = rf_model
        if model is None:
            return 400000.0
        
        budget = preferences.get('budget') or 500000
        med_inc = max(0.5, min(15.0, budget / 100000.0))
        bedrooms = preferences.get('bedrooms') or 3
        
        city_coords = {
            'los angeles': (34.05, -118.24),
            'san francisco': (37.77, -122.42),
            'san diego': (32.72, -117.16),
            'sacramento': (38.58, -121.49),
            'fresno': (36.75, -119.77)
        }
        lat, lon = city_coords.get(preferences.get('city'), (34.05, -118.24))
        
        features = np.array([[
            med_inc, 25.0, (bedrooms + 2) / 3.0, bedrooms / 3.0,
            1200.0, 3.0, lat, lon
        ]])
        
        predicted = model.predict(features)[0] * 100000.0
        return float(np.round(predicted, -3))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 400000.0

def semantic_search(query, top_k=3):
    """Perform semantic search"""
    try:
        if embedder is None or property_embeddings is None:
            return []
        
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, property_embeddings)[0]
        
        ranked_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score >= 0.15 and len(results) < top_k:
                prop = properties[idx].copy()
                prop['similarity_score'] = round(score, 4)
                results.append(prop)
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_model_loaded': rf_model is not None,
        'embedder_loaded': embedder is not None,
        'models_ready': models_loaded
    })

@app.route('/api/chat/semantic', methods=['POST'])
def semantic_chat():
    """Main semantic chat endpoint"""
    try:
        data = request.json or {}
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message required'}), 400
        
        success, error = ensure_models()
        if not success:
            return jsonify({
                'error': 'Models failed to load',
                'details': error,
                'response': f"😅 The AI models failed to load. Please try again in a minute.\n\nError: {error}"
            }), 503
        
        preferences = extract_preferences(user_message)
        predicted_price = predict_price(preferences)
        semantic_matches = semantic_search(user_message, top_k=3)
        
        response_parts = [f"🔍 Here are the best matches for: \"{user_message}\""]
        if predicted_price:
            response_parts.append(f"📊 **ML Estimated Value:** ~${predicted_price:,.0f}\n")
        
        if semantic_matches:
            for i, p in enumerate(semantic_matches, 1):
                response_parts.append(
                    f"{i}. **{p['address']}** (Match: {p['similarity_score'] * 100:.1f}%)\n"
                    f"   💰 **${p['price']:,}** | 🛏️ {p['bedrooms']} bed / {p['bathrooms']} bath | 📐 {p['sqft']} sqft\n"
                    f"   📝 {p['description']}\n"
                )
        else:
            response_parts.append("😅 No properties found matching that query.")
        
        return jsonify({
            'response': "\n".join(response_parts),
            'query': user_message,
            'preferences': preferences,
            'predicted_price': predicted_price,
            'properties': semantic_matches,
            'property_count': len(semantic_matches)
        })
        
    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        return jsonify({
            'response': f"😅 Error: {str(e)}"
        }), 500

# ==========================================
# STARTUP
# ==========================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 FLASK API RUNNING")
    print("=" * 50)
    print("📡 Health check: GET /api/health")
    print("📡 Semantic search: POST /api/chat/semantic")
    print("=" * 50 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)