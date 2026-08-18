import os

# Limit thread usage for low-resource environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import re
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. FLASK APPLICATION INITIALIZATION
# ==========================================

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nyumbahub.vercel.app",  # Your production URL
    "https://nyumbahub.netlify.app"   # Alternative production URL
]}})


print("=" * 50)
print("🤖 ML & SEMANTIC CHATBOT API STARTING...")
print("=" * 50)

# ==========================================
# 2. LAZY LOADING - Memory Efficient
# ==========================================

_embedder = None
_rf_model = None
_property_embeddings = None
_properties = None


def get_embedder():
    """Lazy load sentence transformer only when needed"""
    global _embedder
    if _embedder is None:
        logger.info("🧠 Loading Sentence Transformer (first time)...")
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Embedder loaded!")
    return _embedder


def get_rf_model():
    """Lazy load Random Forest model"""
    global _rf_model
    if _rf_model is None:
        try:
            _rf_model = joblib.load("house_price_model.pkl")
            logger.info("✅ Loaded saved Random Forest model")
        except Exception:
            logger.info("🔄 Training new Random Forest model...")
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            _rf_model = RandomForestRegressor(
                n_estimators=50, random_state=42, n_jobs=1
            )
            _rf_model.fit(X_train, y_train)
            joblib.dump(_rf_model, "house_price_model.pkl")
            logger.info("✅ Model trained and saved successfully")
    return _rf_model


def get_properties():
    """Lazy load properties and embeddings"""
    global _properties, _property_embeddings

    if _properties is None:
        _properties = [
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
                "description": "Beautiful 3-bedroom home in the heart of LA, close to vibrant nightlife and restaurants.",
                "school_rating": 8,
                "walk_score": 85,
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
                "description": "Charming 2-bedroom condo in trendy SF neighborhood with coffee shops and parks.",
                "school_rating": 9,
                "walk_score": 92,
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
                "description": "Spacious family home near the beach with ocean views, top schools, and a big backyard.",
                "school_rating": 10,
                "walk_score": 70,
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
                "description": "Classic Sacramento home in walkable historic neighborhood with shade trees.",
                "school_rating": 6,
                "walk_score": 88,
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
                "description": "Affordable starter home in growing Fresno area with modern kitchen upgrades.",
                "school_rating": 5,
                "walk_score": 65,
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
                "description": "Cozy beach cottage steps away from coastal surf, sand, and boardwalks.",
                "school_rating": 7,
                "walk_score": 90,
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
                "description": "Charming low-maintenance starter home in quiet suburban neighborhood.",
                "school_rating": 6,
                "walk_score": 60,
            },
        ]

    # Precompute embeddings lazily and cache globally
    if _property_embeddings is None:
        embedder = get_embedder()
        corpus = [
            f"Property in {p['neighborhood']}, {p['address']}. "
            f"Priced at ${p['price']} with {p['bedrooms']} bedrooms and {p['bathrooms']} bathrooms. "
            f"{p['sqft']} square feet. Description: {p['description']}"
            for p in _properties
        ]
        _property_embeddings = embedder.encode(corpus, convert_to_numpy=True)
        logger.info("✅ Vector embeddings calculated for properties!")

    return _properties, _property_embeddings


# ==========================================
# 3. PREFERENCE EXTRACTION & ML PREDICTION
# ==========================================


def extract_preferences(user_message):
    """Extract budget, bedrooms, and location metadata"""
    preferences = {
        "budget": None,
        "bedrooms": None,
        "city": None,
        "intent": "general",
    }

    msg = user_message.lower()

    # 1. Budget Regex Patterns
    budget_patterns = [
        r"(?:under|below|less than|budget of|around|max)\s+\$?(\d{1,3}(?:,\d{3})*)\s*(k|thousand|million|m)?",
        r"\$(\d{1,3}(?:,\d{3})*)\s*(k|thousand|million|m)?",
    ]
    for pattern in budget_patterns:
        match = re.search(pattern, msg)
        if match:
            val = int(match.group(1).replace(",", ""))
            unit = match.group(2) if len(match.groups()) > 1 else None
            if unit in ["k", "thousand"] or (val < 1000 and not unit):
                val *= 1000
            elif unit in ["m", "million"]:
                val *= 1000000
            preferences["budget"] = val
            break

    # 2. Bedrooms Regex
    bed_match = re.search(r"(\d+)\s*(?:bed|bdr|bedroom)", msg)
    if bed_match:
        preferences["bedrooms"] = int(bed_match.group(1))

    # 3. City Regex
    cities = {
        "los angeles": [r"\blos angeles\b", r"\bla\b"],
        "san francisco": [r"\bsan francisco\b", r"\bsf\b"],
        "san diego": [r"\bsan diego\b"],
        "sacramento": [r"\bsacramento\b"],
        "fresno": [r"\bfresno\b"],
    }
    for city, patterns in cities.items():
        if any(re.search(p, msg) for p in patterns):
            preferences["city"] = city
            break

    return preferences


def predict_price_from_preferences(preferences):
    """Predict market value using Random Forest model"""
    rf_model = get_rf_model()

    if rf_model is None:
        return 400000.0

    try:
        budget = preferences.get("budget") or 500000
        med_inc = max(0.5, min(15.0, budget / 100000.0))
        bedrooms = preferences.get("bedrooms") or 3

        occupancy = 3.0
        ave_rooms = (bedrooms + 2) / occupancy
        ave_bedrms = bedrooms / occupancy

        city_coords = {
            "los angeles": (34.05, -118.24),
            "san francisco": (37.77, -122.42),
            "san diego": (32.72, -117.16),
            "sacramento": (38.58, -121.49),
            "fresno": (36.75, -119.77),
        }
        lat, lon = city_coords.get(preferences.get("city"), (34.05, -118.24))

        features = np.array(
            [[med_inc, 25.0, ave_rooms, ave_bedrms, 1200.0, occupancy, lat, lon]]
        )

        predicted = rf_model.predict(features)[0] * 100000.0
        return float(np.round(predicted, -3))
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return 400000.0


def vector_search(query_text, top_k=3, min_score=0.15):
    """Calculates cosine similarity between user query and property vectors"""
    embedder = get_embedder()
    properties, property_embeddings = get_properties()

    query_embedding = embedder.encode([query_text], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, property_embeddings)[0]

    # Sort indices by highest similarity
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_indices:
        score = float(similarities[idx])
        if score >= min_score and len(results) < top_k:
            prop = properties[idx].copy()
            prop["similarity_score"] = round(score, 4)
            results.append(prop)

    return results


# ==========================================
# 4. API ENDPOINTS
# ==========================================


@app.route("/api/chat/semantic", methods=["POST"])
def semantic_chat():
    """Main vector search endpoint for semantic queries"""
    try:
        data = request.json or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message required"}), 400

        # Extract basic filters & run predictions
        preferences = extract_preferences(user_message)
        predicted_price = predict_price_from_preferences(preferences)

        # Perform Semantic Search
        semantic_matches = vector_search(user_message, top_k=3)

        # Build Response Message
        response_parts = [
            f'🔍 Here are the best semantic matches for: "{user_message}"'
        ]
        if predicted_price:
            response_parts.append(
                f"📊 **ML Estimated Value:** ~${predicted_price:,.0f}\n"
            )

        if semantic_matches:
            for i, p in enumerate(semantic_matches, 1):
                response_parts.append(
                    f"{i}. **{p['address']}** (Match Score: {p['similarity_score'] * 100:.1f}%)\n"
                    f"   💰 **${p['price']:,}** | 🛏️ {p['bedrooms']} bed / {p['bathrooms']} bath | 📐 {p['sqft']} sqft\n"
                    f"   📝 {p['description']}\n"
                )
        else:
            response_parts.append(
                "😅 No properties found matching that semantic context."
            )

        return jsonify(
            {
                "response": "\n".join(response_parts),
                "query": user_message,
                "preferences": preferences,
                "predicted_price": predicted_price,
                "properties": semantic_matches,
                "property_count": len(semantic_matches),
            }
        )

    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "ml_model_loaded": _rf_model is not None,
            "embedder_loaded": _embedder is not None,
        }
    )


# FIX: Renamed handler function to avoid recursion with helper function get_properties()
@app.route("/api/properties", methods=["GET"])
def list_all_properties():
    properties, _ = get_properties()
    return jsonify(properties)


# ==========================================
# 5. RUN SERVER
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 FLASK API RUNNING (Memory Optimized)")
    print("=" * 50)
    print("📡 Health check: GET /api/health")
    print("📡 Semantic search: POST /api/chat/semantic")
    print("💡 Models load on first request (saves memory)")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)