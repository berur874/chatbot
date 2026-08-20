import re
from rapidfuzz import process, fuzz
import spacy

# Load spaCy's lightweight English model for POS tagging / dependency parsing
nlp = spacy.load("en_core_web_sm")

# Known Location Knowledge Base (Cities & Estates)
LOCATION_MAP = {
    'nairobi': ['nairobi', 'kilimani', 'kileleshwa', 'westlands', 'lavington', 'karen', 'runda', 'ruaka', 'langata', 'parklands', 'muthaiga', 'south c', 'syokimau', 'kitengela', 'kasarani'],
    'mombasa': ['mombasa', 'nyali', 'bamburi', 'kizingo', 'shanzu', 'diani', 'mtwapa'],
    'kisumu': ['kisumu', 'milimani', 'riat'],
    'nakuru': ['nakuru', 'section 58'],
    'eldoret': ['eldoret', 'elgon view'],
    'kiambu': ['kiambu', 'ruiru', 'tatu city', 'kahawa']
}

# Flatten locations for fast fuzzy search
ALL_LOCATIONS = {loc: city for city, sub_locs in LOCATION_MAP.items() for loc in sub_locs}

# Property Type Dictionary
PROPERTY_TYPES = {
    'apartment': ['apartment', 'apt', 'flat'],
    'townhouse': ['townhouse', 'town house', 'maisonette'],
    'villa': ['villa', 'bungalow', 'mansion'],
    'studio': ['studio', 'bedsitter', 'bed-sitter', 'single room']
}

# Amenity Keyword Set
AMENITIES = ['pool', 'swimming pool', 'gym', 'borehole', 'generator', 'balcony', 'parking', 'garden', 'security']


def parse_budget_bounds(msg: str) -> dict:
    """Extracts min, max, or exact budget constraints."""
    bounds = {'min_price': None, 'max_price': None}
    
    # 1. Range Detection (e.g., "between 10m and 15m" or "10m to 15m")
    range_match = re.search(r'(?:between|from)\s+(\d+[\d\.,]*\s*[km]?)\s+(?:and|to|-)\s+(\d+[\d\.,]*\s*[km]?)', msg, re.IGNORECASE)
    if range_match:
        bounds['min_price'] = _convert_price(range_match.group(1))
        bounds['max_price'] = _convert_price(range_match.group(2))
        return bounds

    # 2. Max Price Detection (e.g., "under 15m", "below 50k", "max 20m")
    max_match = re.search(r'(?:under|below|less than|max|up to)\s+(\d+[\d\.,]*\s*(?:k|thousand|m|million)?)', msg, re.IGNORECASE)
    if max_match:
        bounds['max_price'] = _convert_price(max_match.group(1))

    # 3. Min Price Detection (e.g., "above 10m", "at least 50k", "min 5m")
    min_match = re.search(r'(?:above|over|more than|at least|min|from)\s+(\d+[\d\.,]*\s*(?:k|thousand|m|million)?)', msg, re.IGNORECASE)
    if min_match:
        bounds['min_price'] = _convert_price(min_match.group(1))

    return bounds


def _convert_price(price_str: str) -> int | None:
    """Helper to convert text numbers into integers."""
    match = re.search(r'(\d+(?:\.\d+)?)\s*(k|thousand|m|million)?', price_str.lower())
    if not match:
        return None
    val = float(match.group(1))
    unit = match.group(2)
    if unit in ['k', 'thousand']:
        val *= 1_000
    elif unit in ['m', 'million']:
        val *= 1_000_000
    return int(val)


def extract_advanced_preferences(msg: str) -> dict:
    """Enhanced search parser with fuzzy matching, intent detection, and bounds."""
    doc = nlp(msg)
    msg_lower = msg.lower()

    prefs = {
        'intent': 'buy', # Default intent
        'property_type': None,
        'min_price': None,
        'max_price': None,
        'bedrooms': None,
        'city': None,
        'matched_location': None,
        'amenities': []
    }

    # 1. Intent Detection (Rent vs. Buy)
    if any(kw in msg_lower for kw in ['rent', 'to let', 'monthly', 'per month', 'lease']):
        prefs['intent'] = 'rent'

    # 2. Price Bounds Parsing
    price_bounds = parse_budget_bounds(msg_lower)
    prefs['min_price'] = price_bounds['min_price']
    prefs['max_price'] = price_bounds['max_price']

    # 3. Property Type Detection
    for prop_type, synonyms in PROPERTY_TYPES.items():
        if any(syn in msg_lower for syn in synonyms):
            prefs['property_type'] = prop_type
            break

    # 4. Bedrooms parsing (SpaCy Token Matching + Regex fallback)
    word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
    bed_match = re.search(r'(\d+|one|two|three|four|five)\s*(?:bed|bdr|bedroom|br)', msg_lower)
    if bed_match:
        raw_bed = bed_match.group(1)
        prefs['bedrooms'] = word_to_num.get(raw_bed, int(raw_bed) if raw_bed.isdigit() else None)

    # 5. Fuzzy Location Matching (Handles Typos)
    words = [token.text for token in doc if not token.is_stop]
    for word in words:
        if len(word) > 3: # Ignore small words
            best_match = process.extractOne(word, ALL_LOCATIONS.keys(), scorer=fuzz.ratio)
            if best_match and best_match[1] >= 85: # 85% similarity threshold
                matched_loc = best_match[0]
                prefs['matched_location'] = matched_loc
                prefs['city'] = ALL_LOCATIONS[matched_loc]
                break

    # 6. Amenity Extraction
    for amenity in AMENITIES:
        if amenity in msg_lower:
            prefs['amenities'].append(amenity)

    return prefs