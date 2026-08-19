def extract_preferences(msg):
    """Extract budget, bedrooms, city from message"""
    prefs = {'budget': None, 'bedrooms': None, 'city': None}
    msg = msg.lower()
    
    # Budget
    match = re.search(r'(?:under|below|less than|budget|max)\s+\$?(\d{1,3}(?:,\d{3})*)\s*(k|thousand|million)?', msg)
    if match:
        val = int(match.group(1).replace(',', ''))
        unit = match.group(2) if len(match.groups()) > 1 else None
        if unit in ['k', 'thousand']:
            val *= 1000
        elif unit in ['m', 'million']:
            val *= 1000000
        prefs['budget'] = val
    
    # Bedrooms
    match = re.search(r'(\d+)\s*(?:bed|bdr|bedroom)', msg)
    if match:
        prefs['bedrooms'] = int(match.group(1))
    
    # City
    cities = {
        'los angeles': ['los angeles', 'la'],
        'san francisco': ['san francisco', 'sf'],
        'san diego': ['san diego'],
        'sacramento': ['sacramento'],
        'fresno': ['fresno']
    }
    for city, patterns in cities.items():
        if any(p in msg for p in patterns):
            prefs['city'] = city
            break
    
    return prefs
