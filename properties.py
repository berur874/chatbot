# properties.py
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Get Supabase credentials from environment
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_all_properties(limit=None):
    """Get all properties from Supabase"""
    try:
        query = supabase.table("properties").select("*")
        if limit:
            query = query.limit(limit)
        response = query.execute()
        return response.data
    except Exception as e:
        print(f"Error fetching properties: {e}")
        return []

def filter_properties(budget=None, bedrooms=None, city=None, limit=10):
    """Filter properties by user preferences"""
    try:
        query = supabase.table("properties").select("*")
        
        if budget:
            query = query.lte("price", budget)
        if bedrooms:
            query = query.gte("bedrooms", bedrooms)
        if city:
            # Case-insensitive search in address and neighborhood
            query = query.ilike("address", f"%{city}%")
        
        query = query.limit(limit)
        response = query.execute()
        return response.data
    except Exception as e:
        print(f"Error filtering properties: {e}")
        return []

def get_property_by_id(property_id):
    """Get a single property by ID"""
    try:
        response = supabase.table("properties").select("*").eq("id", property_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error fetching property: {e}")
        return None

def count_properties():
    """Get total number of properties"""
    try:
        response = supabase.table("properties").select("*", count="exact").limit(0).execute()
        return response.count if hasattr(response, 'count') else 0
    except Exception as e:
        print(f"Error counting properties: {e}")
        return 0

# Load properties on startup (cached)
properties = get_all_properties()