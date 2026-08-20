# test_supabase.py
import os
from supabase import create_client

# Replace with your actual credentials
SUPABASE_URL="https://fgjcuumoqmlkqrjichtl.supabase.co"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZnamN1dW1vcW1sa3FyamljaHRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODY4NjEwMjcsImV4cCI6MjEwMjQzNzAyN30.gBjwLXPVJ3vac8Rw2EfBpBwgX2B0dshn9n4yXt0GRhY"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Test connection
response = supabase.table("properties").select("*").limit(5).execute()
print(f"Found {len(response.data)} properties")
for prop in response.data:
    print(f"  - {prop['address']}: ${prop['price']:,}")