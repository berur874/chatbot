import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from properties import properties
def simple_search(query, vectorizer, property_vectors, top_k=3):
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