import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional
from properties import get_all_properties


class PropertySearchEngine:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.property_vectors: Optional[Any] = None
        self.properties: List[Dict[str, Any]] = []

    def build_index(self) -> None:
        """Fetch fresh property data from Supabase and build TF-IDF vectors."""
        self.properties = get_all_properties()
        if not self.properties:
            self.vectorizer = None
            self.property_vectors = None
            return

        # Combine text fields for TF-IDF feature extraction
        corpus = [
            f"{p.get('address', '')} {p.get('neighborhood', '')} {p.get('description', '')}"
            for p in self.properties
        ]

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.property_vectors = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.01) -> List[Dict[str, Any]]:
        """Search indexed properties by text similarity."""
        # Auto-build index on first search if empty
        if self.vectorizer is None or self.property_vectors is None:
            self.build_index()

        if not self.properties or self.vectorizer is None:
            return []

        # Vectorize incoming search string
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.property_vectors)[0]
        
        # Extract indices sorted by relevance score descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > score_threshold:
                # Safe copy to attach score without mutating original memory object
                prop = dict(self.properties[idx])
                prop['similarity_score'] = round(score, 4)
                results.append(prop)

        return results


# Global search engine instance
search_engine = PropertySearchEngine()

def search_properties(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Helper wrapper function to trigger searches directly."""
    return search_engine.search(query=query, top_k=top_k)