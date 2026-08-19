# semantic_search.py - Semantic search logic
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from properties import properties

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self):
        self.embedder = None
        self.property_embeddings = None
        self.load_embedder()
    
    def load_embedder(self):
        """Load the sentence transformer model"""
        try:
            logger.info("🧠 Loading Sentence Transformer...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            # Create property embeddings
            logger.info("📊 Computing property embeddings...")
            property_texts = [
                f"Property in {p['neighborhood']}, {p['address']}. "
                f"Priced at ${p['price']} with {p['bedrooms']} bedrooms. "
                f"{p['description']}"
                for p in properties
            ]
            self.property_embeddings = self.embedder.encode(
                property_texts, 
                convert_to_numpy=True
            )
            logger.info("✅ Sentence Transformer loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load embedder: {e}")
            return False
    
    def search(self, query, top_k=3):
        """Perform semantic search"""
        if self.embedder is None or self.property_embeddings is None:
            return []
        
        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(query_embedding, self.property_embeddings)[0]
            
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

# Singleton instance
semantic_search = SemanticSearch()