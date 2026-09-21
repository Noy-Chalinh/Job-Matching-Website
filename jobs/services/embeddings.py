import numpy as np

class EmbeddingService:
    """Lazy-loaded sentence embedding service"""

    _instance = None
    _model = None
    _embedding_cache = {}  # text -> embedding, shared across requests in this worker

    def __new__(cls):
        """Singleton pattern to ensure only one model instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        """Lazy load embedding model only when needed"""
        if EmbeddingService._model is None:
            print("Loading sentence transformer model...")
            try:
                from sentence_transformers import SentenceTransformer
                # Use a smaller, more memory-efficient model
                EmbeddingService._model = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2',
                    device='cpu'  # Force CPU to avoid GPU memory issues
                )
                print("Model loaded successfully!")
            except MemoryError:
                print("ERROR: Not enough memory to load embedding model. Semantic matching disabled.")
                EmbeddingService._model = False  # Mark as failed
                raise
            except Exception as e:
                print(f"Error loading model: {e}. Semantic matching disabled.")
                EmbeddingService._model = False
                raise

    def embed(self, text):
        """Embed a single text"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        """Embed multiple texts, reusing cached embeddings for text seen before"""
        self._load_model()

        texts = list(texts)
        cache = EmbeddingService._embedding_cache
        uncached = [t for t in texts if t not in cache]

        if uncached:
            new_embeddings = EmbeddingService._model.encode(uncached, convert_to_numpy=True)
            for text, embedding in zip(uncached, new_embeddings):
                cache[text] = embedding

        return np.array([cache[t] for t in texts])

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
