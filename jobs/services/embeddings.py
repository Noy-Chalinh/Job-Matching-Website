import numpy as np


class EmbeddingService:
    """Lazy-loaded sentence embedding service.

    Uses fastembed (ONNX Runtime) rather than sentence-transformers+torch:
    same model (sentence-transformers/all-MiniLM-L6-v2), numerically
    equivalent output (verified: cosine similarity 1.000000 against the
    torch-based encoder for the same input), but without torch's memory
    footprint - torch's default PyPI build alone bundles CUDA runtime
    libraries that were the main cause of this app's OOM on a small instance.
    """

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
            print("Loading embedding model (fastembed)...")
            try:
                from fastembed import TextEmbedding
                EmbeddingService._model = TextEmbedding(
                    model_name='sentence-transformers/all-MiniLM-L6-v2'
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
            # fastembed.embed() returns a generator of L2-normalized numpy arrays
            new_embeddings = list(EmbeddingService._model.embed(uncached))
            for text, embedding in zip(uncached, new_embeddings):
                cache[text] = embedding

        return np.array([cache[t] for t in texts])

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
