import re

import numpy as np
from django.conf import settings

# Persistent, project-local cache dir for fastembed's downloaded model files.
# fastembed defaults to tempfile.gettempdir()/fastembed_cache, which meant the
# first request on a cold worker downloaded ~90MB from the HF Hub synchronously
# inside the request, blocking gunicorn's single sync worker past its 300s
# timeout (observed as a WORKER TIMEOUT -> SIGKILL in production). Pointing
# this at a project-local dir lets `manage.py warm_embedding_model` populate it
# during build.sh instead, off the request path.
FASTEMBED_CACHE_DIR = str(settings.BASE_DIR / 'models' / 'fastembed_cache')

# BGE's recommended prefix for the query side of asymmetric search (a short
# query embedded for comparison against long passages). Apply only to the
# user's synthesized query text before embedding it against JobEmbedding
# vectors - never to the job passages themselves, and never to the
# symmetric short-phrase skill-vs-skill comparisons in scorers.py.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def normalize_skill(text):
    """Canonical key for the skill-embedding cache/table: lowercase, collapse
    whitespace, strip. Mirrors data/extraction's job-side normalization so
    ingest-normalized job skills and raw user-typed skills key-match, without
    importing that package (it pulls in spaCy/KeyBERT and is deliberately
    kept out of the web process for the same memory-budget reasons as below).
    """
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower()).strip()


class EmbeddingService:
    """Lazy-loaded sentence embedding service.

    Uses fastembed (ONNX Runtime) rather than sentence-transformers+torch:
    same model (BAAI/bge-small-en-v1.5), numerically equivalent output to
    the torch-based encoder for the same input, but without torch's memory
    footprint - torch's default PyPI build alone bundles CUDA runtime
    libraries that were the main cause of this app's OOM on a small instance.

    BGE models are trained for asymmetric query/passage retrieval and
    officially recommend prefixing the *query* side (never the passage/
    document side) with BGE_QUERY_PREFIX for asymmetric search. The skill-
    to-skill comparisons in scorers.py are symmetric short-phrase matching
    and must stay unprefixed; only job-level full-text search (JobEmbedding)
    is asymmetric and should apply the prefix to the user's query text.

    Embeddings are cached in three tiers, checked in order:
      L1: _embedding_cache, an in-process dict - free, but wiped on every
          worker restart (gunicorn max_requests recycle) and never shared
          across workers.
      L2: the SkillEmbedding table - persists across restarts/deploys and is
          shared by every worker/process. Preloaded into L1 once per worker.
      L3: the fastembed model itself - only for skills truly never seen
          before. Results are written back to L2 (write-through) so this
          cost is paid at most once per distinct skill, ever.
    """

    _instance = None
    _model = None
    _embedding_cache = {}  # normalized skill -> embedding, shared across requests in this worker
    _l2_warmed = False

    MODEL_NAME = 'BAAI/bge-small-en-v1.5'
    DIM = 384

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
                    model_name=self.MODEL_NAME,
                    cache_dir=FASTEMBED_CACHE_DIR,
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

    def _warm_l2_cache(self):
        """Preload the whole SkillEmbedding table into L1, once per worker.
        Cheap: the skill vocabulary is far smaller than the job count (a few
        thousand rows at most), e.g. 5,000 skills * 384 float32 ~= 7.7MB."""
        if EmbeddingService._l2_warmed:
            return
        try:
            from jobs.models import SkillEmbedding
            for skill, vector, dim in SkillEmbedding.objects.filter(
                model_name=self.MODEL_NAME
            ).values_list('skill', 'vector', 'dim'):
                if dim == self.DIM and len(vector) == self.DIM:
                    EmbeddingService._embedding_cache[skill] = np.asarray(vector, dtype=np.float32)
        except Exception as e:
            print(f"Warning: could not warm skill-embedding cache from DB: {e}")
        finally:
            # Don't retry every call if the DB is briefly unavailable.
            EmbeddingService._l2_warmed = True

    def _write_through(self, new_rows):
        """Persist newly-computed embeddings so future requests (this worker
        after a recycle, or any other worker) don't re-run inference for the
        same skill. Best-effort: a DB write failure must not break matching."""
        if not new_rows:
            return
        try:
            from jobs.models import SkillEmbedding
            SkillEmbedding.objects.bulk_create(
                [
                    SkillEmbedding(
                        skill=key,
                        vector=np.asarray(vec, dtype=np.float32).tolist(),
                        model_name=self.MODEL_NAME,
                        dim=self.DIM,
                    )
                    for key, vec in new_rows
                ],
                update_conflicts=True,
                unique_fields=['skill'],
                update_fields=['vector', 'model_name', 'dim', 'updated_at'],
            )
        except Exception as e:
            print(f"Warning: failed to persist new skill embeddings: {e}")

    def embed(self, text):
        """Embed a single text"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        """Embed multiple texts, reusing cached embeddings for text seen
        before - checking the in-process cache, then the persistent
        SkillEmbedding table, before falling back to the model itself."""
        self._load_model()
        self._warm_l2_cache()

        texts = list(texts)
        cache = EmbeddingService._embedding_cache
        key_of = {t: normalize_skill(t) for t in texts}

        # L1 miss -> ask the DB directly for these specific keys, in case
        # another worker/process (or the backfill command) wrote them after
        # this worker warmed its cache.
        l1_misses = sorted({k for k in key_of.values() if k and k not in cache})
        if l1_misses:
            try:
                from jobs.models import SkillEmbedding
                rows = SkillEmbedding.objects.filter(
                    skill__in=l1_misses, model_name=self.MODEL_NAME
                ).values_list('skill', 'vector', 'dim')
                for skill, vector, dim in rows:
                    if dim == self.DIM and len(vector) == self.DIM:
                        cache[skill] = np.asarray(vector, dtype=np.float32)
            except Exception as e:
                print(f"Warning: skill-embedding DB lookup failed, falling back to model: {e}")

        # True misses: never-before-seen skills -> run fastembed, once, batched.
        true_misses = sorted({k for k in key_of.values() if k and k not in cache})
        if true_misses:
            # fastembed.embed() returns a generator of L2-normalized numpy arrays
            new_embeddings = list(EmbeddingService._model.embed(true_misses))
            new_rows = []
            for key, embedding in zip(true_misses, new_embeddings):
                cache[key] = embedding
                new_rows.append((key, embedding))
            self._write_through(new_rows)

        zero_vec = np.zeros(self.DIM, dtype=np.float32)
        return np.array([cache.get(key_of[t], zero_vec) for t in texts])

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
