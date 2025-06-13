import asyncio
from typing import Any, Dict, List, Mapping, Optional
import numpy as np
import pytest

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.cache import SemanticCacheBackend, cacher, CacheInterface

# --- Test Data & Mock Embeddings ---

# Use pre-normalized vectors for exact cosine similarities
embedding_A = np.array([1.0, 0.0, 0.0]) # Reference
embedding_S_095 = np.array([0.95, np.sqrt(1 - 0.95**2), 0.0]) # CosSim with A is 0.95
embedding_S_080 = np.array([0.80, np.sqrt(1 - 0.80**2), 0.0]) # CosSim with A is 0.80
embedding_S_060 = np.array([0.60, np.sqrt(1 - 0.60**2), 0.0]) # CosSim with A is 0.60
embedding_B = np.array([0.0, 1.0, 0.0]) # Orthogonal to A, CosSim with A is 0.0
embedding_unknown = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]) # Default for unknown text

text_original = "What is the capital of France?"
text_paraphrased_high_similarity = "France's capital, what is it?" # Maps to embedding_S_095
text_paraphrased_medium_similarity = "Tell me about the capital of France." # Maps to embedding_S_080
text_paraphrased_low_similarity = "Is Paris the main city in France?" # Maps to embedding_S_060
text_different = "Tell me a joke." # Maps to embedding_B
text_for_zeronorm = "text for zero norm"
text_for_empty_emb = "text for empty_embedding"


class MockEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embedding_map: Dict[str, np.ndarray]):
        self.embedding_map = embedding_map
        self.default_embedding = embedding_unknown

    def embed_query(self, text: str) -> List[float]:
        embedding = self.embedding_map.get(text)
        if embedding is None:
            # As per subtask: "If text is not in map, raise an error or return a default 'unknown' embedding."
            # Returning default here.
            return self.default_embedding.tolist()
        return embedding.tolist()

    async def embed_query_async(self, text: str) -> List[float]:
        await asyncio.sleep(0)
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        await asyncio.sleep(0)
        return self.embed_documents(texts)


@pytest.fixture(scope="function")
def mock_embeddings_instance() -> MockEmbeddings:
    return MockEmbeddings({
        text_original: embedding_A,
        text_paraphrased_high_similarity: embedding_S_095,
        text_paraphrased_medium_similarity: embedding_S_080,
        text_paraphrased_low_similarity: embedding_S_060,
        text_different: embedding_B,
        text_for_zeronorm: np.array([0.0, 0.0, 0.0]), # Zero norm embedding
    })

@pytest.fixture(scope="function")
def semantic_cache_backend_factory(mock_embeddings_instance: MockEmbeddings):
    def _factory(threshold: float) -> SemanticCacheBackend:
        cache = SemanticCacheBackend(
            embedding_model=mock_embeddings_instance,
            similarity_threshold=threshold
        )
        # Ensure cache is clean for each test/instance
        # SemanticCacheBackend.cache is a list, so clear it.
        cache.cache.clear()
        return cache
    return _factory

class CallCounter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.count == other
        return NotImplemented

@pytest.fixture
def call_counter() -> CallCounter:
    return CallCounter()

# --- Test Cases ---

def test_semantic_cache_hit_simple(semantic_cache_backend_factory, call_counter: CallCounter):
    cache = semantic_cache_backend_factory(threshold=0.90) # S_095 sim is 0.95

    @cacher(cache_backend=cache)
    def get_data(prompt: str) -> str:
        call_counter.increment()
        return f"Data for {prompt}"

    res1 = get_data(text_original)
    assert call_counter == 1
    assert res1 == f"Data for {text_original}"

    res2 = get_data(text_paraphrased_high_similarity) # Sim 0.95 >= 0.90, HIT
    assert call_counter == 1
    assert res2 == f"Data for {text_original}" # Returns original cached value

def test_semantic_cache_miss_different_meaning(semantic_cache_backend_factory, call_counter: CallCounter):
    cache = semantic_cache_backend_factory(threshold=0.90)

    @cacher(cache_backend=cache)
    def get_data(prompt: str) -> str:
        call_counter.increment()
        return f"Data for {prompt}"

    get_data(text_original)
    assert call_counter == 1

    get_data(text_different)  # Sim with A is 0.0 < 0.90, MISS
    assert call_counter == 2
    assert len(cache.cache) == 2

def test_semantic_cache_miss_due_to_threshold(semantic_cache_backend_factory):
    # text_paraphrased_medium_similarity has sim 0.80 with text_original (embedding_A)

    # Test with low threshold (should hit)
    cache_low_thresh = semantic_cache_backend_factory(threshold=0.75)
    counter_low = CallCounter()
    @cacher(cache_backend=cache_low_thresh)
    def get_data_low_thresh(prompt: str) -> str:
        counter_low.increment()
        return f"Data for {prompt}"

    get_data_low_thresh(text_original) # Stored
    assert counter_low == 1
    res_low2 = get_data_low_thresh(text_paraphrased_medium_similarity) # Sim 0.80 >= 0.75, HIT
    assert counter_low == 1
    assert res_low2 == f"Data for {text_original}"

    # Test with high threshold (should miss)
    cache_high_thresh = semantic_cache_backend_factory(threshold=0.85)
    counter_high = CallCounter()
    @cacher(cache_backend=cache_high_thresh)
    def get_data_high_thresh(prompt: str) -> str:
        counter_high.increment()
        return f"Data for {prompt}"

    get_data_high_thresh(text_original) # Stored
    assert counter_high == 1
    res_high2 = get_data_high_thresh(text_paraphrased_medium_similarity) # Sim 0.80 < 0.85, MISS
    assert counter_high == 2
    assert res_high2 == f"Data for {text_paraphrased_medium_similarity}"


def test_semantic_cache_hybrid_hit(semantic_cache_backend_factory, call_counter: CallCounter):
    cache = semantic_cache_backend_factory(threshold=0.90)

    @cacher(cache_backend=cache)
    def get_data_hybrid(prompt: str, temp: float) -> str:
        call_counter.increment()
        return f"Data for {prompt} with temp {temp}"

    res1 = get_data_hybrid(text_original, temp=0.5)
    assert call_counter == 1
    assert res1 == f"Data for {text_original} with temp 0.5"

    # Semantic part (prompt) is similar (0.95 >= 0.90), non-semantic part (temp) is identical
    res2 = get_data_hybrid(text_paraphrased_high_similarity, temp=0.5) # HIT
    assert call_counter == 1
    assert res2 == f"Data for {text_original} with temp 0.5"


def test_semantic_cache_hybrid_miss_non_semantic_arg(semantic_cache_backend_factory, call_counter: CallCounter):
    cache = semantic_cache_backend_factory(threshold=0.90)

    @cacher(cache_backend=cache)
    def get_data_hybrid(prompt: str, temp: float) -> str:
        call_counter.increment()
        return f"Data for {prompt} with temp {temp}"

    get_data_hybrid(text_original, temp=0.5)
    assert call_counter == 1

    # Semantic part (prompt) is similar (0.95 >= 0.90), but non-semantic part (temp) is different
    get_data_hybrid(text_paraphrased_high_similarity, temp=0.8) # MISS
    assert call_counter == 2

def test_semantic_cache_no_semantic_part(semantic_cache_backend_factory, call_counter: CallCounter):
    cache = semantic_cache_backend_factory(threshold=0.90)

    @cacher(cache_backend=cache)
    def get_data_no_string(num: int, flag: bool) -> str:
        call_counter.increment()
        return f"Data for {num}, {flag}"

    # _parse_key_str returns None if no string arg is found.
    # SemanticCacheBackend.set and .get return early if parse fails.
    # So, no caching occurs via semantic logic.
    get_data_no_string(10, True)
    assert call_counter == 1
    assert len(cache.cache) == 0 # Nothing cached as semantic part not found by SemanticCacheBackend

    get_data_no_string(10, True) # MISS (no semantic caching occurred)
    assert call_counter == 2
    assert len(cache.cache) == 0


@pytest.mark.asyncio
async def test_async_semantic_cache_hit(semantic_cache_backend_factory, call_counter: CallCounter):
    cache = semantic_cache_backend_factory(threshold=0.90)

    @cacher(cache_backend=cache)
    async def get_data_async(prompt: str) -> str:
        call_counter.increment()
        await asyncio.sleep(0.001)
        return f"Async data for {prompt}"

    res1 = await get_data_async(text_original)
    assert call_counter == 1
    assert res1 == f"Async data for {text_original}"

    res2 = await get_data_async(text_paraphrased_high_similarity) # HIT
    assert call_counter == 1
    assert res2 == f"Async data for {text_original}"


def test_cacher_uses_global_semantic_cache(semantic_cache_backend_factory, call_counter: CallCounter, mocker):
    global_semantic_cache = semantic_cache_backend_factory(threshold=0.90)
    mocker.patch("ragas.cache.ragas_cache", global_semantic_cache)

    @cacher() # Uses global ragas_cache by default
    def get_data_global(prompt: str) -> str:
        call_counter.increment()
        return f"Data for {prompt} from global"

    get_data_global(text_original)
    assert call_counter == 1
    res2 = get_data_global(text_paraphrased_high_similarity) # HIT
    assert call_counter == 1
    assert res2 == f"Data for {text_original} from global"


def test_semantic_cache_zero_norm_embeddings(semantic_cache_backend_factory, call_counter: CallCounter):
    cache = semantic_cache_backend_factory(threshold=0.0) # Threshold is 0.0

    # MockEmbeddings already maps text_for_zeronorm to [0,0,0]

    @cacher(cache_backend=cache)
    def get_data_zero_norm(prompt: str) -> str:
        call_counter.increment()
        return f"Data for {prompt}"

    # Store "text_for_zeronorm" (embedding is [0,0,0])
    res1 = get_data_zero_norm(text_for_zeronorm)
    assert call_counter == 1
    assert len(cache.cache) == 1

    # Query again with "text_for_zeronorm". Embedding is [0,0,0].
    # Stored embedding is [0,0,0]. Similarity is 0.0.
    # Threshold is 0.0. Condition: 0.0 >= 0.0 is True. So, HIT.
    res2 = get_data_zero_norm(text_for_zeronorm)
    assert call_counter == 1 # HIT
    assert res2 == f"Data for {text_for_zeronorm}"

    # Change threshold to be > 0.0
    cache.similarity_threshold = 0.1
    # Now, 0.0 >= 0.1 is False. Should be a MISS.
    res3 = get_data_zero_norm(text_for_zeronorm)
    assert call_counter == 2 # MISS
    assert res3 == f"Data for {text_for_zeronorm}"
    # Cache will now have two entries due to list.append if key representation differs or re-added
    # For this test, we mainly care about the call_counter.


class BadMockEmbeddings(BaseRagasEmbeddings): # Returns empty list for embeddings
    def embed_query(self, text: str) -> List[float]: return []
    async def embed_query_async(self, text: str) -> List[float]: return []
    def embed_documents(self, texts: List[str]) -> List[List[float]]: return [[] for _ in texts]
    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]: return [[] for _ in texts]

def test_semantic_cache_empty_embedding_from_model(semantic_cache_backend_factory, call_counter: CallCounter):
    cache = semantic_cache_backend_factory(threshold=0.9)
    # Replace the good mock embeddings with the bad one for this test
    cache.embedding_model = BadMockEmbeddings()

    @cacher(cache_backend=cache)
    def get_data_bad_embed(prompt: str) -> str:
        call_counter.increment()
        return f"Data for {prompt}"

    # SemanticCacheBackend.set: embedding = np.array([]) -> embedding.size == 0 -> returns early.
    # SemanticCacheBackend.get: input_embedding.size == 0 -> returns None.
    get_data_bad_embed(text_for_empty_emb)
    assert call_counter == 1
    assert len(cache.cache) == 0 # Nothing stored as embedding failed.

    get_data_bad_embed(text_for_empty_emb) # Call again
    assert call_counter == 2 # Called again because get returns None (no cache hit).
    assert len(cache.cache) == 0


def test_semantic_cache_non_string_semantic_part_behavior(semantic_cache_backend_factory, call_counter: CallCounter, mock_embeddings_instance: MockEmbeddings):
    cache = semantic_cache_backend_factory(threshold=0.9)

    # Ensure the mock embeddings instance (used by the factory) is updated for this test
    # if new text values are introduced as semantic parts.
    mock_embeddings_instance.embedding_map["hello_semantic"] = embedding_A
    mock_embeddings_instance.embedding_map["paraphrased_hello_semantic"] = embedding_S_095

    @cacher(cache_backend=cache)
    def get_data_mixed_args(arg_int: int, arg_semantic_str: str, arg_bool: bool) -> str:
        call_counter.increment()
        return f"Data: {arg_int}, {arg_semantic_str}, {arg_bool}"

    # arg_semantic_str ("hello_semantic") will be the semantic_part.
    res1 = get_data_mixed_args(1, "hello_semantic", True)
    assert call_counter == 1
    assert len(cache.cache) == 1

    # Semantic hit on "hello_semantic" vs "paraphrased_hello_semantic" (sim 0.95 >= 0.90)
    # Other args (1, True) are the same. Should be a HIT.
    res2 = get_data_mixed_args(1, "paraphrased_hello_semantic", True)
    assert call_counter == 1 # HIT
    assert res2 == f"Data: 1, hello_semantic, True" # Returns original value

    # Semantic part similar, but non-semantic arg_bool changed: MISS
    res3 = get_data_mixed_args(1, "paraphrased_hello_semantic", False)
    assert call_counter == 2 # MISS
    assert len(cache.cache) == 2
    assert res3 == f"Data: 1, paraphrased_hello_semantic, False"

    # Semantic part similar, non-semantic arg_bool same as previous, but arg_int changed: MISS
    res4 = get_data_mixed_args(2, "paraphrased_hello_semantic", False)
    assert call_counter == 3 # MISS
    assert len(cache.cache) == 3
    assert res4 == f"Data: 2, paraphrased_hello_semantic, False"

    # Semantic part different, other args same as an existing entry: MISS
    mock_embeddings_instance.embedding_map["another_semantic_text"] = embedding_B
    res5 = get_data_mixed_args(1, "another_semantic_text", True)
    assert call_counter == 4 # MISS
    assert len(cache.cache) == 4
    assert res5 == f"Data: 1, another_semantic_text, True"


# --- Tests for SentenceEvaluatorSemanticCache ---
import logging # For caplog
# SentenceEvaluatorSemanticCache is already imported at the top if this is the same file
# from ragas.cache import SentenceEvaluatorSemanticCache
# BaseRagasEmbeddings is already imported

# Mock Embedding Model for SentenceEvaluatorSemanticCache
class MockSentenceEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embedding_map: Dict[str, np.ndarray]):
        self.embedding_map = embedding_map
        # Fallback for texts not in map, to avoid KeyError during tests if not all texts are pre-mapped
        self.unknown_embedding = np.array([0.001, 0.001, 0.001]) # Low magnitude, non-zero, normalized
        norm_unknown = np.linalg.norm(self.unknown_embedding)
        if norm_unknown > 0:
            self.unknown_embedding = self.unknown_embedding / norm_unknown


    def embed_query(self, text: str) -> List[float]:
        embedding = self.embedding_map.get(text)
        if embedding is None:
            return self.unknown_embedding.tolist()
        return embedding.tolist() # Ensure it's a list of floats

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def set_run_config(self, run_config): # Add if BaseRagasEmbeddings requires it
        pass

# Test Data and Embeddings for SentenceEvaluatorSemanticCache
# Using pre-normalized vectors for exact cosine similarities
s_emb_A = np.array([1.0, 0.0, 0.0])
s_emb_A_similar_high = np.array([0.9, np.sqrt(1 - 0.9**2), 0.0]) # CosSim with A is 0.9
s_emb_A_similar_low = np.array([0.7, np.sqrt(1 - 0.7**2), 0.0])  # CosSim with A is 0.7
s_emb_B = np.array([0.0, 1.0, 0.0]) # Orthogonal to A, CosSim = 0
s_emb_zeronorm = np.array([0.0, 0.0, 0.0]) # Zero-norm vector

s_text_A1 = "The cat sat on the mat."
s_text_A2_similar_high = "A feline was resting on the rug."
s_text_A3_similar_low = "The cat was on a mat."
s_text_B_dissimilar = "The dog barked loudly."
s_text_zeronorm = "text for zero norm sentence cache"

s_context1 = "location:house#section:livingroom"
s_context2 = "location:garden#section:lawn"

@pytest.fixture(scope="function")
def mock_sentence_embeddings_instance() -> MockSentenceEmbeddings:
    return MockSentenceEmbeddings({
        s_text_A1: s_emb_A,
        s_text_A2_similar_high: s_emb_A_similar_high,
        s_text_A3_similar_low: s_emb_A_similar_low,
        s_text_B_dissimilar: s_emb_B,
        s_text_zeronorm: s_emb_zeronorm
    })

@pytest.fixture(scope="function")
def sentence_cache_factory_fixture(mock_sentence_embeddings_instance: MockSentenceEmbeddings):
    # Renamed to avoid conflict if a non-fixture factory exists
    def _factory(threshold: float, embedding_model: Optional[BaseRagasEmbeddings] = mock_sentence_embeddings_instance) -> SentenceEvaluatorSemanticCache:
        # Allow passing None for embedding_model for specific tests
        cache = SentenceEvaluatorSemanticCache(
            embedding_model=embedding_model, # type: ignore
            similarity_threshold=threshold
        )
        cache.cache.clear()
        return cache
    return _factory


def test_s_cache_initialization(mock_sentence_embeddings_instance, caplog):
    SentenceEvaluatorSemanticCache(mock_sentence_embeddings_instance, similarity_threshold=0.5) # Valid
    with pytest.raises(ValueError):
        SentenceEvaluatorSemanticCache(mock_sentence_embeddings_instance, similarity_threshold=1.1)
    with pytest.raises(ValueError):
        SentenceEvaluatorSemanticCache(mock_sentence_embeddings_instance, similarity_threshold=-0.1)

    with caplog.at_level(logging.WARNING):
        SentenceEvaluatorSemanticCache(embedding_model=None, similarity_threshold=0.8) # type: ignore
    assert "initialized with no embedding model" in caplog.text


def test_s_cache_set_and_get_semantic_hit(sentence_cache_factory_fixture):
    cache = sentence_cache_factory_fixture(threshold=0.85) # Sim for A1 and A2_similar_high is 0.9

    cache.set(s_text_A1, s_context1, "result_A1")
    assert len(cache.cache) == 1

    result = cache.get(s_text_A2_similar_high, s_context1)
    assert result == "result_A1"

def test_s_cache_semantic_miss_dissimilar(sentence_cache_factory_fixture):
    cache = sentence_cache_factory_fixture(threshold=0.8)

    cache.set(s_text_A1, s_context1, "result_A1")
    result = cache.get(s_text_B_dissimilar, s_context1) # Sim is 0.0
    assert result is None

def test_s_cache_semantic_miss_threshold(sentence_cache_factory_fixture):
    # s_emb_A_similar_low has CosSim 0.7 with s_emb_A (s_text_A1)
    cache_high_thresh = sentence_cache_factory_fixture(threshold=0.75)
    cache_high_thresh.set(s_text_A1, s_context1, "result_A1_high")
    result_miss = cache_high_thresh.get(s_text_A3_similar_low, s_context1) # 0.7 < 0.75
    assert result_miss is None

    cache_low_thresh = sentence_cache_factory_fixture(threshold=0.65)
    cache_low_thresh.set(s_text_A1, s_context1, "result_A1_low")
    result_hit = cache_low_thresh.get(s_text_A3_similar_low, s_context1) # 0.7 >= 0.65
    assert result_hit == "result_A1_low"

    cache_exact_thresh = sentence_cache_factory_fixture(threshold=0.7)
    cache_exact_thresh.set(s_text_A1, s_context1, "result_A1_exact")
    result_exact_hit = cache_exact_thresh.get(s_text_A3_similar_low, s_context1) # 0.7 >= 0.7
    assert result_exact_hit == "result_A1_exact"


def test_s_cache_semantic_hit_context_miss(sentence_cache_factory_fixture):
    cache = sentence_cache_factory_fixture(threshold=0.85)
    cache.set(s_text_A1, s_context1, "result_A1_ctx1")

    result = cache.get(s_text_A2_similar_high, s_context2) # Different context
    assert result is None

def test_s_cache_exact_match_same_sentence_diff_context(sentence_cache_factory_fixture):
    cache = sentence_cache_factory_fixture(threshold=0.99) # High threshold

    cache.set(s_text_A1, s_context1, "result_A1_ctx1")
    cache.set(s_text_A1, s_context2, "result_A1_ctx2")

    assert cache.get(s_text_A1, s_context1) == "result_A1_ctx1"
    assert cache.get(s_text_A1, s_context2) == "result_A1_ctx2"

def test_s_cache_no_embedding_model(caplog):
    # Directly construct as factory expects a (potentially non-None) model by default
    cache = SentenceEvaluatorSemanticCache(embedding_model=None, similarity_threshold=0.8) # type: ignore

    with caplog.at_level(logging.WARNING):
        cache.set(s_text_A1, s_context1, "result_A1")
    assert "embedding model is not available" in caplog.text
    assert len(cache.cache) == 0

    assert cache.get(s_text_A1, s_context1) is None


class FailingMockSentenceEmbeddings(BaseRagasEmbeddings):
    def embed_query(self, text: str) -> List[float]:
        raise RuntimeError("Simulated embedding failure")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise RuntimeError("Simulated embedding failure")
    def set_run_config(self, run_config): pass


def test_s_cache_embedding_failure_get(sentence_cache_factory_fixture, caplog):
    # Set an item with a working model first
    cache = sentence_cache_factory_fixture(threshold=0.8)
    cache.set(s_text_A1, s_context1, "result_A1") # This uses mock_sentence_embeddings_instance

    # Now, simulate failure for the 'get' operation by replacing the model
    cache.embedding_model = FailingMockSentenceEmbeddings()

    with caplog.at_level(logging.ERROR):
        result = cache.get(s_text_A2_similar_high, s_context1)
    assert result is None
    assert "Failed to embed sentence for cache get" in caplog.text


def test_s_cache_embedding_failure_set(sentence_cache_factory_fixture, caplog):
    cache = sentence_cache_factory_fixture(threshold=0.8)
    cache.embedding_model = FailingMockSentenceEmbeddings() # Model will fail during set

    with caplog.at_level(logging.ERROR):
        cache.set(s_text_A1, s_context1, "result_A1")
    assert len(cache.cache) == 0
    assert "Failed to embed sentence for cache set" in caplog.text


def test_s_cache_zero_norm_embedding_handling(sentence_cache_factory_fixture, caplog):
    cache = sentence_cache_factory_fixture(threshold=0.8)
    # mock_sentence_embeddings_instance maps s_text_zeronorm to [0,0,0]

    # Test SET with zero-norm embedding
    # Current SentenceEvaluatorSemanticCache.set implementation *does* store zero-norm embeddings.
    # It only logs for zero-dimensional or empty, not zero-norm.
    cache.set(s_text_zeronorm, s_context1, "result_zero_norm")
    assert len(cache.cache) == 1
    # No warning expected on set for zero-norm, only for zero-size/dim.

    caplog.clear()
    # Test GET with a query sentence that has zero-norm embedding
    with caplog.at_level(logging.WARNING):
        result = cache.get(s_text_zeronorm, s_context1)
    assert result is None # `get` returns None if current_sentence embedding has zero norm
    assert "Skipping zero-norm embedding for sentence" in caplog.text
    assert f"Skipping zero-norm embedding for sentence: {s_text_zeronorm}" in caplog.text


    # Test GET with a normal query sentence, but a cached item has zero-norm embedding
    cache.cache.clear()
    caplog.clear()
    # Store an item with zero-norm embedding first
    cache.set(s_text_zeronorm, s_context1, "result_zero_norm_item")
    assert len(cache.cache) == 1

    # Now query with a normal sentence (s_text_A1)
    # The cache loop in `get` should skip the cached item with zero-norm embedding.
    result_normal_query = cache.get(s_text_A1, s_context1)
    assert result_normal_query is None # Should be a miss as the only cached item is skipped
    assert "Skipping zero-norm embedding for sentence" not in caplog.text # This log is for current_sentence
                                                                        # The loop just continues for cached zero-norm.
```
