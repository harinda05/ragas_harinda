import pytest
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple

from ragas.cache_store import InMemoryStore # Actual class to test
from ragas.embeddings.base import BaseRagasEmbeddings # For type hinting and mock

# --- LLMResult Placeholder ---
LLMResult = Any # Using Any as a placeholder for LLMResult, adjust if defined elsewhere

# --- Mock Embedding Model for InMemoryStore Tests ---
class MockStoreEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embedding_map: Optional[Dict[str, np.ndarray]] = None):
        self.embedding_map = embedding_map if embedding_map is not None else {}
        # Default, non-zero, low magnitude, normalized vector for unknown texts
        self.unknown_embedding = normalize(np.array([0.001, 0.001, 0.001]))
        self.fail_on_text: Optional[str] = None
        self.empty_on_text: Optional[str] = None # Text that should yield empty list
        self.zero_norm_on_text: Optional[str] = None # Text that should yield zero-norm vector

    def set_embedding(self, text: str, embedding: np.ndarray):
        self.embedding_map[text] = embedding

    def embed_query(self, text: str) -> List[float]:
        if self.fail_on_text and text == self.fail_on_text:
            raise RuntimeError(f"Simulated embedding failure for text: {text}")
        if self.empty_on_text and text == self.empty_on_text:
            return []
        if self.zero_norm_on_text and text == self.zero_norm_on_text:
            return [0.0, 0.0, 0.0]

        embedding = self.embedding_map.get(text)
        if embedding is None:
            return self.unknown_embedding.tolist()
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def set_run_config(self, run_config): # From BaseRagasEmbeddings
        pass

# --- Test Data & Embeddings (Normalized) ---
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v # Return zero vector if norm is zero

# Embeddings for testing InMemoryStore (prefixed with 'im_')
im_emb_A_orig = np.array([1.0, 0.0, 0.0])
im_emb_A_similar_high_orig = np.array([0.9, 0.43589, 0.0])
im_emb_A_similar_low_orig = np.array([0.7, 0.71414, 0.0])
im_emb_B_orig = np.array([0.0, 1.0, 0.0])
im_emb_Z_orig = np.array([0.0, 0.0, 0.0])     # Zero-norm embedding

im_emb_A_norm = normalize(im_emb_A_orig)
im_emb_A_similar_high_norm = normalize(im_emb_A_similar_high_orig)
im_emb_A_similar_low_norm = normalize(im_emb_A_similar_low_orig)
im_emb_B_norm = normalize(im_emb_B_orig)
# im_emb_Z_orig remains zero-norm for testing

# Texts corresponding to embeddings
text_A = "Text A for embedding"
text_A_similar_high = "Text similar to A (high)"
text_A_similar_low = "Text similar to A (low)"
text_B = "Text B (dissimilar)"
text_Z = "Text for Zero Norm Embedding" # Will be mapped to emb_Z_orig by mock
text_empty = "Text for Empty Embedding" # Will result in empty list from mock
text_fail = "Text for Failing Embedding" # Will make mock raise error


@pytest.fixture
def mock_embeddings_for_store() -> MockStoreEmbeddings:
    model = MockStoreEmbeddings()
    model.set_embedding(text_A, im_emb_A_norm)
    model.set_embedding(text_A_similar_high, im_emb_A_similar_high_norm)
    model.set_embedding(text_A_similar_low, im_emb_A_similar_low_norm)
    model.set_embedding(text_B, im_emb_B_norm)
    # For text_Z, the mock will be configured to return zero_norm directly by name
    model.zero_norm_on_text = text_Z
    model.empty_on_text = text_empty
    model.fail_on_text = text_fail
    return model

@pytest.fixture
def im_store(mock_embeddings_for_store: MockStoreEmbeddings) -> InMemoryStore:
    """Provides a fresh InMemoryStore instance with a mock embedding model and default threshold."""
    return InMemoryStore(ragas_embedding_model=mock_embeddings_for_store, similarity_threshold=0.85)


# --- Test Cases for InMemoryStore (adapted to Turn 37 interface) ---

def test_im_initialization(mock_embeddings_for_store, caplog):
    with caplog.at_level(logging.INFO):
        store = InMemoryStore(ragas_embedding_model=mock_embeddings_for_store, similarity_threshold=0.75)
    assert not store._exact_match_store # Check internal state
    assert not store._semantic_store   # Check internal state
    assert "InMemoryStore initialized with threshold 0.75" in caplog.text

    with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
        InMemoryStore(ragas_embedding_model=mock_embeddings_for_store, similarity_threshold=1.2)
    with pytest.raises(ValueError, match="ragas_embedding_model cannot be None"):
        InMemoryStore(ragas_embedding_model=None, similarity_threshold=0.8) # type: ignore

def test_im_add_and_get_exact(im_store: InMemoryStore):
    # add_item(test_case_id, key, value, text_to_embed, embedding=None)
    # Using embedding parameter to provide pre-normalized embedding directly for this test part
    im_store.add_item("case1", "part1", "result1", text_to_embed="placeholder for text_A", embedding=im_emb_A_norm.tolist())

    assert im_store.get_exact_match("case1", "part1") == "result1"
    assert im_store.get_exact_match("case1", "part_unknown") is None
    assert im_store.get_exact_match("case_unknown", "part1") is None

    assert len(im_store._semantic_store) == 1
    item = im_store._semantic_store[0]
    assert item["test_case_id"] == "case1" and item["key"] == "part1"
    assert np.allclose(item["embedding"], im_emb_A_norm)


def test_im_add_item_invalid_embedding_types(im_store: InMemoryStore, caplog):
    # Test behavior when a precomputed embedding is of an unexpected type
    caplog.clear()
    # np.array on dict will raise TypeError
    with pytest.raises(TypeError): # Or other relevant error depending on np.array behavior with dicts
        im_store.add_item("case1", "part_bad_emb1", "result_bad_emb1", text_to_embed="text_for_bad_emb", embedding={"not": "a_list_of_float"}) # type: ignore

    # The item should still be added to exact match store because embedding failure happens after exact match set
    assert im_store.get_exact_match("case1", "part_bad_emb1") == "result_bad_emb1"
    # No semantic entry should be made if embedding processing fails.
    assert not any(item["key"] == "part_bad_emb1" for item in im_store._semantic_store)


    # Good precomputed embedding (list of floats)
    im_store.add_item("case1", "part_good_emb_list", "result_good_emb_list", text_to_embed="text_for_good_list_emb", embedding=[0.1, 0.2, 0.3])
    good_item = next(item for item in im_store._semantic_store if item["key"] == "part_good_emb_list")
    assert isinstance(good_item['embedding'], np.ndarray)
    assert np.allclose(good_item['embedding'], normalize(np.array([0.1,0.2,0.3]))) # add_item normalizes
    assert im_store.get_exact_match("case1", "part_good_emb_list") == "result_good_emb_list"

    # Test with text_to_embed that results in zero-norm embedding (from mock)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        im_store.add_item("case1", "part_zero_dim_emb_from_text", "result_zero_dim_from_text", text_to_embed=text_Z)
    assert "Zero-norm embedding generated" in caplog.text
    assert im_store.get_exact_match("case1", "part_zero_dim_emb_from_text") == "result_zero_dim_from_text"
    zero_item = next(i for i in im_store._semantic_store if i["key"] == "part_zero_dim_emb_from_text")
    assert np.all(zero_item["embedding"] == 0) # Check it stored zero vector


def test_im_get_semantic_hit(im_store: InMemoryStore):
    im_store.add_item("case1", "partA", "resultA", text_to_embed=text_A) # text_A maps to emb_A_norm
    # text_A_similar_high maps to emb_A_similar_high_norm. sim(A_norm, A_similar_high_norm) ~ 0.9
    matches = im_store.get_semantic_matches(query_text=text_A_similar_high, test_case_id="case1", top_k=1)
    assert len(matches) == 1
    assert matches[0][0] == "resultA"
    assert matches[0][1] == pytest.approx(np.dot(im_emb_A_norm, im_emb_A_similar_high_norm))

def test_im_get_semantic_miss_threshold(im_store: InMemoryStore):
    im_store.similarity_threshold = 0.85 # Store's threshold
    im_store.add_item("case1", "partA", "resultA", text_to_embed=text_A)
    # text_A_similar_low maps to emb_A_similar_low_norm. sim(A_norm, A_similar_low_norm) ~ 0.7
    matches = im_store.get_semantic_matches(query_text=text_A_similar_low, test_case_id="case1", top_k=1)
    assert len(matches) == 0

def test_im_get_semantic_miss_different_case_id(im_store: InMemoryStore):
    im_store.add_item("case1", "partA", "resultA", text_to_embed=text_A)
    matches = im_store.get_semantic_matches(query_text=text_A, test_case_id="case2_is_different", top_k=1)
    assert len(matches) == 0

def test_im_get_semantic_top_k(im_store: InMemoryStore):
    im_store.similarity_threshold = 0.65 # Lower threshold
    # Values are text_A (sim 1.0 to text_A), text_A_similar_high (sim ~0.9), text_A_similar_low (sim ~0.7)
    im_store.add_item("case1", "item_low", "result_low", text_to_embed=text_A_similar_low)
    im_store.add_item("case1", "item_exact", "result_exact", text_to_embed=text_A)
    im_store.add_item("case1", "item_high", "result_high", text_to_embed=text_A_similar_high)

    matches_k1 = im_store.get_semantic_matches(query_text=text_A, test_case_id="case1", top_k=1)
    assert len(matches_k1) == 1
    assert matches_k1[0][0] == "result_exact"

    matches_k2 = im_store.get_semantic_matches(query_text=text_A, test_case_id="case1", top_k=2)
    assert len(matches_k2) == 2
    assert matches_k2[0][0] == "result_exact"
    assert matches_k2[1][0] == "result_high" # text_A_similar_high is next most similar
    assert matches_k2[0][1] >= matches_k2[1][1] # Check scores are ordered

    matches_all = im_store.get_semantic_matches(query_text=text_A, test_case_id="case1", top_k=3)
    assert len(matches_all) == 3
    assert {m[0] for m in matches_all} == {"result_exact", "result_high", "result_low"}
    assert matches_all[0][1] >= matches_all[1][1] >= matches_all[2][1]


def test_im_semantic_search_query_embedding_issues(im_store: InMemoryStore, caplog):
    im_store.add_item("case1", "partA", "resultA", text_to_embed=text_A)

    # 1. Query text gives zero-norm embedding
    with caplog.at_level(logging.WARNING):
      matches_zero_norm = im_store.get_semantic_matches(query_text=text_Z, test_case_id="case1") # text_Z gives zero-norm via mock
    assert len(matches_zero_norm) == 0
    assert "produced a zero-norm embedding" in caplog.text
    caplog.clear()

    # 2. Query text makes embedding model fail
    with caplog.at_level(logging.ERROR):
      matches_fail = im_store.get_semantic_matches(query_text=text_fail, test_case_id="case1")
    assert len(matches_fail) == 0
    assert "Failed to embed query text" in caplog.text
    caplog.clear()

    # 3. Query text gives empty list embedding
    with caplog.at_level(logging.WARNING):
        matches_empty = im_store.get_semantic_matches(query_text=text_empty, test_case_id="case1")
    assert len(matches_empty) == 0
    assert "produced an empty embedding" in caplog.text


def test_im_add_item_and_search_stored_zero_norm_embedding(im_store: InMemoryStore, caplog):
    # Add item that will have zero-norm embedding
    with caplog.at_level(logging.WARNING):
        im_store.add_item("case1", "partZ", "resultZ_zero_norm", text_to_embed=text_Z)
    # This log comes from InMemoryStore.add_item when it normalizes the embedding
    assert "Zero-norm embedding generated" in caplog.text
    assert im_store.get_exact_match("case1", "partZ") == "resultZ_zero_norm" # Exact match still stored

    # Add a normal item
    im_store.add_item("case1", "partA", "resultA", text_to_embed=text_A)

    # Search with a normal query text.
    # The zero-norm item in _semantic_store should be skipped by get_semantic_matches's loop
    # because its cached_embedding.size will be > 0 but norm_cached_embedding will be 0, leading to 0 similarity.
    matches = im_store.get_semantic_matches(query_text=text_A_similar_high, test_case_id="case1", top_k=2)
    assert len(matches) == 1
    assert matches[0][0] == "resultA"


def test_im_clear(im_store: InMemoryStore):
    im_store.add_item("case1", "part1", "result1", text_to_embed=text_A)
    im_store.add_item("case2", "part1", "result2", text_to_embed=text_B)
    assert len(im_store._exact_match_store) == 2
    assert len(im_store._semantic_store) == 2

    im_store.clear()
    assert not im_store._exact_match_store
    assert not im_store._semantic_store

def test_im_clear_test_case(im_store: InMemoryStore):
    im_store.add_item("case1", "part1", "result1_c1", text_to_embed=text_A)
    im_store.add_item("case2", "part1", "result1_c2", text_to_embed=text_B)

    im_store.clear_test_case("case1")
    assert ("case1", "part1") not in im_store._exact_match_store
    assert ("case2", "part1") in im_store._exact_match_store
    assert len(im_store._semantic_store) == 1
    assert im_store._semantic_store[0]["test_case_id"] == "case2"

```
