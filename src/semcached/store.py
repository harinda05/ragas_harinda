from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
import logging
import numpy as np

# Define type for LLM evaluation results, can be any structure.
LLMResult = t.Any

logger = logging.getLogger(__name__)

class TestCasePartCacheStore(ABC):
    """
    Abstract base class defining the interface for storing and retrieving
    cached evaluations of test case parts.
    Organized by `test_case_id` and `part_id`.
    """

    @abstractmethod
    def get_exact_match(self, test_case_id: str, part_id: str) -> t.Optional[LLMResult]:
        """
        Retrieve based on exact match of test_case_id and part_id.
        """
        pass

    @abstractmethod
    def get_semantic_matches(
        self,
        test_case_id: str,
        part_embedding: np.ndarray,
        similarity_threshold: float,
        k: int = 1,
    ) -> t.List[t.Tuple[LLMResult, float]]:
        """
        Retrieve semantically similar items within a test_case_id.
        """
        pass

    @abstractmethod
    def add_item(
        self,
        test_case_id: str,
        part_id: str,
        part_embedding: np.ndarray,
        llm_result: LLMResult,
    ) -> None:
        """
        Add an item to the cache.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all items from the store.
        """
        pass

    @abstractmethod
    def clear_test_case(self, test_case_id: str) -> None:
        """Clear items for a specific test_case_id."""
        pass


class InMemoryStore(TestCasePartCacheStore):
    """
    An in-memory implementation of TestCasePartCacheStore that supports
    both exact and semantic matching.
    """

    def __init__(self, embedding_model: t.Optional[t.Any] = None, similarity_threshold: float = 0.85):
        # For exact matches: {test_case_id: {part_id: LLMResult}}
        self._exact_match_store: t.Dict[str, t.Dict[str, LLMResult]] = {}

        # For semantic matches: {test_case_id: [{part_id: str, embedding: np.ndarray, result: LLMResult}]}
        self._semantic_match_store: t.Dict[str, t.List[t.Dict[str, t.Any]]] = {}

        # Store the embedding model and threshold if provided (though not directly used in this specific store impl)
        # These would be more relevant if this store was also responsible for generating embeddings.
        # For now, it assumes embeddings are provided to `add_item` and `get_semantic_matches`.
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold # Default, can be overridden by get_semantic_matches call

        logger.info(f"InMemoryStore initialized. Embedding model: {'Provided' if embedding_model else 'Not Provided'}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Helper to calculate cosine similarity."""
        if vec1.size == 0 or vec2.size == 0: return 0.0
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
        dot_product = np.dot(vec1, vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def get_exact_match(self, test_case_id: str, part_id: str) -> t.Optional[LLMResult]:
        if test_case_id in self._exact_match_store and part_id in self._exact_match_store[test_case_id]:
            logger.debug(f"Exact match HIT for test_case_id='{test_case_id}', part_id='{part_id}'")
            return self._exact_match_store[test_case_id][part_id]
        else:
            logger.debug(f"Exact match MISS for test_case_id='{test_case_id}', part_id='{part_id}'")
            return None

    def get_semantic_matches(
        self,
        test_case_id: str,
        part_embedding: np.ndarray,
        similarity_threshold: float, # Threshold for this specific query
        k: int = 1,
    ) -> t.List[t.Tuple[LLMResult, float]]:

        if test_case_id not in self._semantic_match_store:
            logger.debug(f"No semantic entries for test_case_id='{test_case_id}'")
            return []

        candidate_items = self._semantic_match_store[test_case_id]
        matches: t.List[t.Tuple[LLMResult, float]] = []

        for item in candidate_items:
            similarity = self._cosine_similarity(item['embedding'], part_embedding)
            if similarity >= similarity_threshold:
                matches.append((item['result'], similarity))

        # Sort by similarity in descending order and take top k
        matches.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Found {len(matches)} semantic matches for test_case_id='{test_case_id}' above threshold {similarity_threshold}, returning top {k}")
        return matches[:k]

    def add_item(
        self,
        test_case_id: str,
        part_id: str,
        part_embedding: np.ndarray, # Embedding is now mandatory
        llm_result: LLMResult,
    ) -> None:
        # Add to exact match store
        if test_case_id not in self._exact_match_store:
            self._exact_match_store[test_case_id] = {}
        self._exact_match_store[test_case_id][part_id] = llm_result
        logger.debug(f"Added/updated exact match for test_case_id='{test_case_id}', part_id='{part_id}'")

        # Add to semantic match store
        if test_case_id not in self._semantic_match_store:
            self._semantic_match_store[test_case_id] = []

        # Avoid duplicate part_id entries in semantic store for the same test_case_id
        # If part_id exists, update it; otherwise, append.
        found_semantic_entry = False
        for entry in self._semantic_match_store[test_case_id]:
            if entry['part_id'] == part_id:
                entry['embedding'] = part_embedding
                entry['result'] = llm_result
                found_semantic_entry = True
                break
        if not found_semantic_entry:
            self._semantic_match_store[test_case_id].append({
                'part_id': part_id,
                'embedding': part_embedding,
                'result': llm_result
            })
        logger.debug(f"Added/updated semantic entry for test_case_id='{test_case_id}', part_id='{part_id}'")


    def clear(self) -> None:
        self._exact_match_store.clear()
        self._semantic_match_store.clear()
        logger.info("Cleared all items from InMemoryStore (exact and semantic).")

    def clear_test_case(self, test_case_id: str) -> None:
        cleared_exact = False
        if test_case_id in self._exact_match_store:
            del self._exact_match_store[test_case_id]
            cleared_exact = True

        cleared_semantic = False
        if test_case_id in self._semantic_match_store:
            del self._semantic_match_store[test_case_id]
            cleared_semantic = True

        if cleared_exact or cleared_semantic:
            logger.info(f"Cleared cache items for test_case_id='{test_case_id}' (exact: {cleared_exact}, semantic: {cleared_semantic}).")
        else:
            logger.debug(f"No cache items found to clear for test_case_id='{test_case_id}'.")
