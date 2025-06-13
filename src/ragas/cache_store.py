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
    In-memory implementation of TestCasePartCacheStore.
    Uses nested dictionaries and performs iterative semantic search.
    """
    def __init__(self):
        self.store: t.Dict[str, t.Dict[str, t.Dict[str, t.Any]]] = {}
        # Structure: self.store[test_case_id][part_id] = {'embedding': np.array, 'llm_result': LLMResult}
        logger.info("InMemoryStore initialized.")

    def get_exact_match(self, test_case_id: str, part_id: str) -> t.Optional[LLMResult]:
        if test_case_id in self.store and part_id in self.store[test_case_id]:
            logger.debug(f"Exact match found for test_case_id='{test_case_id}', part_id='{part_id[:50]}...'")
            return self.store[test_case_id][part_id]['llm_result']
        logger.debug(f"No exact match for test_case_id='{test_case_id}', part_id='{part_id[:50]}...'")
        return None

    def get_semantic_matches(
        self,
        test_case_id: str,
        part_embedding: np.ndarray,
        similarity_threshold: float,
        k: int = 1,
    ) -> t.List[t.Tuple[LLMResult, float]]:
        if test_case_id not in self.store:
            logger.debug(f"No items found for test_case_id='{test_case_id}' during semantic search.")
            return []

        # Ensure part_embedding is a valid numpy array before norm calculation
        if not isinstance(part_embedding, np.ndarray) or part_embedding.ndim == 0 or part_embedding.size == 0:
            logger.warning(f"Invalid or empty query embedding provided for semantic search in test_case_id='{test_case_id}'.")
            return []

        current_emb_flat = part_embedding.flatten() # Ensure 1D
        norm_part_embedding = np.linalg.norm(current_emb_flat)
        if norm_part_embedding == 0:
            logger.warning(f"Input part_embedding for semantic search has zero norm for test_case_id='{test_case_id}'.")
            return []
        normalized_part_embedding = current_emb_flat / norm_part_embedding

        matches: t.List[t.Tuple[LLMResult, float]] = []
        items_for_case = self.store[test_case_id]

        for _, item_data in items_for_case.items(): # Iterate through parts of the test case
            cached_embedding = item_data.get('embedding')
            if not isinstance(cached_embedding, np.ndarray) or cached_embedding.ndim == 0 or cached_embedding.size == 0:
                continue

            cached_emb_flat = cached_embedding.flatten() # Ensure 1D
            norm_cached_embedding = np.linalg.norm(cached_emb_flat)
            if norm_cached_embedding == 0:
                continue
            normalized_cached_embedding = cached_emb_flat / norm_cached_embedding

            if normalized_part_embedding.shape != normalized_cached_embedding.shape:
                logger.warning(f"Skipping semantic comparison due to mismatched embedding shapes for test_case_id='{test_case_id}'. Query shape {normalized_part_embedding.shape}, cached shape {normalized_cached_embedding.shape}")
                continue

            similarity = np.dot(normalized_part_embedding, normalized_cached_embedding)

            if similarity >= similarity_threshold:
                matches.append((item_data['llm_result'], similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Found {len(matches)} semantic matches for test_case_id='{test_case_id}' above threshold {similarity_threshold:.2f}, returning top {k}.")
        return matches[:k]

    def add_item(
        self,
        test_case_id: str,
        part_id: str,
        part_embedding: np.ndarray,
        llm_result: LLMResult,
    ) -> None:
        if not isinstance(part_embedding, np.ndarray):
            try:
                embedding_array = np.array(part_embedding, dtype=float)
            except Exception as e:
                logger.error(f"Failed to convert part_embedding to numpy array for test_case_id='{test_case_id}', part_id='{part_id[:50]}...'. Error: {e}")
                return
        else:
            embedding_array = part_embedding

        # Ensure embedding is flattened and not zero-norm before storing
        embedding_flat = embedding_array.flatten()
        if embedding_flat.ndim == 0 or embedding_flat.size == 0 or np.linalg.norm(embedding_flat) == 0:
             logger.warning(f"Skipping add_item for zero-dim, empty, or zero-norm embedding. test_case_id='{test_case_id}', part_id='{part_id[:50]}...'")
             return

        if test_case_id not in self.store:
            self.store[test_case_id] = {}

        self.store[test_case_id][part_id] = {
            'embedding': embedding_flat, # Store the flattened, valid embedding
            'llm_result': llm_result,
        }
        logger.debug(f"Added item for test_case_id='{test_case_id}', part_id='{part_id[:50]}...'")

    def clear(self) -> None:
        self.store.clear()
        logger.info("InMemoryStore cleared.")

    def clear_test_case(self, test_case_id: str) -> None:
        if test_case_id in self.store:
            del self.store[test_case_id]
            logger.info(f"Cleared items for test_case_id='{test_case_id}' from InMemoryStore.")
        else:
            logger.debug(f"No items found for test_case_id='{test_case_id}' to clear.")
