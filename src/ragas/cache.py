import json # Added import
from abc import ABC, abstractmethod
from typing import Any, List, Optional # Added Optional

import numpy as np


class CacheInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def has_key(self, key: str) -> bool:
        ...


class SemanticCacheBackend(CacheInterface):
    def __init__(self, embedding_model: Any, similarity_threshold: float):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        # Each item in the cache will be a dictionary:
        # {"embedding": np.ndarray, "original_key_str": str, "value": Any, "semantic_arg_index": int}
        self.cache: List[dict[str, Any]] = []

    def _parse_key_str(self, key_str: str) -> Optional[dict[str, Any]]:
        """Helper to parse JSON key string and identify semantic part."""
        try:
            parsed_key = json.loads(key_str)
            if not all(k in parsed_key for k in ["function", "args", "kwargs"]):
                # Log or handle malformed key
                return None
            
            semantic_arg_index = -1
            semantic_part = None
            for i, arg in enumerate(parsed_key["args"]):
                if isinstance(arg, str):
                    semantic_part = arg
                    semantic_arg_index = i
                    break # Found first string arg
            
            if semantic_part is None:
                # No string argument found for semantic comparison
                # Or handle as exact match only? For now, treat as no semantic component.
                return None 

            parsed_key["semantic_part"] = semantic_part
            parsed_key["semantic_arg_index"] = semantic_arg_index
            return parsed_key
        except json.JSONDecodeError:
            # Log or handle error
            return None

    def _compare_non_semantic_parts(
        self, 
        input_parsed_key: dict, 
        stored_parsed_key: dict, 
        semantic_arg_idx: int
    ) -> bool:
        """Compares non-semantic parts of parsed keys."""
        if input_parsed_key["function"] != stored_parsed_key["function"]:
            return False
        if input_parsed_key["kwargs"] != stored_parsed_key["kwargs"]:
            return False
        
        input_args = input_parsed_key["args"]
        stored_args = stored_parsed_key["args"]
        if len(input_args) != len(stored_args):
            return False
            
        for i in range(len(input_args)):
            if i == semantic_arg_idx:
                continue # Skip semantic part, already matched by similarity
            if input_args[i] != stored_args[i]:
                return False
        return True

    def get(self, key_str: str) -> Any:
        if not self.cache:
            return None

        input_parsed_key = self._parse_key_str(key_str)
        if not input_parsed_key or "semantic_part" not in input_parsed_key:
            # Cannot perform semantic search if key is invalid or no semantic part
            return None 
        
        input_semantic_part = input_parsed_key["semantic_part"]
        input_semantic_arg_idx = input_parsed_key["semantic_arg_index"]
        input_embedding = np.array(self.embedding_model.embed_query(input_semantic_part))
        
        if input_embedding.size == 0: # Should not happen with valid models
             return None

        for cached_item in self.cache:
            # Skip if semantic arg index doesn't match type of comparison
            if cached_item["semantic_arg_index"] != input_semantic_arg_idx:
                continue

            stored_embedding = cached_item["embedding"]
            
            norm_input_embedding = np.linalg.norm(input_embedding)
            norm_stored_embedding = np.linalg.norm(stored_embedding)

            if norm_input_embedding == 0 or norm_stored_embedding == 0:
                similarity = 0.0
            else:
                similarity = np.dot(input_embedding, stored_embedding) / \
                             (norm_input_embedding * norm_stored_embedding)

            if similarity >= self.similarity_threshold:
                stored_parsed_key = self._parse_key_str(cached_item["original_key_str"])
                if not stored_parsed_key: 
                    continue # Should not happen if cache is consistent

                if self._compare_non_semantic_parts(input_parsed_key, stored_parsed_key, input_semantic_arg_idx):
                    return cached_item["value"]
        
        return None

    def set(self, key_str: str, value: Any) -> None:
        parsed_key = self._parse_key_str(key_str)
        if not parsed_key or "semantic_part" not in parsed_key:
            # Cannot cache if key is invalid or no semantic part to embed
            # Optionally, log this event
            return

        semantic_part = parsed_key["semantic_part"]
        semantic_arg_index = parsed_key["semantic_arg_index"]
        
        embedding = np.array(self.embedding_model.embed_query(semantic_part))
        if embedding.size == 0: # Embedding failed
            return

        self.cache.append(
            {
                "embedding": embedding,
                "original_key_str": key_str,
                "value": value,
                "semantic_arg_index": semantic_arg_index,
            }
        )

    def has_key(self, key_str: str) -> bool:
        if not self.cache:
            return False

        input_parsed_key = self._parse_key_str(key_str)
        if not input_parsed_key or "semantic_part" not in input_parsed_key:
            return False

        input_semantic_part = input_parsed_key["semantic_part"]
        input_semantic_arg_idx = input_parsed_key["semantic_arg_index"]
        input_embedding = np.array(self.embedding_model.embed_query(input_semantic_part))

        if input_embedding.size == 0:
            return False
            
        for cached_item in self.cache:
            if cached_item["semantic_arg_index"] != input_semantic_arg_idx:
                continue

            stored_embedding = cached_item["embedding"]

            norm_input_embedding = np.linalg.norm(input_embedding)
            norm_stored_embedding = np.linalg.norm(stored_embedding)

            if norm_input_embedding == 0 or norm_stored_embedding == 0:
                similarity = 0.0
            else:
                similarity = np.dot(input_embedding, stored_embedding) / \
                             (norm_input_embedding * norm_stored_embedding)

            if similarity >= self.similarity_threshold:
                stored_parsed_key = self._parse_key_str(cached_item["original_key_str"])
                if not stored_parsed_key:
                    continue
                
                if self._compare_non_semantic_parts(input_parsed_key, stored_parsed_key, input_semantic_arg_idx):
                    return True
        
        return False


# -- Standard Disk Cache (Exact Matching) --
import os
import pickle
import hashlib
import functools
from pathlib import Path

# Attempt to import ragas_cache from ragas.config
# If this file is ragas/cache.py, then ragas.config should be .config
try:
    from ragas.config import ragas_cache
except ImportError:
    # Fallback for environments where direct import might fail (e.g. isolated testing)
    # or if ragas.config is not yet fully set up during initial load.
    # In a fully integrated system, ragas_cache should be available.
    print(
        "Warning: ragas.config.ragas_cache could not be imported directly in cache.py. "
        "The cacher decorator might not use the global cache by default if not explicitly passed."
    )
    ragas_cache = None


class DiskCacheBackend(CacheInterface):
    def __init__(self, cache_dir: str = ".ragas_cache/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, key: str) -> Path:
        # Hash the key to create a filename
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / hashed_key

    def get(self, key: str) -> Any:
        filepath = self._get_filepath(key)
        if filepath.exists():
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, key: str, value: Any) -> None:
        filepath = self._get_filepath(key)
        with open(filepath, "wb") as f:
            pickle.dump(value, f)

    def has_key(self, key: str) -> bool:
        filepath = self._get_filepath(key)
        return filepath.exists()


# Sentinel object to detect if cache_backend was explicitly passed
_DEFAULT_CACHE_SENTINEL = object()


def cacher(cache_backend: Optional[CacheInterface] = _DEFAULT_CACHE_SENTINEL):
    """
    Decorator to cache the results of a function call.
    Uses the global `ragas.config.ragas_cache` by default if no specific
    `cache_backend` is provided.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine which cache backend to use
            actual_cache_backend = cache_backend
            if actual_cache_backend is _DEFAULT_CACHE_SENTINEL:
                actual_cache_backend = ragas_cache  # Use global cache

            if actual_cache_backend is None:
                # Cache is explicitly disabled or global cache is None (disabled)
                return func(*args, **kwargs)

            # Ensure the chosen backend is a CacheInterface instance
            # This check is more for robustness, assuming ragas_cache is correctly typed
            if not isinstance(actual_cache_backend, CacheInterface):
                 # Log a warning or raise an error if the backend is invalid
                 print(f"Warning: Invalid cache backend type: {type(actual_cache_backend)}. Caching disabled for this call.")
                 return func(*args, **kwargs)

            # Create a cache key from the function name, args, and kwargs
            # Using json.dumps for robust serialization of args/kwargs
            key_dict = {
                "function": func.__name__,
                "args": args,
                "kwargs": sorted(kwargs.items()), # Sort kwargs for consistent key
            }
            try:
                key = json.dumps(key_dict)
            except TypeError:
                # If args/kwargs are not JSON serializable, fall back to a simpler key
                # or raise an error. For simplicity, we'll skip caching here.
                # In a real-world scenario, more sophisticated key generation might be needed.
                print(
                    f"Warning: Could not serialize arguments for {func.__name__} to create cache key. "
                    "Skipping cache for this call."
                )
                return func(*args, **kwargs)


            if actual_cache_backend.has_key(key):
                return actual_cache_backend.get(key)
            else:
                result = func(*args, **kwargs)
                actual_cache_backend.set(key, result)
                return result

        return wrapper

    return decorator

# Imports for SentenceEvaluatorSemanticCache
import typing as t # Already have Optional, Any, List from top of file. Adding Dict.
# numpy as np is already imported
import logging
from ragas.embeddings.base import BaseRagasEmbeddings # Assuming this path is correct from project structure

logger = logging.getLogger(__name__)

class SentenceEvaluatorSemanticCache:
    """
    A semantic cache specific to sentence evaluation within metrics.
    This cache stores verdicts or results for individual sentences based on their
    semantic similarity to previously evaluated sentences, within a primary context 
    and optionally matching a secondary context hash.
    """
    def __init__(self, embedding_model: BaseRagasEmbeddings, similarity_threshold: float):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache: t.List[t.Dict[str, t.Any]] = [] # Cache stores dicts
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.embedding_model is None: # Note: Type hint is BaseRagasEmbeddings, not Optional
            logger.warning("SentenceEvaluatorSemanticCache initialized with no embedding model. Cache will be ineffective.")

    def get(
        self,
        current_sentence: str,
        primary_context_or_hash: str, # Renamed from primary_context for clarity
        secondary_context_or_hash: t.Optional[str] = None,
    ) -> t.Optional[t.Any]:
        if not self.cache or self.embedding_model is None:
            return None

        current_full_key = ( # For exact matching of context parts
            current_sentence, 
            primary_context_or_hash, 
            secondary_context_or_hash, 
        )
        
        try:
            current_sentence_embedding_list = self.embedding_model.embed_query(current_sentence)
            current_sentence_embedding = np.array(current_sentence_embedding_list)
        except Exception as e:
            logger.error(f"Failed to embed sentence for cache get ['{current_sentence}']: {e}")
            return None 

        # Ensure embedding is 1D for consistent processing
        if current_sentence_embedding.ndim > 1:
            current_sentence_embedding = current_sentence_embedding.squeeze()
        if current_sentence_embedding.ndim == 0 or current_sentence_embedding.size == 0:
            logger.warning(f"Skipping zero-dimensional or empty embedding for sentence: {current_sentence}")
            return None

        norm_current_embedding = np.linalg.norm(current_sentence_embedding)
        if norm_current_embedding == 0:
            logger.warning(f"Skipping zero-norm embedding for sentence: {current_sentence}")
            return None 
        normalized_current_embedding = current_sentence_embedding / norm_current_embedding

        for item in self.cache:
            cached_embedding = item.get("key_embedding")
            if cached_embedding is None or not isinstance(cached_embedding, np.ndarray):
                continue
            
            # Ensure cached_embedding is 1D
            if cached_embedding.ndim > 1:
                cached_embedding = cached_embedding.squeeze()
            if cached_embedding.ndim == 0 or cached_embedding.size == 0:
                continue

            norm_cached_embedding = np.linalg.norm(cached_embedding)
            if norm_cached_embedding == 0:
                continue 

            normalized_cached_embedding = cached_embedding / norm_cached_embedding
            
            similarity = np.dot(normalized_current_embedding, normalized_cached_embedding)

            if similarity >= self.similarity_threshold:
                cached_full_key = item.get("full_key", (None, None, None))
                # Compare context parts (index 1 and 2 of full_key)
                if (
                    primary_context_or_hash == cached_full_key[1] and # Direct comparison of primary context
                    secondary_context_or_hash == cached_full_key[2] # Direct comparison of secondary context
                ):
                    logger.debug(f"Semantic cache GET hit for sentence: {current_sentence}")
                    return item.get("value")
        return None

    def set(
        self,
        sentence_to_cache: str,
        primary_context_or_hash: str, # Renamed from primary_context
        llm_output_for_sentence: t.Any,
        secondary_context_or_hash: t.Optional[str] = None,
    ):
        if self.embedding_model is None:
            logger.warning("Cannot set cache item: embedding model is not available.")
            return

        try:
            sentence_embedding_list = self.embedding_model.embed_query(sentence_to_cache)
            sentence_embedding = np.array(sentence_embedding_list)
            # Ensure embedding is stored as a 1D array
            if sentence_embedding.ndim > 1:
                sentence_embedding = sentence_embedding.squeeze()
            if sentence_embedding.ndim == 0 or sentence_embedding.size == 0:
                logger.warning(f"Not caching zero-dimensional or empty embedding for sentence: {sentence_to_cache}")
                return
        except Exception as e:
            logger.error(f"Failed to embed sentence for cache set ['{sentence_to_cache}']: {e}")
            return 

        full_key = ( # Store the full context for precise retrieval if needed, and for clarity
            sentence_to_cache, # Though semantic, storing it can be useful for debugging/inspection
            primary_context_or_hash,
            secondary_context_or_hash,
        )
        
        self.cache.append({
            "key_embedding": sentence_embedding, 
            "full_key": full_key, # Store for context matching
            "value": llm_output_for_sentence,
        })
        logger.debug(f"Semantic cache SET for sentence: {sentence_to_cache}")


# --- ComprehensiveSemanticCache Implementation ---
# Assuming LLMResult and TestCasePartCacheStore are correctly importable
# For LLMResult, if it's a custom Pydantic model or dataclass, its definition would be needed.
# For this task, we'll assume it's a generic type or defined elsewhere.
# If TestCasePartCacheStore is in a different file, adjust import.
try:
    from ragas.cache_store import TestCasePartCacheStore # Assuming this is the location
except ImportError:
    # Create a dummy placeholder if not found, so the class can be defined.
    # This should be resolved by ensuring cache_store.py is correctly populated and importable.
    logger.error("TestCasePartCacheStore not found. ComprehensiveSemanticCache may not function correctly.")
    class TestCasePartCacheStore(ABC): # type: ignore
        def get(self, test_case_id: str, key: str) -> Optional[Any]: return None
        def set(self, test_case_id: str, key: str, value: Any) -> None: pass
        def has_key(self, test_case_id: str, key: str) -> bool: return False
        def clear(self, test_case_id: Optional[str] = None) -> None: pass


# Define LLMResult as a generic type for this context if not already defined/imported.
# In a real scenario, this would be a specific Pydantic model or dataclass.
if "LLMResult" not in t.TYPE_CHECKING_CONTEXT.globals: # Check if already available via typing
    LLMResult = t.TypeVar('LLMResult') # Generic type, replace with actual if available


class ComprehensiveSemanticCache:
    """
    A comprehensive semantic cache that uses a TestCasePartCacheStore for persistence
    and BaseRagasEmbeddings for semantic similarity comparisons.
    """

    CACHE_KEY_PREFIX_PROMPT = "prompt_embedding"
    CACHE_KEY_PREFIX_RESPONSE = "response_text" # For exact match of response

    def __init__(
        self,
        store: TestCasePartCacheStore,
        embedding_model: BaseRagasEmbeddings,
        similarity_threshold: float = 0.8,
        logger_level: int = logging.INFO, # Changed from logger to logger_level
    ):
        self.store = store
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logger_level)

        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.embedding_model is None:
            self.logger.warning("ComprehensiveSemanticCache initialized with no embedding model. Semantic caching will be ineffective.")
        if self.store is None:
            self.logger.warning("ComprehensiveSemanticCache initialized with no store. Caching will be ineffective.")


    def _get_prompt_embedding_key(self) -> str:
        return self.CACHE_KEY_PREFIX_PROMPT

    def _get_response_text_key(self) -> str:
        return self.CACHE_KEY_PREFIX_RESPONSE

    def _calculate_similarity(
        self, emb1: np.ndarray, emb2: np.ndarray
    ) -> float:
        """Calculates cosine similarity between two embeddings."""
        # Ensure embeddings are 1D numpy arrays
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()

        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)

        if norm_emb1 == 0 or norm_emb2 == 0:
            return 0.0  # Similarity is 0 if one or both embeddings are zero vectors
        
        # Manually compute dot product for clarity if dimensions are mismatched (though ideally they shouldn't be)
        if emb1.shape != emb2.shape:
            self.logger.warning(f"Embedding shape mismatch: {emb1.shape} vs {emb2.shape}. Returning 0 similarity.")
            return 0.0

        similarity = np.dot(emb1, emb2) / (norm_emb1 * norm_emb2)
        return float(similarity)


    def get(self, test_case_id: str, current_prompt: str) -> Optional[LLMResult]:
        """
        Retrieves an LLMResult from the cache based on semantic similarity of the prompt.
        """
        if self.embedding_model is None or self.store is None:
            self.logger.debug(f"[{test_case_id}] Cache GET: Embedding model or store is None. Skipping.")
            return None

        prompt_embedding_key = self._get_prompt_embedding_key()
        cached_prompt_embedding_serial = self.store.get(test_case_id, prompt_embedding_key)

        if cached_prompt_embedding_serial is None:
            self.logger.debug(f"[{test_case_id}] Cache GET: No cached prompt embedding found.")
            return None
        
        # Assuming cached_prompt_embedding_serial is a list of floats
        cached_prompt_embedding = np.array(cached_prompt_embedding_serial)

        try:
            current_prompt_embedding_list = self.embedding_model.embed_query(current_prompt)
            current_prompt_embedding = np.array(current_prompt_embedding_list)
        except Exception as e:
            self.logger.error(f"[{test_case_id}] Cache GET: Failed to embed current prompt '{current_prompt[:50]}...': {e}")
            return None
        
        if current_prompt_embedding.ndim == 0 or current_prompt_embedding.size == 0:
            self.logger.warning(f"[{test_case_id}] Cache GET: Skipping zero-dimensional or empty embedding for prompt: {current_prompt[:50]}...")
            return None
        if cached_prompt_embedding.ndim == 0 or cached_prompt_embedding.size == 0:
            self.logger.warning(f"[{test_case_id}] Cache GET: Skipping zero-dimensional or empty cached prompt embedding.")
            return None

        similarity = self._calculate_similarity(current_prompt_embedding, cached_prompt_embedding)
        self.logger.debug(f"[{test_case_id}] Cache GET: Similarity for prompt '{current_prompt[:50]}...' is {similarity:.4f}")

        if similarity >= self.similarity_threshold:
            response_text_key = self._get_response_text_key()
            cached_response = self.store.get(test_case_id, response_text_key)
            if cached_response is not None:
                # Assuming cached_response is already in the correct LLMResult format or can be cast.
                # For this example, we'll assume it's directly usable.
                # If LLMResult is a Pydantic model, you might need to parse/validate here.
                self.logger.info(f"[{test_case_id}] Cache GET: HIT for prompt '{current_prompt[:50]}...' (Similarity: {similarity:.4f})")
                return t.cast(LLMResult, cached_response) # Use cast if LLMResult is TypeVar
            else:
                self.logger.warning(f"[{test_case_id}] Cache GET: Prompt embedding match, but no response found for key '{response_text_key}'.")
        else:
            self.logger.info(f"[{test_case_id}] Cache GET: MISS for prompt '{current_prompt[:50]}...' (Similarity: {similarity:.4f} < Threshold: {self.similarity_threshold:.4f})")
        
        return None


    def set(self, test_case_id: str, prompt: str, llm_result: LLMResult) -> None:
        """
        Stores the prompt embedding and the LLMResult in the cache.
        """
        if self.embedding_model is None or self.store is None:
            self.logger.debug(f"[{test_case_id}] Cache SET: Embedding model or store is None. Skipping.")
            return

        try:
            prompt_embedding_list = self.embedding_model.embed_query(prompt)
            # Ensure prompt_embedding_list is stored as a simple list of floats for broader compatibility (e.g. JSON)
            if isinstance(prompt_embedding_list, np.ndarray):
                prompt_embedding_list = prompt_embedding_list.flatten().tolist() 
            
            if not prompt_embedding_list or (isinstance(prompt_embedding_list, list) and not prompt_embedding_list[0]): # Check for empty or [[None]]
                 self.logger.warning(f"[{test_case_id}] Cache SET: Received empty or invalid embedding for prompt '{prompt[:50]}...'. Skipping cache set.")
                 return

        except Exception as e:
            self.logger.error(f"[{test_case_id}] Cache SET: Failed to embed prompt '{prompt[:50]}...': {e}")
            return

        prompt_embedding_key = self._get_prompt_embedding_key()
        response_text_key = self._get_response_text_key()

        self.store.set(test_case_id, prompt_embedding_key, prompt_embedding_list)
        # Assuming llm_result is directly serializable by the store.
        # If LLMResult is a Pydantic model, it should serialize to dict for many stores.
        self.store.set(test_case_id, response_text_key, llm_result)
        self.logger.info(f"[{test_case_id}] Cache SET: Stored prompt embedding and response for prompt '{prompt[:50]}...'")

    def clear(self, test_case_id: Optional[str] = None) -> None:
        """
        Clears cache entries. If test_case_id is provided, clears only for that ID.
        Otherwise, clears all entries managed by this cache instance (by clearing underlying store).
        """
        if self.store is None:
            self.logger.debug("Cache CLEAR: Store is None. Skipping.")
            return
            
        if test_case_id is not None:
            # This will clear all parts for the test_case_id, including prompt and response.
            self.store.clear(test_case_id)
            self.logger.info(f"[{test_case_id}] Cache CLEAR: Cleared entries for test case ID.")
        else:
            # This assumes the store's clear(None) clears everything it holds.
            # If this cache instance should only clear its *own* prefixed keys from a shared store,
            # this logic would need to be more granular (e.g., iterate and delete specific keys).
            # For now, deferring to store.clear() behavior.
            self.store.clear() 
            self.logger.info("Cache CLEAR: Cleared all entries in the underlying store.")

    def has_key(self, test_case_id: str, current_prompt: Optional[str] = None) -> bool:
        """
        Checks if a relevant cache entry exists for the test_case_id.
        If current_prompt is provided, it performs a semantic check similar to get().
        If current_prompt is None, it checks if any cache data (prompt embedding or response) exists.
        """
        if self.store is None:
            return False

        prompt_key = self._get_prompt_embedding_key()
        response_key = self._get_response_text_key()

        if current_prompt is None:
            # Check if either prompt embedding or response exists
            return self.store.has_key(test_case_id, prompt_key) or \
                   self.store.has_key(test_case_id, response_key)
        else:
            # Perform semantic check if prompt is provided
            if self.embedding_model is None:
                return False # Cannot perform semantic check

            cached_prompt_embedding_serial = self.store.get(test_case_id, prompt_key)
            if cached_prompt_embedding_serial is None:
                return False
            
            cached_prompt_embedding = np.array(cached_prompt_embedding_serial)
            try:
                current_prompt_embedding = np.array(self.embedding_model.embed_query(current_prompt))
            except Exception:
                return False # Cannot embed, so cannot compare

            if current_prompt_embedding.size == 0 or cached_prompt_embedding.size == 0:
                return False # Cannot compare empty embeddings

            similarity = self._calculate_similarity(current_prompt_embedding, cached_prompt_embedding)
            if similarity >= self.similarity_threshold:
                # Also ensure the response part exists for a full "hit"
                return self.store.has_key(test_case_id, response_key)
            return False
