from __future__ import annotations

import json
from abc import ABC, abstractmethod
import typing as t
from typing import Any, List, Optional

import numpy as np
import logging
import hashlib

from ragas.embeddings.base import BaseRagasEmbeddings
try:
    from ragas.cache_store import TestCasePartCacheStore, LLMResult
except ImportError:
    logger_for_imports = logging.getLogger(__name__)
    logger_for_imports.error(
        "TestCasePartCacheStore or LLMResult not found. ComprehensiveSemanticCache may not function correctly. "
        "Ensure ragas.cache_store is correctly defined and importable."
    )
    class TestCasePartCacheStore(ABC): # type: ignore
        @abstractmethod
        def get(self, test_case_id: str, key: str) -> Optional[Any]: ...
        @abstractmethod
        def set(self, test_case_id: str, key: str, value: Any) -> None: ...
        @abstractmethod
        def has_key(self, test_case_id: str, key: str) -> bool: ...
        @abstractmethod
        def clear(self, test_case_id: Optional[str] = None) -> None: ...
    LLMResult = t.Any


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
        self.cache: List[dict[str, Any]] = []

    def _parse_key_str(self, key_str: str) -> Optional[dict[str, Any]]:
        try:
            parsed_key = json.loads(key_str)
            if not all(k in parsed_key for k in ["function", "args", "kwargs"]):
                return None

            semantic_arg_index = -1
            semantic_part = None
            for i, arg in enumerate(parsed_key["args"]):
                if isinstance(arg, str):
                    semantic_part = arg
                    semantic_arg_index = i
                    break

            if semantic_part is None:
                return None

            parsed_key["semantic_part"] = semantic_part
            parsed_key["semantic_arg_index"] = semantic_arg_index
            return parsed_key
        except json.JSONDecodeError:
            return None

    def _compare_non_semantic_parts(
        self,
        input_parsed_key: dict,
        stored_parsed_key: dict,
        semantic_arg_idx: int
    ) -> bool:
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
                continue
            if input_args[i] != stored_args[i]:
                return False
        return True

    def get(self, key_str: str) -> Any:
        if not self.cache:
            return None

        input_parsed_key = self._parse_key_str(key_str)
        if not input_parsed_key or "semantic_part" not in input_parsed_key:
            return None

        input_semantic_part = input_parsed_key["semantic_part"]
        input_semantic_arg_idx = input_parsed_key["semantic_arg_index"]
        input_embedding = np.array(self.embedding_model.embed_query(input_semantic_part))

        if input_embedding.size == 0:
             return None

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
                    return cached_item["value"]
        return None

    def set(self, key_str: str, value: Any) -> None:
        parsed_key = self._parse_key_str(key_str)
        if not parsed_key or "semantic_part" not in parsed_key:
            return

        semantic_part = parsed_key["semantic_part"]
        semantic_arg_index = parsed_key["semantic_arg_index"]
        embedding = np.array(self.embedding_model.embed_query(semantic_part))
        if embedding.size == 0:
            return

        self.cache.append({
            "embedding": embedding,
            "original_key_str": key_str,
            "value": value,
            "semantic_arg_index": semantic_arg_index,
        })

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
# hashlib is already imported
import functools
from pathlib import Path

try:
    from ragas.config import ragas_cache
except ImportError:
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
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / hashed_key

    def get(self, key: str) -> Any:
        filepath = self._get_filepath(key)
        if filepath.exists():
            with open(filepath, "rb") as f: return pickle.load(f)
        return None

    def set(self, key: str, value: Any) -> None:
        filepath = self._get_filepath(key)
        with open(filepath, "wb") as f: pickle.dump(value, f)

    def has_key(self, key: str) -> bool:
        return self._get_filepath(key).exists()

_DEFAULT_CACHE_SENTINEL = object()

def cacher(cache_backend: Optional[CacheInterface] = _DEFAULT_CACHE_SENTINEL):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            actual_cache_backend = cache_backend
            if actual_cache_backend is _DEFAULT_CACHE_SENTINEL:
                actual_cache_backend = ragas_cache
            if actual_cache_backend is None:
                return func(*args, **kwargs)
            if not isinstance(actual_cache_backend, CacheInterface):
                 print(f"Warning: Invalid cache backend type: {type(actual_cache_backend)}. Caching disabled.")
                 return func(*args, **kwargs)
            key_dict = {"function": func.__name__, "args": args, "kwargs": sorted(kwargs.items())}
            try:
                key = json.dumps(key_dict)
            except TypeError:
                print(f"Warning: Could not serialize args for {func.__name__}. Skipping cache.")
                return func(*args, **kwargs)
            if actual_cache_backend.has_key(key):
                return actual_cache_backend.get(key)
            else:
                result = func(*args, **kwargs)
                actual_cache_backend.set(key, result)
                return result
        return wrapper
    return decorator

# Global logger for this module, if not already defined at the top by other classes
logger = logging.getLogger(__name__) # Ensure logger is defined for SentenceEvaluatorSemanticCache

class SentenceEvaluatorSemanticCache:
    def __init__(self, embedding_model: BaseRagasEmbeddings, similarity_threshold: float):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache: t.List[t.Dict[str, t.Any]] = []
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.embedding_model is None:
            logger.warning("SentenceEvaluatorSemanticCache initialized with no embedding model.")

    def get(
        self, current_sentence: str, primary_context_or_hash: str,
        secondary_context_or_hash: t.Optional[str] = None,
    ) -> t.Optional[t.Any]:
        if not self.cache or self.embedding_model is None: return None
        try:
            current_sentence_embedding = np.array(self.embedding_model.embed_query(current_sentence)).squeeze()
        except Exception as e:
            logger.error(f"Failed to embed sentence for cache get ['{current_sentence}']: {e}")
            return None
        if current_sentence_embedding.size == 0 or np.linalg.norm(current_sentence_embedding) == 0:
            logger.warning(f"Skipping zero/empty embedding for sentence: {current_sentence}")
            return None
        normalized_current_embedding = current_sentence_embedding / np.linalg.norm(current_sentence_embedding)
        for item in self.cache:
            cached_embedding = item.get("key_embedding")
            if cached_embedding is None or not isinstance(cached_embedding, np.ndarray): continue
            cached_embedding = cached_embedding.squeeze()
            if cached_embedding.size == 0 or np.linalg.norm(cached_embedding) == 0 : continue
            normalized_cached_embedding = cached_embedding / np.linalg.norm(cached_embedding)
            similarity = np.dot(normalized_current_embedding, normalized_cached_embedding)
            if similarity >= self.similarity_threshold:
                cached_full_key = item.get("full_key", (None, None, None))
                if primary_context_or_hash == cached_full_key[1] and secondary_context_or_hash == cached_full_key[2]:
                    logger.debug(f"Semantic cache GET hit for sentence: {current_sentence}")
                    return item.get("value")
        return None

    def set(
        self, sentence_to_cache: str, primary_context_or_hash: str,
        llm_output_for_sentence: t.Any, secondary_context_or_hash: t.Optional[str] = None,
    ):
        if self.embedding_model is None: return
        try:
            sentence_embedding = np.array(self.embedding_model.embed_query(sentence_to_cache)).squeeze()
        except Exception as e:
            logger.error(f"Failed to embed sentence for cache set ['{sentence_to_cache}']: {e}")
            return
        if sentence_embedding.size == 0: return
        full_key = (sentence_to_cache, primary_context_or_hash, secondary_context_or_hash)
        self.cache.append({"key_embedding": sentence_embedding, "full_key": full_key, "value": llm_output_for_sentence})
        logger.debug(f"Semantic cache SET for sentence: {sentence_to_cache}")

# --- ComprehensiveSemanticCache Implementation (from Turn 59) ---
class ComprehensiveSemanticCache:
    CACHE_KEY_PREFIX_PROMPT_EMBEDDING = "prompt_embedding"
    CACHE_KEY_PREFIX_LLM_RESULT = "llm_result"

    def __init__(
        self,
        store: TestCasePartCacheStore,
        embedding_model: BaseRagasEmbeddings,
        similarity_threshold: float = 0.8,
        logger_level: int = logging.INFO,
    ):
        self.store = store
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logger_level)
        if not (0.0 <= self.similarity_threshold <= 1.0): raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.embedding_model is None: raise ValueError("embedding_model cannot be None.")
        if self.store is None: raise ValueError("store cannot be None.")
        self.logger.info(f"ComprehensiveSemanticCache initialized with threshold {self.similarity_threshold}.")

    def _get_prompt_embedding_key(self) -> str:
        return self.CACHE_KEY_PREFIX_PROMPT_EMBEDDING

    def _get_llm_result_key(self) -> str:
        return self.CACHE_KEY_PREFIX_LLM_RESULT

    def _normalize_embedding(self, embedding: t.Union[List[float], np.ndarray]) -> Optional[np.ndarray]:
        if not isinstance(embedding, (np.ndarray, list)): self.logger.warning(f"Invalid embedding type: {type(embedding)}"); return None
        np_embedding = np.array(embedding, dtype=float).flatten()
        if np_embedding.size == 0: self.logger.warning("Empty embedding."); return None
        norm = np.linalg.norm(np_embedding)
        if norm == 0: self.logger.warning("Zero-norm embedding."); return np_embedding
        return np_embedding / norm

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if emb1.shape != emb2.shape: self.logger.warning(f"Shape mismatch: {emb1.shape} vs {emb2.shape}."); return 0.0
        if emb1.ndim != 1 or emb2.ndim != 1: self.logger.warning("Embeddings not 1D."); return 0.0
        return float(np.dot(emb1, emb2))

    def get(self, test_case_id: str, current_prompt: str) -> Optional[LLMResult]:
        prompt_emb_store_key = self._get_prompt_embedding_key()
        cached_emb_serial = self.store.get(test_case_id, prompt_emb_store_key)
        if cached_emb_serial is None: self.logger.debug(f"[{test_case_id}] GET MISS (no embedding): '{current_prompt[:50]}...'"); return None

        norm_cached_emb = self._normalize_embedding(cached_emb_serial)
        if norm_cached_emb is None: self.logger.warning(f"[{test_case_id}] GET: Invalid cached embedding."); return None

        try:
            curr_emb_list = self.embedding_model.embed_query(current_prompt)
            norm_curr_emb = self._normalize_embedding(np.array(curr_emb_list))
        except Exception as e: self.logger.error(f"[{test_case_id}] GET embed error: {e}"); return None
        if norm_curr_emb is None: self.logger.warning(f"[{test_case_id}] GET: Current prompt embedding invalid."); return None

        similarity = self._calculate_similarity(norm_curr_emb, norm_cached_emb)
        self.logger.debug(f"[{test_case_id}] GET: Sim for '{current_prompt[:50]}...' is {similarity:.4f}")

        if similarity >= self.similarity_threshold:
            llm_res_key = self._get_llm_result_key()
            cached_res = self.store.get(test_case_id, llm_res_key)
            if cached_res is not None: self.logger.info(f"[{test_case_id}] GET HIT: '{current_prompt[:50]}...' (Sim: {similarity:.4f})"); return t.cast(LLMResult, cached_res)
            else: self.logger.warning(f"[{test_case_id}] GET: Prompt matched but no LLM result for '{llm_res_key}'.")
        else: self.logger.info(f"[{test_case_id}] GET MISS (low sim {similarity:.4f}): '{current_prompt[:50]}...'")
        return None

    def set(self, test_case_id: str, prompt: str, llm_result: LLMResult) -> None:
        try:
            prompt_emb_list = self.embedding_model.embed_query(prompt)
            norm_prompt_emb = self._normalize_embedding(np.array(prompt_emb_list))
            if norm_prompt_emb is None: self.logger.warning(f"[{test_case_id}] SET: Embedding for '{prompt[:50]}...' invalid. Skipping."); return
        except Exception as e: self.logger.error(f"[{test_case_id}] SET embed error: {e}"); return

        prompt_emb_key = self._get_prompt_embedding_key()
        llm_res_key = self._get_llm_result_key()
        self.store.set(test_case_id, prompt_emb_key, norm_prompt_emb.tolist())
        self.store.set(test_case_id, llm_res_key, llm_result)
        self.logger.info(f"[{test_case_id}] SET item for '{prompt[:50]}...'")

    def clear(self, test_case_id: Optional[str] = None) -> None:
        self.store.clear(test_case_id)
        self.logger.info(f"Cache CLEAR: {'test_case_id '+test_case_id if test_case_id else 'All entries'}.")

    def has_key(self, test_case_id: str, current_prompt: Optional[str] = None) -> bool:
        prompt_emb_key = self._get_prompt_embedding_key()
        llm_res_key = self._get_llm_result_key()
        if not self.store.has_key(test_case_id, prompt_emb_key) or \
           not self.store.has_key(test_case_id, llm_res_key): return False
        if current_prompt is None: return True
        else:
            cached_emb_serial = self.store.get(test_case_id, prompt_emb_key)
            if cached_emb_serial is None: return False
            norm_cached_emb = self._normalize_embedding(np.array(cached_emb_serial))
            if norm_cached_emb is None: return False
            try:
                curr_emb_list = self.embedding_model.embed_query(current_prompt)
                norm_curr_emb = self._normalize_embedding(np.array(curr_emb_list))
            except Exception: return False
            if norm_curr_emb is None: return False
            return self._calculate_similarity(norm_curr_emb, norm_cached_emb) >= self.similarity_threshold

```
