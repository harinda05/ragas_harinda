from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field, field_validator

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.losses import Loss
from ragas.optimizers import GeneticOptimizer, Optimizer

DEFAULT_OPTIMIZER_CONFIG = {"max_steps": 100}


class DemonstrationConfig(BaseModel):
    embedding: t.Any  # this has to be of type Any because BaseRagasEmbedding is an ABC
    enabled: bool = True
    top_k: int = 3
    threshold: float = 0.7
    technique: t.Literal["random", "similarity"] = "similarity"

    @field_validator("embedding")
    def validate_embedding(cls, v):
        if not isinstance(v, BaseRagasEmbeddings):
            raise ValueError("embedding must be an instance of BaseRagasEmbeddings")
        return v


class InstructionConfig(BaseModel):
    llm: BaseRagasLLM
    enabled: bool = True
    loss: t.Optional[Loss] = None
    optimizer: Optimizer = GeneticOptimizer()
    optimizer_config: t.Dict[str, t.Any] = Field(
        default_factory=lambda: DEFAULT_OPTIMIZER_CONFIG
    )


InstructionConfig.model_rebuild()


# -- Ragas Cache Configuration --
import os
from typing import Optional

from ragas.cache import CacheInterface, DiskCacheBackend, SemanticCacheBackend
from ragas.embeddings.base import BaseRagasEmbeddings
# Assuming specific embedding classes are in these locations, adjust if necessary
# We'll try-except import these in the helper function to avoid hard dependencies
# from ragas.embeddings.openai import OpenAIEmbeddings
# from ragas.embeddings.hf_embeddings import HuggingfaceEmbeddings


# Default values for cache configuration
DEFAULT_CACHE_ENABLED = True
DEFAULT_CACHE_BACKEND = "exact"  # "exact" or "semantic"
DEFAULT_CACHE_DIR = ".ragas_cache"
DEFAULT_SEMANTIC_CACHE_THRESHOLD = 0.85
DEFAULT_SEMANTIC_EMBEDDING_PROVIDER = "openai"
DEFAULT_SEMANTIC_EMBEDDING_MODEL_OPENAI = "text-embedding-ada-002"
DEFAULT_SEMANTIC_EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embedding_model(
    provider: str, model_name: Optional[str]
) -> Optional[BaseRagasEmbeddings]:
    """
    Helper function to instantiate and return an embedding model.
    Handles potential import errors for embedding providers.
    """
    provider = provider.lower()
    if provider == "openai":
        try:
            from ragas.embeddings.openai import OpenAIEmbeddings  # type: ignore
        except ImportError:
            # Log warning: OpenAI dependencies not installed
            print(
                "Warning: OpenAI embeddings specified but 'openai' or related packages not found. "
                "Install with 'pip install ragas[openai]'."
            )
            return None
        model_to_use = (
            model_name
            if model_name
            else DEFAULT_SEMANTIC_EMBEDDING_MODEL_OPENAI
        )
        return OpenAIEmbeddings(model_name=model_to_use)
    elif provider == "huggingface":
        try:
            from ragas.embeddings.hf_embeddings import HuggingfaceEmbeddings # type: ignore
        except ImportError:
            # Log warning: Huggingface dependencies not installed
            print(
                "Warning: Huggingface embeddings specified but 'sentence_transformers' not found. "
                "Install with 'pip install ragas[hf]'."
            )
            return None
        model_to_use = (
            model_name
            if model_name
            else DEFAULT_SEMANTIC_EMBEDDING_MODEL_HF
        )
        # Assuming HuggingfaceEmbeddings constructor takes model_name or similar
        return HuggingfaceEmbeddings(model_name=model_to_use)
    # Add other providers here as elif blocks
    else:
        # Log warning: Unsupported provider
        print(f"Warning: Unsupported embedding provider specified: {provider}")
        return None


def get_ragas_cache() -> Optional[CacheInterface]:
    """
    Initializes and returns the Ragas cache backend based on environment variables.
    """
    cache_enabled_str = os.environ.get("RAGAS_CACHE_ENABLED", str(DEFAULT_CACHE_ENABLED)).lower()
    if cache_enabled_str == "false":
        return None

    backend_type = os.environ.get("RAGAS_CACHE_BACKEND", DEFAULT_CACHE_BACKEND).lower()

    if backend_type == "exact":
        cache_dir = os.environ.get("RAGAS_CACHE_DIR", DEFAULT_CACHE_DIR)
        return DiskCacheBackend(cache_dir=cache_dir)
    elif backend_type == "semantic":
        threshold_str = os.environ.get(
            "RAGAS_SEMANTIC_CACHE_THRESHOLD", str(DEFAULT_SEMANTIC_CACHE_THRESHOLD)
        )
        try:
            threshold = float(threshold_str)
        except ValueError:
            print(
                f"Warning: Invalid RAGAS_SEMANTIC_CACHE_THRESHOLD value '{threshold_str}'. "
                f"Using default {DEFAULT_SEMANTIC_CACHE_THRESHOLD}."
            )
            threshold = DEFAULT_SEMANTIC_CACHE_THRESHOLD

        provider = os.environ.get(
            "RAGAS_SEMANTIC_CACHE_EMBEDDING_PROVIDER",
            DEFAULT_SEMANTIC_EMBEDDING_PROVIDER,
        ).lower()
        model_name = os.environ.get("RAGAS_SEMANTIC_CACHE_EMBEDDING_MODEL_NAME") # Can be None

        embedding_model = _get_embedding_model(provider=provider, model_name=model_name)
        if embedding_model is None:
            print(
                "Warning: Semantic cache specified but failed to load embedding model. "
                "Disabling cache."
            )
            return None

        return SemanticCacheBackend(
            embedding_model=embedding_model, similarity_threshold=threshold
        )
    else:
        print(
            f"Warning: Invalid RAGAS_CACHE_BACKEND type '{backend_type}'. "
            "Cache will be disabled."
        )
        return None


# Initialize the global cache object
ragas_cache: Optional[CacheInterface] = get_ragas_cache()


# -- Ragas Sentence Evaluator Semantic Cache Configuration --
from ragas.cache import SentenceEvaluatorSemanticCache # Added import

# Default values for Sentence Evaluator Semantic Cache configuration
DEFAULT_SENTENCE_EVAL_CACHE_ENABLED = True
DEFAULT_SENTENCE_EVAL_EMBEDDING_PROVIDER = "openai" # Default provider
DEFAULT_SENTENCE_EVAL_EMBEDDING_MODEL_OPENAI = "text-embedding-ada-002" # Default for OpenAI
DEFAULT_SENTENCE_EVAL_EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2" # Default for HF
DEFAULT_SENTENCE_EVAL_SIMILARITY_THRESHOLD = 0.80


class SentenceCacheConfig(BaseModel):
    """Configuration for the Sentence Evaluator Semantic Cache."""
    enabled: bool = DEFAULT_SENTENCE_EVAL_CACHE_ENABLED
    embedding_provider: str = DEFAULT_SENTENCE_EVAL_EMBEDDING_PROVIDER
    embedding_model_name: Optional[str] = None # Uses provider-specific default if None
    similarity_threshold: float = DEFAULT_SENTENCE_EVAL_SIMILARITY_THRESHOLD

    # If needed, add a validator for embedding_provider or other fields later.


def get_sentence_eval_cache() -> Optional[SentenceEvaluatorSemanticCache]:
    """
    Initializes and returns the Ragas Sentence Evaluator Semantic Cache
    based on environment variables or defaults.
    """
    enabled_str = os.environ.get(
        "RAGAS_SENTENCE_EVAL_CACHE_ENABLED", str(DEFAULT_SENTENCE_EVAL_CACHE_ENABLED)
    ).lower()
    if enabled_str == "false":
        return None

    provider = os.environ.get(
        "RAGAS_SENTENCE_EVAL_EMBEDDING_PROVIDER",
        DEFAULT_SENTENCE_EVAL_EMBEDDING_PROVIDER,
    ).lower()

    model_name = os.environ.get("RAGAS_SENTENCE_EVAL_EMBEDDING_MODEL_NAME") # Can be None

    threshold_str = os.environ.get(
        "RAGAS_SENTENCE_EVAL_SIMILARITY_THRESHOLD",
        str(DEFAULT_SENTENCE_EVAL_SIMILARITY_THRESHOLD),
    )
    try:
        threshold = float(threshold_str)
    except ValueError:
        print(
            f"Warning: Invalid RAGAS_SENTENCE_EVAL_SIMILARITY_THRESHOLD value '{threshold_str}'. "
            f"Using default {DEFAULT_SENTENCE_EVAL_SIMILARITY_THRESHOLD}."
        )
        threshold = DEFAULT_SENTENCE_EVAL_SIMILARITY_THRESHOLD

    # Determine the actual model name to use for the _get_embedding_model helper
    # The helper _get_embedding_model already handles provider-specific defaults if model_name is None
    actual_model_name = model_name
    if model_name is None: # Pass the provider-specific default if no env var is set
        if provider == "openai":
            actual_model_name = DEFAULT_SENTENCE_EVAL_EMBEDDING_MODEL_OPENAI
        elif provider == "huggingface":
            actual_model_name = DEFAULT_SENTENCE_EVAL_EMBEDDING_MODEL_HF
        # _get_embedding_model will use its own internal defaults if actual_model_name is still None,
        # but passing explicit defaults here ensures alignment with SentenceCacheConfig defaults.

    embedding_model = _get_embedding_model(provider=provider, model_name=actual_model_name)
    if embedding_model is None:
        print(
            "Warning: Sentence Evaluator Semantic Cache specified but failed to load embedding model. "
            "Disabling this cache."
        )
        return None

    return SentenceEvaluatorSemanticCache(
        embedding_model=embedding_model, similarity_threshold=threshold
    )


# Initialize the global sentence evaluator cache object
ragas_sentence_eval_cache: Optional[SentenceEvaluatorSemanticCache] = get_sentence_eval_cache()


# -- Ragas Comprehensive Semantic Cache Configuration --
from ragas.cache import ComprehensiveSemanticCache # Added
from ragas.cache_store import TestCasePartCacheStore, InMemoryStore # Added
import logging # Should already be available if other loggers are used, or add if not.

# Default values for Comprehensive Semantic Cache configuration
DEFAULT_COMPREHENSIVE_CACHE_ENABLED = True
DEFAULT_COMPREHENSIVE_CACHE_STORE_BACKEND = "inmemory" # Only "inmemory" supported for now
DEFAULT_COMPREHENSIVE_CACHE_EMBEDDING_PROVIDER = "openai"
DEFAULT_COMPREHENSIVE_CACHE_EMBEDDING_MODEL_OPENAI = "text-embedding-ada-002"
DEFAULT_COMPREHENSIVE_CACHE_EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COMPREHENSIVE_CACHE_SIMILARITY_THRESHOLD = 0.80
DEFAULT_COMPREHENSIVE_CACHE_LOGGER_LEVEL = logging.INFO


def get_comprehensive_cache() -> Optional[ComprehensiveSemanticCache]:
    """
    Initializes and returns the Ragas Comprehensive Semantic Cache
    based on environment variables or defaults.
    """
    enabled_str = os.environ.get(
        "RAGAS_COMPREHENSIVE_CACHE_ENABLED", str(DEFAULT_COMPREHENSIVE_CACHE_ENABLED)
    ).lower()
    if enabled_str == "false":
        print("ComprehensiveSemanticCache is disabled via RAGAS_COMPREHENSIVE_CACHE_ENABLED=false.")
        return None

    # Store Backend Configuration (currently only InMemoryStore)
    store_backend_type = os.environ.get(
        "RAGAS_COMPREHENSIVE_CACHE_STORE_BACKEND", DEFAULT_COMPREHENSIVE_CACHE_STORE_BACKEND
    ).lower()

    store_instance: Optional[TestCasePartCacheStore] = None
    if store_backend_type == "inmemory":
        store_instance = InMemoryStore()
    else:
        print(
            f"Warning: Unsupported RAGAS_COMPREHENSIVE_CACHE_STORE_BACKEND '{store_backend_type}'. "
            "Currently, only 'inmemory' is supported. Disabling ComprehensiveSemanticCache."
        )
        return None # Or fallback to InMemoryStore if preferred: store_instance = InMemoryStore()

    if store_instance is None: # Should be caught by else above, but as safeguard
        print("Warning: Failed to initialize cache store backend. Disabling ComprehensiveSemanticCache.")
        return None

    # Embedding Model Configuration
    embedding_provider = os.environ.get(
        "RAGAS_COMPREHENSIVE_CACHE_EMBEDDING_PROVIDER", DEFAULT_COMPREHENSIVE_CACHE_EMBEDDING_PROVIDER
    ).lower()

    embedding_model_name_env = os.environ.get("RAGAS_COMPREHENSIVE_CACHE_EMBEDDING_MODEL_NAME") # Can be None

    # Determine the actual model name to pass to _get_embedding_model
    actual_embedding_model_name = embedding_model_name_env
    if embedding_model_name_env is None: # Pass the provider-specific default if no env var is set
        if embedding_provider == "openai":
            actual_embedding_model_name = DEFAULT_COMPREHENSIVE_CACHE_EMBEDDING_MODEL_OPENAI
        elif embedding_provider == "huggingface":
            actual_embedding_model_name = DEFAULT_COMPREHENSIVE_CACHE_EMBEDDING_MODEL_HF

    embedding_model = _get_embedding_model(provider=embedding_provider, model_name=actual_embedding_model_name)
    if embedding_model is None:
        print(
            "Warning: ComprehensiveSemanticCache specified but failed to load embedding model. "
            "Disabling this cache."
        )
        return None

    # Similarity Threshold
    threshold_str = os.environ.get(
        "RAGAS_COMPREHENSIVE_CACHE_SIMILARITY_THRESHOLD",
        str(DEFAULT_COMPREHENSIVE_CACHE_SIMILARITY_THRESHOLD),
    )
    try:
        similarity_threshold = float(threshold_str)
    except ValueError:
        print(
            f"Warning: Invalid RAGAS_COMPREHENSIVE_CACHE_SIMILARITY_THRESHOLD value '{threshold_str}'. "
            f"Using default {DEFAULT_COMPREHENSIVE_CACHE_SIMILARITY_THRESHOLD}."
        )
        similarity_threshold = DEFAULT_COMPREHENSIVE_CACHE_SIMILARITY_THRESHOLD

    # Logger Level (Example, assuming ComprehensiveSemanticCache takes logger_level)
    logger_level_str = os.environ.get(
        "RAGAS_COMPREHENSIVE_CACHE_LOGGER_LEVEL", str(DEFAULT_COMPREHENSIVE_CACHE_LOGGER_LEVEL)
    )
    # Convert string to int for logging level if necessary, or handle specific string values
    # For simplicity, assuming ComprehensiveSemanticCache handles int or can parse common strings.
    # Here, we'll just pass the string or default int.
    try:
        logger_level = int(logger_level_str)
    except ValueError: # If it's "INFO", "DEBUG" etc.
        # ComprehensiveSemanticCache's __init__ should handle this, or we add mapping here.
        # For now, assume direct int or a default if parsing fails.
        logger_level_val = logging.getLevelName(logger_level_str.upper())
        if isinstance(logger_level_val, int):
            logger_level = logger_level_val
        else: # Fallback if string is not a valid level name
            print(f"Warning: Invalid RAGAS_COMPREHENSIVE_CACHE_LOGGER_LEVEL '{logger_level_str}'. Using default.")
            logger_level = DEFAULT_COMPREHENSIVE_CACHE_LOGGER_LEVEL


    print(
        f"Initializing ComprehensiveSemanticCache with: Store='{store_backend_type}', "
        f"Embeddings='{embedding_provider}:{actual_embedding_model_name or 'default'}', "
        f"Threshold='{similarity_threshold}', LoggerLevel='{logging.getLevelName(logger_level)}'"
    )

    return ComprehensiveSemanticCache(
        store=store_instance,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        logger_level=logger_level # Pass the determined logger level
    )

# Initialize the global comprehensive semantic cache object
ragas_comprehensive_cache: Optional[ComprehensiveSemanticCache] = get_comprehensive_cache()
