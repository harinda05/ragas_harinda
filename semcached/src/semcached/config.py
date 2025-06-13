import os
import logging
import typing as t
import threading

# Try importing gptcache modules with error handling for environments where it might not be installed.
try:
    from gptcache import cache as gptcache_cache # Renamed to avoid conflict with a 'cache' directory
    from gptcache.manager import manager_factory
    from gptcache.processor.pre import get_prompt
    from gptcache.embedding import Onnx as EmbeddingOnnx
    from gptcache.embedding import OpenAI as EmbeddingOpenAI
    # Add other embedding providers if needed, e.g., Huggingface, Cohere
    # from gptcache.embedding import Huggingface as EmbeddingHuggingface
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
    # Alternative similarity evaluations:
    # from gptcache.similarity_evaluation.exact_match import ExactMatchEvaluation
    # from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation
    GPTCACHE_AVAILABLE = True
except ImportError:
    GPTCACHE_AVAILABLE = False
    # Define dummy types or placeholders if gptcache is not available,
    # so the rest of the file can be parsed without errors.
    # This helps in environments where gptcache might be an optional dependency.
    EmbeddingOnnx = t.TypeVar('EmbeddingOnnx')
    EmbeddingOpenAI = t.TypeVar('EmbeddingOpenAI')
    SearchDistanceEvaluation = t.TypeVar('SearchDistanceEvaluation')
    manager_factory = lambda *args, **kwargs: None
    get_prompt = lambda *args, **kwargs: None # type: ignore
    class gptcache_cache: # Dummy class
        @staticmethod
        def init(*args: t.Any, **kwargs: t.Any) -> None: pass
        @staticmethod
        def set_openai_key(*args: t.Any, **kwargs: t.Any) -> None: pass

        # This is a class decorator in gptcache, needs to return a decorator function
        @staticmethod
        def Cache(*args: t.Any, **kwargs: t.Any) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
            def decorator(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
                return func
            return decorator


# Global logger for this module
logger = logging.getLogger(__name__)

# Default configuration values
SEMCACHED_CACHE_ENABLE_DEFAULT: bool = True
SEMCACHED_DATA_DIR_DEFAULT: str = ".semcached_data" # Default data directory for cache persistence
SEMCACHED_SIMILARITY_THRESHOLD_DEFAULT: float = 0.8
SEMCACHED_EMBEDDING_MODEL_DEFAULT: str = "onnx" # "onnx" or "openai" or other supported gptcache embedders
SEMCACHED_CACHE_STORAGE_DEFAULT: str = "sqlite" # e.g., "sqlite", "duckdb"
SEMCACHED_VECTOR_STORE_DEFAULT: str = "faiss" # e.g., "faiss", "chroma", "docarray"

# Environment variable names
SEMCACHED_CACHE_ENABLE_ENV_VAR: str = "SEMCACHED_CACHE_ENABLE"
SEMCACHED_DATA_DIR_ENV_VAR: str = "SEMCACHED_DATA_DIR"
SEMCACHED_SIMILARITY_THRESHOLD_ENV_VAR: str = "SEMCACHED_SIMILARITY_THRESHOLD"
SEMCACHED_EMBEDDING_MODEL_ENV_VAR: str = "SEMCACHED_EMBEDDING_MODEL"
SEMCACHED_OPENAI_API_KEY_ENV_VAR: str = "OPENAI_API_KEY" # Used if embedding model is OpenAI
SEMCACHED_CACHE_STORAGE_ENV_VAR: str = "SEMCACHED_CACHE_STORAGE"
SEMCACHED_VECTOR_STORE_ENV_VAR: str = "SEMCACHED_VECTOR_STORE"


# Internal state
_gptcache_initialized: bool = False
_gptcache_init_lock = threading.Lock()


def ensure_gptcache_initialized() -> bool:
    """
    Initializes gptcache if it hasn't been already, using environment variables for configuration.
    Returns True if gptcache is initialized AND was enabled, False otherwise.
    """
    global _gptcache_initialized
    if not GPTCACHE_AVAILABLE:
        logger.warning("gptcache library is not installed. Semcached will not function.")
        return False

    with _gptcache_init_lock:
        if _gptcache_initialized:
            # If already marked as initialized, check the actual enabled status
            cache_enabled_str = os.getenv(SEMCACHED_CACHE_ENABLE_ENV_VAR, str(SEMCACHED_CACHE_ENABLE_DEFAULT))
            cache_enabled = cache_enabled_str.lower() in ("true", "1", "yes")
            return cache_enabled # Return true only if it was initialized AND is enabled

        # Proceed with initialization attempt
        try:
            cache_enabled_str = os.getenv(SEMCACHED_CACHE_ENABLE_ENV_VAR, str(SEMCACHED_CACHE_ENABLE_DEFAULT))
            cache_enabled = cache_enabled_str.lower() in ("true", "1", "yes")

            _gptcache_initialized = True # Mark that we've gone through the init logic once

            if not cache_enabled:
                logger.info("Semcached (gptcache integration) is disabled via environment variable.")
                return False # Cache is not active

            data_dir = os.getenv(SEMCACHED_DATA_DIR_ENV_VAR, SEMCACHED_DATA_DIR_DEFAULT)
            similarity_threshold = float(os.getenv(SEMCACHED_SIMILARITY_THRESHOLD_ENV_VAR, str(SEMCACHED_SIMILARITY_THRESHOLD_DEFAULT)))
            embedding_model_name = os.getenv(SEMCACHED_EMBEDDING_MODEL_ENV_VAR, SEMCACHED_EMBEDDING_MODEL_DEFAULT).lower()
            cache_storage = os.getenv(SEMCACHED_CACHE_STORAGE_ENV_VAR, SEMCACHED_CACHE_STORAGE_DEFAULT).lower()
            vector_store = os.getenv(SEMCACHED_VECTOR_STORE_ENV_VAR, SEMCACHED_VECTOR_STORE_DEFAULT).lower()

            os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists

            embedding_func: t.Any = None
            if embedding_model_name == "onnx":
                embedding_func = EmbeddingOnnx()
            elif embedding_model_name == "openai":
                api_key = os.getenv(SEMCACHED_OPENAI_API_KEY_ENV_VAR)
                if not api_key:
                    logger.error(
                        f"{SEMCACHED_OPENAI_API_KEY_ENV_VAR} not set. Cannot use OpenAI embeddings for Semcached."
                    )
                    return False # Initialization failed due to missing config
                gptcache_cache.set_openai_key(api_key)
                embedding_func = EmbeddingOpenAI()
            else:
                logger.error(f"Unsupported embedding model for Semcached: {embedding_model_name}")
                return False # Initialization failed due to unsupported model

            logger.info(
                f"Initializing Semcached (gptcache) with: data_dir='{data_dir}', "
                f"similarity_threshold={similarity_threshold}, embedding_model='{embedding_model_name}', "
                f"cache_storage='{cache_storage}', vector_store='{vector_store}'"
            )

            gptcache_cache.init(
                pre_embedding_func=get_prompt,
                embedding_func=embedding_func,
                data_manager=manager_factory(
                    data_dir=data_dir,
                    vector_store_name=vector_store,
                    data_storage_name=cache_storage,
                ),
                similarity_evaluation=SearchDistanceEvaluation(),
                similarity_threshold=similarity_threshold,
            )
            logger.info("Semcached (gptcache integration) initialized successfully and is enabled.")
            return True # Successfully initialized and enabled
        except Exception as e:
            logger.error(f"Failed to initialize Semcached (gptcache integration): {e}", exc_info=True)
            # _gptcache_initialized is already True, but initialization failed.
            # This state means we won't try again, but it's not truly usable.
            # The function will return False in this case.
            return False


def is_semcached_initialized_and_enabled() -> bool:
    """
    Checks if Semcached (via gptcache) has successfully completed its initialization
    and was enabled during that process.
    """
    if not GPTCACHE_AVAILABLE:
        return False

    # This relies on ensure_gptcache_initialized() having been called at least once.
    # If _gptcache_initialized is False, it means ensure_gptcache_initialized was never called or failed very early.
    if not _gptcache_initialized: # Check if the initialization process has been attempted
        logger.debug("Semcached initialization has not been attempted yet.")
        # Optionally, trigger initialization here if desired, or rely on explicit calls elsewhere.
        # For a simple status check, if it hasn't been initialized, it's not enabled.
        return False

    # If initialization was attempted (_gptcache_initialized is True),
    # then re-check the enabled status from environment variables.
    # This ensures that if init failed but set _gptcache_initialized, we still get correct "enabled" status.
    cache_enabled_str = os.getenv(SEMCACHED_CACHE_ENABLE_ENV_VAR, str(SEMCACHED_CACHE_ENABLE_DEFAULT))
    cache_enabled = cache_enabled_str.lower() in ("true", "1", "yes")

    # True only if init was attempted AND cache is configured to be enabled.
    # A more robust check might involve a separate flag that's only True if gptcache.cache.init() succeeded.
    # However, ensure_gptcache_initialized() itself returns False if init fails or if not enabled.
    # So, if _gptcache_initialized is true, it means init process ran. We then check env var for "enabled".
    return cache_enabled
```
