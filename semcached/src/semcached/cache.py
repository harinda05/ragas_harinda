import hashlib
import logging
import typing as t
import asyncio

# Attempt to import gptcache and related modules
try:
    # Renamed to avoid conflict with any local 'cache' module/directory if this file were named differently
    from gptcache import cache as gptcache_cache_module
    from gptcache.adapter.api import get as gptcache_get
    from gptcache.adapter.api import put as gptcache_put
    from gptcache.adapter.api import delete as gptcache_delete # For potential future use
    # GPTCacheSession might not be explicitly used if relying on global session context from init
    # from gptcache.session import Session as GPTCacheSession
    GPTCACHE_AVAILABLE = True
except ImportError:
    GPTCACHE_AVAILABLE = False
    # Define dummy/placeholder types for type hinting and basic parsing
    _T = t.TypeVar('_T')
    def _dummy_decorator(*args: t.Any, **kwargs: t.Any) -> t.Callable[[t.Callable[..., _T]], t.Callable[..., _T]]:
        def decorator(func: t.Callable[..., _T]) -> t.Callable[..., _T]:
            return func
        return decorator

    class _GPTCacheCacheModuleDummy:
        Cache = staticmethod(_dummy_decorator)
    gptcache_cache_module = _GPTCacheCacheModuleDummy() # type: ignore

    async def _dummy_gptcache_get(*args: t.Any, **kwargs: t.Any) -> t.Any: return None
    async def _dummy_gptcache_put(*args: t.Any, **kwargs: t.Any) -> None: pass
    def _dummy_gptcache_delete(*args: t.Any, **kwargs: t.Any) -> None: pass # Not async

    gptcache_get = _dummy_gptcache_get
    gptcache_put = _dummy_gptcache_put
    gptcache_delete = _dummy_gptcache_delete


# Import from local .config module
from .config import ensure_gptcache_initialized, is_semcached_initialized_and_enabled

logger = logging.getLogger(__name__)

# Define a type for the async function that will be executed if cache misses
AsyncExecutable = t.Callable[[], t.Awaitable[t.Any]]


class RagasGPTCacheWrapper:
    """
    A wrapper class to interact with the initialized gptcache.
    This version uses a composite key for gptcache prompts and does not
    explicitly manage GPTCacheSession objects in get/put calls, relying on
    the global gptcache context.
    """

    def __init__(self) -> None:
        """
        Initializes the RagasGPTCacheWrapper.
        Ensures that gptcache is initialized globally.
        """
        if not ensure_gptcache_initialized():
            logger.warning(
                "RagasGPTCacheWrapper: Semcached (gptcache) is not available or not enabled/initialized. "
                "Caching will be bypassed."
            )
        else:
            logger.info("RagasGPTCacheWrapper initialized, Semcached (gptcache) is active.")

    def _generate_test_case_id_hash(self, test_case_id: str) -> str:
        """Generates a SHA256 hash for the test_case_id for consistent key generation."""
        return hashlib.sha256(test_case_id.encode('utf-8')).hexdigest()

    async def get_or_execute(
        self,
        test_case_id: str,
        part_to_evaluate_id: str,
        executable_async_func: AsyncExecutable,
        # session_name: t.Optional[str] = None, # No longer explicitly used with gptcache_get/put
        on_hit: t.Optional[t.Callable[[str, t.Any], None]] = None,
        on_miss: t.Optional[t.Callable[[str], None]] = None,
    ) -> t.Any:
        """
        Tries to get a result from gptcache using a composite key.
        If it's a miss, executes the provided async function, caches its result,
        and returns the result.

        Args:
            test_case_id: An identifier for the test case or broader context.
            part_to_evaluate_id: A specific identifier for the part being evaluated.
                                 This, combined with test_case_id, forms the key.
            executable_async_func: An async function to execute if the cache misses.
            on_hit: Optional callback on cache hit. Args: composite_key, cached_value.
            on_miss: Optional callback on cache miss. Args: composite_key.

        Returns:
            The cached result or the result from executing the function.
        """
        if not is_semcached_initialized_and_enabled():
            logger.debug("Cache not initialized or enabled, executing function directly.")
            return await executable_async_func()

        test_case_hash = self._generate_test_case_id_hash(test_case_id)
        # The composite key is now used as the 'prompt' for gptcache
        gptcache_prompt_key = f"{test_case_hash}::PART::{part_to_evaluate_id}"

        try:
            # gptcache uses the 'prompt' kwarg as the key for embedding & lookup
            cached_value = await gptcache_get(prompt=gptcache_prompt_key)

            if cached_value is not None:
                logger.info(f"Cache HIT for key: '{gptcache_prompt_key}'")
                if on_hit:
                    on_hit(gptcache_prompt_key, cached_value)
                return cached_value
            else:
                logger.info(f"Cache MISS for key: '{gptcache_prompt_key}'")
                if on_miss:
                    on_miss(gptcache_prompt_key)

                llm_result = await executable_async_func()

                # Store the result. 'prompt' is the key, 'data' is the value.
                await gptcache_put(prompt=gptcache_prompt_key, data=llm_result)
                logger.info(f"Stored result for key: '{gptcache_prompt_key}' in cache.")
                return llm_result
        except Exception as e:
            logger.error(f"Error during cache get/execute for key '{gptcache_prompt_key}': {e}", exc_info=True)
            logger.warning(f"Executing function directly due to cache error for key '{gptcache_prompt_key}'.")
            return await executable_async_func()

    # clear_session_cache method removed as it's not directly compatible with this simplified keying
    # and gptcache's typical session handling via global init or context.
    # If session-like behavior is needed, test_case_id (or its hash) is part of the key.
    # Clearing would involve iterating keys with that prefix, which is complex and backend-dependent,
    # or gptcache supporting wildcard deletes on keys (which it generally doesn't).
    # For complete cache clearing, one would typically clear the entire gptcache data store.
```
