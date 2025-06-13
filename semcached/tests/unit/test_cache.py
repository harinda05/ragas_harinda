import pytest
from unittest import mock
import asyncio
import hashlib

# Assuming pytest is run from the repository root (/app)
from semcached.src.semcached.cache import RagasGPTCacheWrapper
# LLMResult is t.Any, so not strictly needed for import if not type checking deeply in tests.
# from semcached.src.semcached.config import LLMResult

# --- Mocks for gptcache.adapter.api and config functions ---
# These will be applied via monkeypatch in fixtures.

@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mocks dependencies for RagasGPTCacheWrapper tests."""
    # Mock config functions from the context of where they are imported in cache.py
    mock_ensure_init = mock.MagicMock()
    mock_is_init_and_enabled = mock.MagicMock()
    monkeypatch.setattr("semcached.src.semcached.cache.ensure_gptcache_initialized", mock_ensure_init)
    monkeypatch.setattr("semcached.src.semcached.cache.is_semcached_initialized_and_enabled", mock_is_init_and_enabled)

    # Mock gptcache.adapter.api methods (async mocks needed for get/put)
    # These are imported as gptcache_get, gptcache_put in cache.py
    mock_gptcache_get_api = mock.AsyncMock()
    mock_gptcache_put_api = mock.AsyncMock()
    monkeypatch.setattr("semcached.src.semcached.cache.gptcache_get", mock_gptcache_get_api)
    monkeypatch.setattr("semcached.src.semcached.cache.gptcache_put", mock_gptcache_put_api)

    return {
        "ensure_init": mock_ensure_init,
        "is_init_and_enabled": mock_is_init_and_enabled,
        "gptcache_get": mock_gptcache_get_api, # Use the name matching the fixture key for clarity
        "gptcache_put": mock_gptcache_put_api, # Use the name matching the fixture key for clarity
    }

@pytest.fixture
def cache_wrapper(mock_dependencies):
    """Provides a RagasGPTCacheWrapper instance with mocked dependencies."""
    # By default, set up mocks as if cache is initialized and enabled for most tests
    mock_dependencies["ensure_init"].return_value = True
    mock_dependencies["is_init_and_enabled"].return_value = True
    return RagasGPTCacheWrapper()

async def mock_llm_func(param="default_param"):
    """A mock asynchronous LLM function."""
    await asyncio.sleep(0.001) # Simulate minimal async behavior
    return f"LLM_Response_to_{param}"


@pytest.mark.asyncio
async def test_rgw_init(mock_dependencies):
    """Tests RagasGPTCacheWrapper initialization calls ensure_gptcache_initialized."""
    # RagasGPTCacheWrapper() instance is created, __init__ should call ensure_init
    RagasGPTCacheWrapper()
    mock_dependencies["ensure_init"].assert_called_once()

@pytest.mark.asyncio
async def test_rgw_generate_test_case_id_hash(cache_wrapper): # cache_wrapper fixture implicitly calls __init__
    """Tests the hash generation for test_case_id."""
    test_id = "my_test_case_123"
    expected_hash = hashlib.sha256(test_id.encode('utf-8')).hexdigest()
    assert cache_wrapper._generate_test_case_id_hash(test_id) == expected_hash

@pytest.mark.asyncio
async def test_rgw_get_or_execute_cache_disabled(mock_dependencies, cache_wrapper):
    """Tests get_or_execute when cache is disabled (is_semcached_initialized_and_enabled returns False)."""
    mock_dependencies["is_init_and_enabled"].return_value = False # Override default from fixture

    part_id = "part1_disabled_cache"
    test_case_id = "case1_disabled_cache"

    llm_response_content = f"LLM_Response_to_{part_id}"

    result = await cache_wrapper.get_or_execute(
        test_case_id=test_case_id,
        part_to_evaluate_id=part_id,
        executable_async_func=lambda: mock_llm_func(part_id)
    )

    assert result == llm_response_content
    mock_dependencies["gptcache_get"].assert_not_called()
    mock_dependencies["gptcache_put"].assert_not_called()

@pytest.mark.asyncio
async def test_rgw_get_or_execute_cache_hit(mock_dependencies, cache_wrapper):
    """Tests get_or_execute behavior on a cache hit."""
    mock_cached_value = "cached_llm_response_on_hit"
    mock_dependencies["gptcache_get"].return_value = mock_cached_value # Simulate API returning cached data

    part_id = "part_on_hit"
    test_case_id = "case_hit"
    test_case_hash = cache_wrapper._generate_test_case_id_hash(test_case_id)
    expected_gptcache_key = f"{test_case_hash}::PART::{part_id}"

    # This mock executable should not be called if cache hits
    mock_executable = mock.AsyncMock(return_value="new_llm_response_should_not_be_used")
    on_hit_callback = mock.MagicMock()
    on_miss_callback = mock.MagicMock()

    result = await cache_wrapper.get_or_execute(
        test_case_id=test_case_id,
        part_to_evaluate_id=part_id,
        executable_async_func=mock_executable,
        on_hit=on_hit_callback,
        on_miss=on_miss_callback
    )

    assert result == mock_cached_value
    mock_dependencies["gptcache_get"].assert_awaited_once_with(prompt=expected_gptcache_key)
    mock_executable.assert_not_called() # Crucial check for cache hit
    mock_dependencies["gptcache_put"].assert_not_called() # Nothing new to put
    on_hit_callback.assert_called_once_with(expected_gptcache_key, mock_cached_value)
    on_miss_callback.assert_not_called()

@pytest.mark.asyncio
async def test_rgw_get_or_execute_cache_miss(mock_dependencies, cache_wrapper):
    """Tests get_or_execute behavior on a cache miss."""
    mock_dependencies["gptcache_get"].return_value = None # Simulate API returning no cached data

    fresh_llm_response = "fresh_llm_response_from_miss"
    part_id = "part_on_miss"
    test_case_id = "case_miss"
    test_case_hash = cache_wrapper._generate_test_case_id_hash(test_case_id)
    expected_gptcache_key = f"{test_case_hash}::PART::{part_id}"

    # This mock executable should be called on cache miss
    mock_executable = mock.AsyncMock(return_value=fresh_llm_response)
    on_hit_callback = mock.MagicMock()
    on_miss_callback = mock.MagicMock()

    result = await cache_wrapper.get_or_execute(
        test_case_id=test_case_id,
        part_to_evaluate_id=part_id,
        executable_async_func=mock_executable,
        on_hit=on_hit_callback,
        on_miss=on_miss_callback
    )

    assert result == fresh_llm_response
    mock_dependencies["gptcache_get"].assert_awaited_once_with(prompt=expected_gptcache_key)
    mock_executable.assert_awaited_once() # Crucial check for cache miss
    mock_dependencies["gptcache_put"].assert_awaited_once_with(prompt=expected_gptcache_key, data=fresh_llm_response)
    on_miss_callback.assert_called_once_with(expected_gptcache_key)
    on_hit_callback.assert_not_called()

# Placeholder for more tests mentioned in thought block (Turn 89):
# @pytest.mark.asyncio
# async def test_rgw_get_or_execute_llm_func_failure(mock_dependencies, cache_wrapper):
#     # Test when executable_async_func raises an exception
#     # Should not cache, should propagate exception (or handle as defined)
#     pass

# @pytest.mark.asyncio
# async def test_rgw_get_or_execute_gptcache_get_failure(mock_dependencies, cache_wrapper):
#     # Test when gptcache_get itself raises an exception
#     # Should fall back to executing function and not attempt put (or handle gracefully)
#     pass

# @pytest.mark.asyncio
# async def test_rgw_get_or_execute_gptcache_put_failure(mock_dependencies, cache_wrapper):
#     # Test when gptcache_put raises an exception after LLM call
#     # Result should still be returned, but error logged, cache not saved
#     pass
```
