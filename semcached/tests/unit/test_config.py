import pytest
from unittest import mock
import os
import sys
import logging

# Assuming pytest is run from the repository root (/app),
# this import should work if Python's path resolution includes the root.
from semcached.src.semcached import config as semcached_config


# --- Mock Objects for GPTCache components ---
class MockEmbedding:
    def __init__(self, model_name="mock_onnx_model"):
        self.model_name = model_name
        # logger.debug(f"MockEmbedding initialized with {model_name}")

class MockManagerFactory:
    def __init__(self, data_dir, vector_store_name, data_storage_name):
        self.data_dir = data_dir
        self.vector_store_name = vector_store_name
        self.data_storage_name = data_storage_name
        # logger.debug(f"MockManagerFactory: dir={data_dir}, vs={vector_store_name}, ds={data_storage_name}")

class MockSearchDistanceEvaluation:
    pass

class MockGPTCacheModuleSingleton: # Renamed to avoid confusion if gptcache_cache is also a module name
    def __init__(self):
        self.api_key = None
        self.init_params = None # Store init params here
        # logger.debug("MockGPTCacheModuleSingleton instance created")

    def init(self, pre_embedding_func, embedding_func, data_manager, similarity_evaluation, similarity_threshold):
        # logger.debug(f"MockGPTCacheModuleSingleton.init called: emb_func={embedding_func}, threshold={similarity_threshold}")
        # logger.debug(f"  DataManager with: {data_manager.data_dir}, {data_manager.vector_store_name}, {data_manager.data_storage_name}")
        # logger.debug(f"  SimilarityEval: {similarity_evaluation.__class__.__name__}")
        self.init_params = {
            "embedding_func_instance": embedding_func, # Store instance
            "data_manager_instance": data_manager, # Store instance
            "similarity_evaluation_type": type(similarity_evaluation).__name__,
            "similarity_threshold": similarity_threshold,
        }
    def set_openai_key(self, api_key):
        self.api_key = api_key
        # logger.debug(f"MockGPTCacheModuleSingleton.set_openai_key called with key: {api_key[:5]}...")

    def reset_mock(self): # Helper to reset for tests
        self.api_key = None
        self.init_params = None


@pytest.fixture
def mock_gptcache_singleton():
    # This ensures a single, resettable mock instance for gptcache_cache module behavior
    return MockGPTCacheModuleSingleton()

@pytest.fixture
def reset_config_globals(monkeypatch, mock_gptcache_singleton):
    """Resets global state in semcached_config and mocks gptcache modules."""
    monkeypatch.setattr(semcached_config, '_gptcache_initialized', False)

    # Use the singleton mock for gptcache_cache
    monkeypatch.setattr(semcached_config, 'gptcache_cache', mock_gptcache_singleton)
    mock_gptcache_singleton.reset_mock() # Ensure it's clean before test

    monkeypatch.setattr(semcached_config, 'manager_factory', MockManagerFactory)
    # Lambda creates new instance each time, which is fine for these mocks
    monkeypatch.setattr(semcached_config, 'EmbeddingOnnx', lambda: MockEmbedding(model_name="onnx"))
    monkeypatch.setattr(semcached_config, 'EmbeddingOpenAI', lambda: MockEmbedding(model_name="openai"))
    monkeypatch.setattr(semcached_config, 'SearchDistanceEvaluation', MockSearchDistanceEvaluation)
    monkeypatch.setattr(semcached_config, 'get_prompt', mock.MagicMock()) # Simple mock for get_prompt

    # Clear relevant environment variables before each test
    env_vars_to_clear = [
        semcached_config.SEMCACHED_CACHE_ENABLE_ENV_VAR,
        semcached_config.SEMCACHED_DATA_DIR_ENV_VAR,
        semcached_config.SEMCACHED_SIMILARITY_THRESHOLD_ENV_VAR,
        semcached_config.SEMCACHED_EMBEDDING_MODEL_ENV_VAR,
        semcached_config.SEMCACHED_OPENAI_API_KEY_ENV_VAR,
        semcached_config.SEMCACHED_CACHE_STORAGE_ENV_VAR,
        semcached_config.SEMCACHED_VECTOR_STORE_ENV_VAR,
    ]
    for var_name in env_vars_to_clear:
        if var_name in os.environ:
            monkeypatch.delenv(var_name, raising=False) # raising=False to avoid error if already deleted

    yield # Test runs here


def test_ensure_gptcache_initialized_disabled_via_env(reset_config_globals, monkeypatch, caplog):
    """
    Tests that gptcache initialization returns False and logs correctly if SEMCACHED_CACHE_ENABLE is 'false'.
    """
    monkeypatch.setenv(semcached_config.SEMCACHED_CACHE_ENABLE_ENV_VAR, "false")

    with caplog.at_level(logging.INFO):
        result = semcached_config.ensure_gptcache_initialized()

    assert not result, "Should return False when cache is disabled"
    assert semcached_config._gptcache_initialized, "Flag _gptcache_initialized should be True as init logic was run"
    assert "Semcached (gptcache integration) is disabled via environment variable." in caplog.text

    # Ensure gptcache_cache.init was NOT called by checking init_params on the mock
    assert semcached_config.gptcache_cache.init_params is None, "gptcache.init should not have been called"


def test_ensure_gptcache_initialized_default_onnx_faiss(reset_config_globals, monkeypatch, caplog):
    """
    Tests successful initialization with default settings (ONNX embeddings, SQLite/Faiss).
    """
    monkeypatch.setenv(semcached_config.SEMCACHED_CACHE_ENABLE_ENV_VAR, "true")
    mock_makedirs = mock.MagicMock()
    monkeypatch.setattr(os, "makedirs", mock_makedirs)

    with caplog.at_level(logging.INFO):
        result = semcached_config.ensure_gptcache_initialized()

    assert result, "Should return True for successful initialization"
    assert semcached_config._gptcache_initialized, "Flag _gptcache_initialized should be True"
    assert "Semcached (gptcache integration) initialized successfully and is enabled." in caplog.text

    mock_cache_init_params = semcached_config.gptcache_cache.init_params
    assert mock_cache_init_params is not None, "gptcache.init should have been called"

    # Check the type of the embedding_func instance passed to the mock init
    assert isinstance(mock_cache_init_params["embedding_func_instance"], MockEmbedding)
    assert mock_cache_init_params["embedding_func_instance"].model_name == "onnx"

    assert isinstance(mock_cache_init_params["data_manager_instance"], MockManagerFactory)
    assert mock_cache_init_params["data_manager_instance"].data_dir == semcached_config.SEMCACHED_DATA_DIR_DEFAULT
    assert mock_cache_init_params["data_manager_instance"].vector_store_name == semcached_config.SEMCACHED_VECTOR_STORE_DEFAULT
    assert mock_cache_init_params["data_manager_instance"].data_storage_name == semcached_config.SEMCACHED_CACHE_STORAGE_DEFAULT

    assert mock_cache_init_params["similarity_threshold"] == semcached_config.SEMCACHED_SIMILARITY_THRESHOLD_DEFAULT

    mock_makedirs.assert_called_once_with(semcached_config.SEMCACHED_DATA_DIR_DEFAULT, exist_ok=True)

# Example of a test for OpenAI (requires API key, so usually skipped in CI unless key is available)
@pytest.mark.skipif(not os.getenv("SEMCACHED_TEST_WITH_OPENAI_KEY") or not os.getenv("OPENAI_API_KEY"),
                    reason="Test requires OPENAI_API_KEY and SEMCACHED_TEST_WITH_OPENAI_KEY to be set")
def test_ensure_gptcache_initialized_openai(reset_config_globals, monkeypatch, caplog):
    monkeypatch.setenv(semcached_config.SEMCACHED_CACHE_ENABLE_ENV_VAR, "true")
    monkeypatch.setenv(semcached_config.SEMCACHED_EMBEDDING_MODEL_ENV_VAR, "openai")
    # OPENAI_API_KEY should be set in the environment for this test to pass if not skipped

    mock_makedirs = mock.MagicMock()
    monkeypatch.setattr(os, "makedirs", mock_makedirs)

    with caplog.at_level(logging.INFO):
        result = semcached_config.ensure_gptcache_initialized()

    assert result, "Should return True for successful OpenAI initialization"
    assert "Semcached (gptcache integration) initialized successfully and is enabled." in caplog.text
    assert semcached_config.gptcache_cache.api_key == os.getenv("OPENAI_API_KEY")

    mock_cache_init_params = semcached_config.gptcache_cache.init_params
    assert mock_cache_init_params is not None
    assert isinstance(mock_cache_init_params["embedding_func_instance"], MockEmbedding)
    assert mock_cache_init_params["embedding_func_instance"].model_name == "openai"
```
