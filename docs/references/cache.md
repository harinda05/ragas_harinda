# Caching in Ragas

Ragas provides a flexible caching mechanism to store and retrieve results of expensive computations, such as LLM calls or embedding generations. This can save time and costs during repeated evaluations or development iterations.

## Global Cache Configuration

Caching behavior in Ragas is primarily controlled by a global cache object, `ragas.config.ragas_cache`. This object is initialized based on environment variables when Ragas is imported. You can configure the cache by setting these environment variables before running your Ragas application:

*   **`RAGAS_CACHE_ENABLED`**: (boolean, default: "true")
    *   Set to "false" to disable caching entirely.
*   **`RAGAS_CACHE_BACKEND`**: (string, default: "exact")
    *   Determines the type of cache backend to use.
    *   `"exact"`: Uses `DiskCacheBackend` for exact matching of serialized inputs.
    *   `"semantic"`: Uses `SemanticCacheBackend` for similarity-based matching.
*   **`RAGAS_CACHE_DIR`**: (string, default: ".ragas_cache")
    *   Directory used by `DiskCacheBackend` to store cache files.
*   **`RAGAS_SEMANTIC_CACHE_THRESHOLD`**: (float, default: 0.85)
    *   Cosine similarity threshold for `SemanticCacheBackend`. Hits occur if similarity is `>=` this value.
*   **`RAGAS_SEMANTIC_CACHE_EMBEDDING_PROVIDER`**: (string, default: "openai")
    *   Specifies the embedding provider for `SemanticCacheBackend` (e.g., "openai", "huggingface").
*   **`RAGAS_SEMANTIC_CACHE_EMBEDDING_MODEL_NAME`**: (string, optional)
    *   Specifies the exact model name for the chosen embedding provider.
    *   If not set, defaults to "text-embedding-ada-002" for OpenAI or "sentence-transformers/all-MiniLM-L6-v2" for Huggingface.

## The `cacher` Decorator

Ragas uses a decorator pattern to apply caching to functions.

::: ragas.cache.cacher
    options:
        show_signature: true
        show_docstring: true

By default, the `@cacher()` decorator (when called without arguments) will use the globally configured `ragas.config.ragas_cache`. You can also pass a specific cache instance to it, e.g., `@cacher(my_custom_cache_instance)`.

## Cache Backends

Ragas includes different cache backend implementations that conform to the `CacheInterface`.

### `CacheInterface`

This is the abstract base class defining the interface for all cache backends.

::: ragas.cache.CacheInterface
    options:
        members: ["get", "set", "has_key"]
        show_signature: true
        show_docstring: true
        members_order: "source"

### `DiskCacheBackend`

This backend provides exact caching by storing results on disk. It serializes function arguments to create a key and stores the output. If the same function is called with the exact same arguments, the result is retrieved from the disk.

It is the default backend when `RAGAS_CACHE_BACKEND` is set to `"exact"` or not set.

::: ragas.cache.DiskCacheBackend
    options:
        show_signature: true
        show_docstring: true
        members_order: "source"

### `SemanticCacheBackend`

This backend provides a more advanced caching mechanism that combines semantic similarity with exact matching. It's particularly useful for LLM calls where slight variations in prompts (e.g., paraphrasing) might still yield the same or very similar results.

When `RAGAS_CACHE_BACKEND` is set to `"semantic"`, the global cache will be an instance of `SemanticCacheBackend`.

::: ragas.cache.SemanticCacheBackend
    options:
        show_signature: true
        show_docstring: true
        members_order: "source"

**How `SemanticCacheBackend` Works:**

1.  **Key Generation**: The `@cacher` decorator generates a JSON string key representing the function call (function name, args, kwargs).
2.  **Key Parsing**: `SemanticCacheBackend` parses this JSON key. It expects a structure like `{"function": "func_name", "args": [...], "kwargs": {...}}`.
3.  **Semantic Part Identification**: It identifies the first string argument within the `"args"` list as the "semantic part" of the key. This part is used for generating an embedding.
4.  **Embedding Generation**: The identified semantic part is converted into a vector embedding using the configured `embedding_model`.
5.  **Similarity Search**:
    *   When checking for a cache hit (`get` or `has_key`), the embedding of the input key's semantic part is compared against embeddings of previously cached keys using cosine similarity.
    *   If the similarity between the input embedding and a cached embedding is greater than or equal to `similarity_threshold`, it's considered a potential semantic match.
6.  **Exact Match Verification**: For each potential semantic match, `SemanticCacheBackend` then performs an exact match on the non-semantic parts of the key:
    *   Function name must be identical.
    *   Keyword arguments (`kwargs`) must be identical.
    *   All other positional arguments (`args`, excluding the one used for semantic comparison) must be identical and in the same order.
7.  **Cache Hit/Miss**: A cache hit occurs only if both the semantic similarity condition and the exact match verification pass. The corresponding stored value is then returned. If no such item is found, it's a cache miss.
8.  **Storing**: When setting a new item, the embedding of the semantic part, the original JSON key string, the value, and the index of the semantic argument are stored.

This hybrid approach ensures that the cache is resilient to minor, semantically equivalent changes in one part of the input while maintaining strictness for other parameters.

## `SentenceEvaluatorSemanticCache`

This is a specialized semantic cache designed for use within Ragas metrics that evaluate individual sentences or statements against a given context (e.g., the `Faithfulness` metric). It helps avoid redundant LLM calls by caching the evaluation result (e.g., a verdict) for a sentence if a semantically similar sentence has already been evaluated against the exact same context.

::: ragas.cache.SentenceEvaluatorSemanticCache
    options:
        show_signature: true
        show_docstring: true
        members: ["get", "set"] # Only show relevant public methods
        members_order: "source"

**Key Concepts and Usage:**

*   **Purpose**: To cache the evaluation (often an LLM-based verdict) of a specific statement in relation to a fixed primary context and an optional secondary context.
*   **Semantic Matching for Sentences**: The cache determines hits based on the semantic similarity of the `current_sentence` being evaluated to previously cached sentences (`sentence_to_cache`).
*   **Exact Matching for Context**: Crucially, the `primary_context_or_hash` and `secondary_context_or_hash` must match *exactly* for a cache hit. This ensures that semantic similarity is only considered when the surrounding context is identical.
    *   `primary_context_or_hash`: Typically, this is a hash of the main retrieved context against which the sentence is evaluated. Using a hash avoids storing potentially very long context strings directly in the cache key comparison logic (though the cache implementation might store it for debugging).
    *   `secondary_context_or_hash` (Optional): This can be used if there's another piece of context that needs to be an exact match, for example, a hash of a question if sentence evaluation is question-dependent beyond the primary context.
*   **Integration with Metrics**: Metrics like `Faithfulness` can be configured to use an instance of `SentenceEvaluatorSemanticCache`. If a `sentence_cache` attribute is set on the metric instance, it will be used.

**How it Works (Simplified):**

1.  **`set` Operation**:
    *   When a sentence is evaluated (e.g., by an LLM call within a metric), the `sentence_to_cache`, its embedding, the `primary_context_or_hash`, the `secondary_context_or_hash` (if any), and the `llm_output_for_sentence` (the result to cache) are stored.
2.  **`get` Operation**:
    *   When evaluating a `current_sentence`:
        *   Its embedding is generated.
        *   The cache is searched for entries where `primary_context_or_hash` and `secondary_context_or_hash` match the current ones.
        *   For these matching entries, the embedding of `current_sentence` is compared to the stored sentence embeddings.
        *   If the cosine similarity exceeds `similarity_threshold`, a cache hit occurs, and the stored `llm_output_for_sentence` is returned.

### Global Configuration for Sentence Evaluation Cache

Similar to the general Ragas cache, a global instance of `SentenceEvaluatorSemanticCache` can be configured via environment variables. This global instance is available at `ragas.config.ragas_sentence_eval_cache`.

Metrics that support sentence-level caching (like `Faithfulness`) may be designed to automatically use this global instance if their specific `sentence_cache` attribute is not set manually.

The following environment variables control its behavior:

*   **`RAGAS_SENTENCE_EVAL_CACHE_ENABLED`**: (boolean, default: "true")
    *   Set to "false" to disable this specific sentence evaluation cache.
*   **`RAGAS_SENTENCE_EVAL_EMBEDDING_PROVIDER`**: (string, default: "openai")
    *   Embedding provider (e.g., "openai", "huggingface") for sentence similarity.
*   **`RAGAS_SENTENCE_EVAL_EMBEDDING_MODEL_NAME`**: (string, optional)
    *   Specific model name for the chosen embedding provider. Defaults to standard models like "text-embedding-ada-002" for OpenAI.
*   **`RAGAS_SENTENCE_EVAL_SIMILARITY_THRESHOLD`**: (float, default: 0.80)
    *   Cosine similarity threshold for considering sentences semantically similar.

Setting these variables allows for global control over sentence-level caching within applicable metrics without needing to manually instantiate and assign cache objects to each metric.
