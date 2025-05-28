## Faithfulness

The **Faithfulness** metric measures how factually consistent a `response` is with the `retrieved context`. It ranges from 0 to 1, with higher scores indicating better consistency.  

A response is considered **faithful** if all its claims can be supported by the retrieved context.  

To calculate this:  
1. Identify all the claims in the response.  
2. Check each claim to see if it can be inferred from the retrieved context.  
3. Compute the faithfulness score using the formula:  

$$
\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}
$$


### Example

```python
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import Faithfulness

sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
scorer = Faithfulness(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```
Output
```
1.0
```


## Faithfullness with HHEM-2.1-Open

[Vectara's HHEM-2.1-Open](https://vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model/) is a classifier model (T5) that is trained to detect hallucinations from LLM generated text. This model can be used in the second step of calculating faithfulness, i.e. when claims are cross-checked with the given context to determine if it can be inferred from the context. The model is free, small, and open-source, making it very efficient in production use cases. To use the model to calculate faithfulness, you can use the following code snippet:

```python
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import FaithfulnesswithHHEM


sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)

```

You can load the model onto a specified device by setting the `device` argument and adjust the batch size for inference using the `batch_size` parameter. By default, the model is loaded on the CPU with a batch size of 10

```python

my_device = "cuda:0"
my_batch_size = 10

scorer = FaithfulnesswithHHEM(device=my_device, batch_size=my_batch_size)
await scorer.single_turn_ascore(sample)
```


### How It’s Calculated 

!!! example
    **Question**: Where and when was Einstein born?

    **Context**: Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time

    **High faithfulness answer**: Einstein was born in Germany on 14th March 1879.

    **Low faithfulness answer**:  Einstein was born in Germany on 20th March 1879.

Let's examine how faithfulness was calculated using the low faithfulness answer:

- **Step 1:** Break the generated answer into individual statements.
    - Statements:
        - Statement 1: "Einstein was born in Germany."
        - Statement 2: "Einstein was born on 20th March 1879."

- **Step 2:** For each of the generated statements, verify if it can be inferred from the given context.
    - Statement 1: Yes
    - Statement 2: No

- **Step 3:** Use the formula depicted above to calculate faithfulness.

    $$
    \text{Faithfulness} = { \text{1} \over \text{2} } = 0.5
    $$

## Optimizing with Sentence-Level Semantic Caching

The `Faithfulness` metric evaluates individual statements generated from the response against the provided context. This process, especially the verification step (`_create_verdicts`), can involve multiple LLM calls for each statement. To optimize this and reduce redundant computations, `Faithfulness` can leverage a specialized cache: `SentenceEvaluatorSemanticCache`.

This cache stores the verdict for a statement based on its semantic similarity to previously evaluated statements, but only if the surrounding context remains identical.

### How `Faithfulness` Uses `SentenceEvaluatorSemanticCache`

1.  **Context Hashing**: For each (question, answer, retrieved_contexts) group, the `retrieved_contexts` are concatenated and hashed. This hash serves as the `primary_context_or_hash` for the `SentenceEvaluatorSemanticCache`. This ensures that statement verdicts are only compared if they were made against the exact same body of retrieved context.
2.  **Statement Evaluation**: When `_create_verdicts` is called:
    *   For each statement, it first checks the `sentence_cache` using the statement text and the context hash.
    *   If a semantically similar statement (above the cache's `similarity_threshold`) is found for the *exact same context hash*, the cached verdict is used.
    *   Only statements not found in the cache (or not meeting the similarity threshold) are sent to the LLM for evaluation.
    *   New verdicts obtained from the LLM are then stored in the `sentence_cache` for future use.

### Enabling Sentence-Level Caching for `Faithfulness`

You can enable sentence-level caching for a `Faithfulness` metric instance by assigning a `SentenceEvaluatorSemanticCache` object to its `sentence_cache` attribute.

**1. Using the Globally Configured Sentence Cache:**

Ragas provides a globally configured instance at `ragas.config.ragas_sentence_eval_cache`. You can assign this to your metric:

```python
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper # Assuming you have an LLM wrapper
from langchain_openai import ChatOpenAI # Example LLM
import ragas.config # Import the config module

# Initialize your LLM
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-3.5-turbo"))

# Initialize Faithfulness metric
faithfulness_metric = Faithfulness(llm=evaluator_llm)

# Enable sentence-level caching using the global cache
# This global cache is configured by RAGAS_SENTENCE_EVAL_* environment variables
if ragas.config.ragas_sentence_eval_cache is not None:
    faithfulness_metric.sentence_cache = ragas.config.ragas_sentence_eval_cache
    print("Faithfulness metric is now using the global sentence evaluation cache.")
else:
    print("Global sentence evaluation cache is not enabled/configured.")

# Now, when you use faithfulness_metric.ascore(...) or evaluate(...),
# it will attempt to use this cache for sentence verdicts.
```

**2. Using a Manually Configured `SentenceEvaluatorSemanticCache`:**

If you need more specific control or want to use a different configuration than the global one:

```python
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.cache import SentenceEvaluatorSemanticCache
from ragas.embeddings import OpenAIEmbeddings # Or any other Ragas-compatible embedding model

# Initialize your LLM
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-3.5-turbo"))

# Initialize your embedding model for the cache
# Ensure API keys are set if using cloud-based embeddings like OpenAI
my_sentence_embed_model = OpenAIEmbeddings(model_name="text-embedding-ada-002")

# Create and configure the sentence cache instance
manual_sentence_cache = SentenceEvaluatorSemanticCache(
    embedding_model=my_sentence_embed_model,
    similarity_threshold=0.85 # Adjust as needed
)

# Initialize Faithfulness metric and assign the manual cache
faithfulness_metric_manual_cache = Faithfulness(llm=evaluator_llm)
faithfulness_metric_manual_cache.sentence_cache = manual_sentence_cache
print("Faithfulness metric is now using a manually configured sentence evaluation cache.")

# This instance will use 'manual_sentence_cache'.
```

By enabling sentence-level caching, you can significantly speed up `Faithfulness` evaluations, especially on large datasets or when re-evaluating with minor changes, while ensuring that semantic nuances are still considered for cache hits. Refer to the [Caching Reference (`docs/references/cache.md`)](../references/cache.md) for more details on `SentenceEvaluatorSemanticCache` and its global configuration.
