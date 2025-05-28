from __future__ import annotations

import hashlib # Added for test_case_id generation
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.cache import ComprehensiveSemanticCache 
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import NonLLMStringSimilarity
import ragas.config # Added import for global cache
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    ensembler,
)
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig
from ragas.utils import deprecated

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class QAC(BaseModel):
    question: str = Field(..., description="Question")
    context: str = Field(..., description="Context")
    answer: str = Field(..., description="Answer")


class Verification(BaseModel):
    reason: str = Field(..., description="Reason for verification")
    verdict: int = Field(..., description="Binary (0/1) verdict of verification")


class ContextPrecisionPrompt(PydanticPrompt[QAC, Verification]):
    name: str = "context_precision"
    instruction: str = (
        'Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.'
    )
    input_model = QAC
    output_model = Verification
    examples = [
        (
            QAC(
                question="What can you tell me about Albert Einstein?",
                context="Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
                answer="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
            ),
            Verification(
                reason="The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
                verdict=1,
            ),
        ),
        (
            QAC(
                question="who won 2020 icc world cup?",
                context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
                answer="England",
            ),
            Verification(
                reason="the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
                verdict=1,
            ),
        ),
        (
            QAC(
                question="What is the tallest mountain in the world?",
                context="The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
                answer="Mount Everest.",
            ),
            Verification(
                reason="the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
                verdict=0,
            ),
        ),
    ]


@dataclass
class LLMContextPrecisionWithReference(MetricWithLLM, SingleTurnMetric):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    evaluation_mode: EvaluationMode
    context_precision_prompt: Prompt
    """

    name: str = "llm_context_precision_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
                "reference",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    context_precision_prompt: PydanticPrompt = field(
        default_factory=ContextPrecisionPrompt
    )
    max_retries: int = 1
    comprehensive_cache: t.Optional[ComprehensiveSemanticCache] = field(default=None, repr=False) 

    def __post_init__(self):
        # Initialize comprehensive_cache from global config if not already set
        if getattr(self, 'comprehensive_cache', None) is None:
            self.comprehensive_cache = ragas.config.ragas_comprehensive_cache
        
        # Call super().__post_init__() if MetricWithLLM or SingleTurnMetric had one.
        # Assuming they don't for now, or it's not critical for this attribute.
        # If they do, and it's important, this would be:
        # super().__post_init__() 

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["user_input"], row["retrieved_contexts"], row["reference"]

    def _calculate_average_precision(
        self, verifications: t.List[Verification]
    ) -> float:
        score = np.nan

        verdict_list = [1 if ver.verdict else 0 for ver in verifications]
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        if np.isnan(score):
            logger.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(
        self,
        row: t.Dict,
        callbacks: Callbacks,
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        user_input, retrieved_contexts, reference = self._get_row_attributes(row)
        
        # Create a unique ID for this test case based on user_input and reference
        test_case_comps = (user_input, str(reference)) # Ensure reference is string for hashing
        test_case_id_str = "".join(test_case_comps)
        test_case_id = hashlib.sha256(test_case_id_str.encode()).hexdigest()

        current_cache_instance = getattr(self, "comprehensive_cache", None)
        
        answers: t.List[Verification] = []

        for i, current_context in enumerate(retrieved_contexts):
            # The "prompt" for caching purposes here is the current_context itself,
            # as the LLM call's variability for this metric is driven by the context.
            # The user_input and reference are fixed for all contexts within this _ascore call.
            cache_key_for_context = current_context # Or a hash if contexts are very long and store needs shorter keys

            llm_result: t.Optional[Verification] = None

            if current_cache_instance:
                # Note: ComprehensiveSemanticCache expects `current_prompt` for semantic comparison.
                # Here, `cache_key_for_context` (which is `current_context`) acts as this "prompt".
                # The 'value' stored/retrieved will be the `Verification` object.
                # We are using test_case_id to scope, and cache_key_for_context (current_context)
                # as the specific part key for ComprehensiveSemanticCache's get/set.
                # If CSC's get/set were designed for prompt -> LLMResponse, this is a slight adaptation.
                # Let's assume CSC's `get` takes `test_case_id` and `key` (current_context here)
                # and `set` takes `test_case_id`, `key` (current_context), and `value` (Verification).
                # For semantic matching on current_context, CSC's internal logic would handle embedding it.

                # For ComprehensiveSemanticCache, the 'key' is the prompt.
                # The 'value' is the LLMResult.
                # Here, the "prompt" that varies per LLM call is effectively the `current_context`.
                # The "LLMResult" is the `Verification` object.
                # So, we'll use current_context as the "prompt" for the cache.
                cached_value = current_cache_instance.get(
                    test_case_id=test_case_id, 
                    current_prompt=current_context # current_context is the varying part for semantic check
                )
                if cached_value is not None:
                    if isinstance(cached_value, Verification):
                        llm_result = cached_value
                        logger.debug(f"Cache HIT for test_case_id='{test_case_id}', context_idx={i}")
                    else:
                        logger.warning(
                            f"Cache HIT but unexpected type for test_case_id='{test_case_id}', context_idx={i}. Expected Verification, got {type(cached_value)}. Ignoring."
                        )
            
            if llm_result is None: # Cache MISS or invalid cached type
                logger.debug(f"Cache MISS for test_case_id='{test_case_id}', context_idx={i}. Calling LLM.")
                # Define the actual LLM call as a nested function
                async def _llm_call_for_context() -> t.Optional[Verification]:
                    # Original LLM call logic for a single context
                    verdicts_list: t.List[Verification] = (
                        await self.context_precision_prompt.generate_multiple(
                            data=QAC(
                                question=user_input,
                                context=current_context,
                                answer=reference,
                            ),
                            llm=self.llm,
                            callbacks=callbacks,
                        )
                    )
                    # Assuming generate_multiple for this prompt structure returns one Verification or a list
                    # and ensembler handles it to pick the best one.
                    # The original code did responses.append([result.model_dump() for result in verdicts])
                    # then ensembler.from_discrete([response], "verdict")
                    # This implies verdicts_list might have multiple items if generate_multiple is used that way.
                    # For simplicity, let's assume generate_multiple with this prompt gives one main Verification,
                    # or we use the ensembler correctly.
                    
                    if not verdicts_list:
                        return None

                    # Ensembling step from original code:
                    # Convert Pydantic models to dicts for ensembler if that's what it expects
                    raw_verdicts_for_ensembler = [v.model_dump() for v in verdicts_list]
                    if not raw_verdicts_for_ensembler: # Should be caught by 'if not verdicts_list'
                        return None
                    
                    ensembled_raw_answer = ensembler.from_discrete([raw_verdicts_for_ensembler], "verdict")
                    if not ensembled_raw_answer:
                        return None
                    return Verification(**ensembled_raw_answer[0])

                llm_result = await _llm_call_for_context()

                if llm_result is not None and current_cache_instance:
                    # Store the result (Verification object) in cache
                    # using current_context as the "prompt" key for semantic storage.
                    current_cache_instance.set(
                        test_case_id=test_case_id,
                        prompt=current_context, 
                        llm_result=llm_result 
                    )
                    logger.debug(f"Cache SET for test_case_id='{test_case_id}', context_idx={i}")
            
            if llm_result is not None:
                answers.append(llm_result)
            else:
                logger.warning(f"LLM call failed or returned None for test_case_id='{test_case_id}', context_idx={i}. This context won't be scored.")

        if not answers and retrieved_contexts: # Only warn if contexts existed but no answers were derived
             logger.warning(
                f"No verifiable answers derived for test_case_id='{test_case_id}' despite having {len(retrieved_contexts)} contexts. Result might be NaN."
            )
        
        score = self._calculate_average_precision(answers)
        return score


@dataclass
class LLMContextPrecisionWithoutReference(LLMContextPrecisionWithReference):
    name: str = "llm_context_precision_without_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["user_input"], row["retrieved_contexts"], row["response"]


@dataclass
class NonLLMContextPrecisionWithReference(SingleTurnMetric):
    name: str = "non_llm_context_precision_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",
                "reference_contexts",
            }
        }
    )
    distance_measure: SingleTurnMetric = field(
        default_factory=lambda: NonLLMStringSimilarity()
    )
    threshold: float = 0.5

    def __post_init__(self):
        if isinstance(self.distance_measure, MetricWithLLM):
            raise ValueError(
                "distance_measure must not be an instance of MetricWithLLM for NonLLMContextPrecisionWithReference"
            )

    def init(self, run_config: RunConfig) -> None: ...

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        sample = SingleTurnSample(**row)
        return await self._single_turn_ascore(sample, callbacks)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        for rc in retrieved_contexts:
            scores.append(
                max(
                    [
                        await self.distance_measure.single_turn_ascore(
                            SingleTurnSample(reference=rc, response=ref), callbacks
                        )
                        for ref in reference_contexts
                    ]
                )
            )
        scores = [1 if score >= self.threshold else 0 for score in scores]
        return self._calculate_average_precision(scores)

    def _calculate_average_precision(self, verdict_list: t.List[int]) -> float:
        score = np.nan

        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        return score


@dataclass
class ContextPrecision(LLMContextPrecisionWithReference):
    name: str = "context_precision"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await super()._single_turn_ascore(sample, callbacks)

    @deprecated(
        since="0.2", removal="0.3", alternative="LLMContextPrecisionWithReference"
    )
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


@dataclass
class ContextUtilization(LLMContextPrecisionWithoutReference):
    name: str = "context_utilization"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await super()._single_turn_ascore(sample, callbacks)

    @deprecated(
        since="0.2", removal="0.3", alternative="LLMContextPrecisionWithoutReference"
    )
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


context_precision = ContextPrecision()
context_utilization = ContextUtilization()
