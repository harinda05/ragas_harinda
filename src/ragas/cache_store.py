from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class TestCasePartCacheStore(ABC):
    """
    Abstract base class defining the interface for a cache store
    that handles parts of a test case.

    This cache is designed to store and retrieve values associated with
    a specific part of a test case, identified by the test case's unique ID
    and a key for the specific part.
    """

    @abstractmethod
    def get(self, test_case_id: str, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache.

        Parameters
        ----------
        test_case_id : str
            The unique identifier for the test case.
        key : str
            The key identifying the specific part of the test case.

        Returns
        -------
        Optional[Any]
            The cached value if found, otherwise None.
        """
        ...


# --- InMemoryStore Implementation ---
import logging
from typing import Dict, Tuple # Optional and Any are already imported at the top

logger = logging.getLogger(__name__)

class InMemoryStore(TestCasePartCacheStore):
    """
    An in-memory implementation of `TestCasePartCacheStore`.

    This class stores test case parts in a Python dictionary.
    Data will be lost when the process terminates.
    """

    def __init__(self):
        # The cache is a dictionary where:
        #   - keys are test_case_id (str)
        #   - values are another dictionary where:
        #     - keys are part_key (str)
        #     - values are the cached_value (Any)
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info("InMemoryStore initialized.")

    def get(self, test_case_id: str, key: str) -> Optional[Any]:
        if test_case_id in self._cache and key in self._cache[test_case_id]:
            logger.debug(f"Cache HIT for test_case_id='{test_case_id}', key='{key}'")
            return self._cache[test_case_id][key]
        else:
            logger.debug(f"Cache MISS for test_case_id='{test_case_id}', key='{key}'")
            return None

    def set(self, test_case_id: str, key: str, value: Any) -> None:
        if test_case_id not in self._cache:
            self._cache[test_case_id] = {}
        self._cache[test_case_id][key] = value
        logger.debug(f"Cache SET for test_case_id='{test_case_id}', key='{key}'")

    def has_key(self, test_case_id: str, key: str) -> bool:
        exists = test_case_id in self._cache and key in self._cache[test_case_id]
        logger.debug(f"Cache HAS_KEY for test_case_id='{test_case_id}', key='{key}': {exists}")
        return exists

    def clear(self, test_case_id: Optional[str] = None) -> None:
        if test_case_id is not None:
            if test_case_id in self._cache:
                del self._cache[test_case_id]
                logger.info(f"Cleared cache for test_case_id='{test_case_id}'")
            else:
                logger.info(f"No cache found for test_case_id='{test_case_id}' to clear.")
        else:
            self._cache.clear()
            logger.info("Cleared all items from InMemoryStore.")

    @abstractmethod
    def set(self, test_case_id: str, key: str, value: Any) -> None:
        """
        Stores an item in the cache.

        Parameters
        ----------
        test_case_id : str
            The unique identifier for the test case.
        key : str
            The key identifying the specific part of the test case.
        value : Any
            The value to store.
        """
        ...

    @abstractmethod
    def has_key(self, test_case_id: str, key: str) -> bool:
        """
        Checks if an item exists in the cache.

        Parameters
        ----------
        test_case_id : str
            The unique identifier for the test case.
        key : str
            The key identifying the specific part of the test case.

        Returns
        -------
        bool
            True if the item exists, False otherwise.
        """
        ...

    @abstractmethod
    def clear(self, test_case_id: Optional[str] = None) -> None:
        """
        Clears items from the cache.

        If `test_case_id` is provided, only items associated with that
        test case are cleared. If `test_case_id` is None, all items
        in the cache are cleared.

        Parameters
        ----------
        test_case_id : Optional[str], optional
            The unique identifier for the test case to clear.
            If None, clears the entire cache. Defaults to None.
        """
        ...
