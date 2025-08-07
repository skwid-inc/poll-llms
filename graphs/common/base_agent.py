import abc
from typing import Any, Callable, List, Literal, Optional

from langchain_core.runnables import Runnable

State = Any


class BaseCollectionAgent(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the agent."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_tools(self) -> List[Callable]:
        """Returns the list of tools available to the agent."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_runnable(
        self, model: Optional[Any] = None, prompt: Optional[Any] = None
    ) -> Runnable:
        """Returns the runnable chain/agent executor for the agent."""
        raise NotImplementedError

    @abc.abstractmethod
    def route(self, state: State) -> Literal["__end__", str]:
        """Determines the next node in the graph based on the current state."""
        raise NotImplementedError

    def get_entry_message(self) -> Optional[str]:
        """
        Returns an optional entry message for the agent.
        Returns None if no specific entry message is required.
        """
        return None
