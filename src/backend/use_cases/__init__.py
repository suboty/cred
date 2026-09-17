from abc import ABC, abstractmethod
from typing import Any


class UseCase(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass
