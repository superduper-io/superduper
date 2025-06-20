from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class IntegrationStatus:
    """Status information for a secret."""
    Name: str
    Phase: str
    Reason: str
    Msg: str

@dataclass_json
@dataclass
class IntegrationStatusList:
    Items: List[IntegrationStatus]

class RuntimeEnvironment(ABC):
    """
    Abstract base class for the environment where services are running, and the capabilities it provides.
    """
    @abstractmethod
    def name(self) -> str:
        """Return the name of the runtime environment."""
        pass

    @abstractmethod
    def check_integrations(self, required_integrations: List[str]) -> IntegrationStatusList:
        pass

    @abstractmethod
    def load_integrations(self, required_integrations: List[str]) -> None:
        pass