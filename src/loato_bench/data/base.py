"""Base classes for the data pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedSample:
    """A single sample in the unified dataset format.

    All dataset loaders must produce instances of this class.
    """

    text: str
    label: int  # 0=benign, 1=injection
    source: str
    attack_category: str | None = None
    original_category: str | None = None
    language: str = "en"
    is_indirect: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Each concrete loader handles one source dataset (Deepset, HackAPrompt, etc.)
    and returns a list of UnifiedSample instances.
    """

    @abstractmethod
    def load(self) -> list[UnifiedSample]:
        """Load and return all samples from this dataset source."""
        ...
