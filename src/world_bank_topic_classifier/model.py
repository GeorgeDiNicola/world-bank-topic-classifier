from dataclasses import dataclass
from typing import List

@dataclass
class Indicator:
    id: int
    name: str
    source_id: str
    source: str
    source_organization: str
    topics: List[str]

@dataclass
class CategorizedIndicator:
    id: int
    name: str
    source_id: str
    source: str
    source_organization: str
    topic: str