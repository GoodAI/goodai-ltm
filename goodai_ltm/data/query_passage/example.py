from dataclasses import dataclass
from typing import List


@dataclass
class QueryPassageExample:
    queryIds: List[int]
    passageIds: List[int]
    match: bool

