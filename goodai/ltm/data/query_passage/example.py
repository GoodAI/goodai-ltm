from dataclasses import dataclass
from typing import List


@dataclass
class QueryPassageExample:
    queryIds: List[int]
    """
    Query token IDs
    """

    passageIds: List[int]
    """
    Passage token IDs
    """

    match: bool
    """
    Whether the query and the passage are a good match
    """
