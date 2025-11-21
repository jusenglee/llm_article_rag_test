from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class RagResult:
    stack: str
    expanded_query: str
    keywords: List[str]
    hits: Any
    reranked_hits: Any
    context: str
    refs: List[str]
    timings: Dict[str, float]
    llm_answer: Optional[str] = None