import json
from collections import defaultdict
from goodai.ltm.mem.auto import AutoTextMemory, DefaultTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.base import RetrievedMemory, PassageInfo


class LTMSystem:

    def __init__(
        self, max_retrieve_capacity: int = 2000, chunk_capacity: int = 50,
        chunk_overlap_fraction=0, **other_params,
    ):
        self.max_retrieve_capacity = max_retrieve_capacity
        self.semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(
            chunk_capacity=chunk_capacity,
            chunk_overlap_fraction=chunk_overlap_fraction,
            **other_params,
        ))
        self.keyword_index = defaultdict(list)

    def add_content(self, content: str, timestamp: float = None, keywords: list[str] = None):
        keywords = keywords or []
        text_key = self.semantic_memory.add_text(
            content, timestamp=timestamp, metadata={"keywords": keywords}
        )
        self.semantic_memory.add_separator()
        for kw in keywords:
            self.keyword_index[kw].append(text_key)

    def is_empty(self) -> bool:
        return self.semantic_memory.is_empty()

    def clear(self):
        self.semantic_memory.clear()

    def state_as_text(self) -> str:
        return json.dumps(dict(
            max_retrieve_capacity=self.max_retrieve_capacity,
            semantic_memory=self.semantic_memory.state_as_text(),
            keyword_index=self.keyword_index,
        ))

    def set_state(self, state_text: str):
        state = json.loads(state_text)
        self.max_retrieve_capacity = state["max_retrieve_capacity"]
        self.keyword_index = defaultdict(list, state["keyword_index"])
        self.semantic_memory.set_state(state["semantic_memory"])

    def retrieve(self, query: str, k: int) -> list[RetrievedMemory]:
        return self.semantic_memory.retrieve(query, k)

    def retrieve_from_keywords(self, keywords: list[str]) -> list[RetrievedMemory]:
        assert len(keywords) > 0
        text_keys = set()
        for kw in keywords:
            text_keys.update(self.keyword_index[kw])
        mem: DefaultTextMemory = self.semantic_memory
        memories = list()
        for chunk in mem.get_all_chunks():
            assert len(chunk.associated_keys) == 1
            text_key = chunk.associated_keys[0]
            if text_key not in text_keys:
                continue
            text_keys.remove(text_key)
            token_ids = mem.chunk_queue.get_sequence_token_ids(text_key)
            text = mem.chunk_tokenizer.decode(token_ids, skip_special_tokens=True)
            matching_keywords = set(chunk.metadata["keywords"]) & set(keywords)
            relevance = len(matching_keywords) / len(keywords)
            memories.append(RetrievedMemory(
                passage=text,
                passage_info=PassageInfo(
                    chunk.from_token_seq_id, chunk.to_token_seq_id, token_ids,
                ),
                timestamp=chunk.timestamp,
                distance=1 - relevance,
                relevance=relevance,
                textKeys=[text_key],
                metadata=chunk.metadata,
            ))
        return memories
