import json
import time
from typing import Any
from copy import deepcopy
from collections import defaultdict
from goodai.ltm.mem.auto import AutoTextMemory, DefaultTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.base import RetrievedMemory, PassageInfo
from multiprocessing import SimpleQueue, Process


def get(queue: SimpleQueue):
    v = queue.get()
    assert v is not None
    return v


def memory_server(
    mem_kwargs: dict, query_queue: SimpleQueue, add_queue: SimpleQueue,
    bkg_queue: SimpleQueue, processed_queue: SimpleQueue, out_queue: SimpleQueue,
):
    ltm = LTMSystem(**mem_kwargs)
    while True:
        # Give additions and changes some slack. Otherwise, queries might hoard time.
        # Add memories and send it to background processing
        # TODO: process these in random order or randomly drop some, to avoid bottlenecks
        for _ in range(10):
            if add_queue.empty():
                break
            kwargs = get(add_queue)
            content, metadata = ltm.content_addition_preprocessing(**kwargs)
            text_key = ltm.add_content(**kwargs)
            bkg_kwargs = dict(text=content, metadata=metadata, text_key=text_key)
            bkg_queue.put(bkg_kwargs)
        # Attend queries
        if not query_queue.empty():
            d = get(query_queue)
            assert d["method"] in {"retrieve", "retrieve_from_keywords"}
            mems = getattr(ltm, d["method"])(**d["kwargs"])
            out_queue.put(mems)
        # Take processed memories and update memory database
        for _ in range(10):
            if processed_queue.empty():
                break
            kwargs = get(processed_queue)
            ltm.semantic_memory.replace_text(**kwargs)
        # Yield to other processes
        time.sleep(0)


def background_process(bkg_queue: SimpleQueue, processed_queue: SimpleQueue):
    # TODO: implement limits & include configuration
    while True:
        kwargs = bkg_queue.get()
        kwargs["metadata"]["processed"] = True
        processed_queue.put(kwargs)


class RealTimeLTMSystem:
    def __init__(self):
        self.query_queue = SimpleQueue()
        self.out_queue = SimpleQueue()
        self.add_queue = SimpleQueue()
        self.bkg_queue = SimpleQueue()
        self.processed_queue = SimpleQueue()
        self.mem_server = Process(target=memory_server, kwargs=dict(
            mem_kwargs=dict(), query_queue=self.query_queue, out_queue=self.out_queue,
            add_queue=self.add_queue, bkg_queue=self.bkg_queue,
            processed_queue=self.processed_queue,
        ))
        self.mem_server.start()
        self.bkg_proc = Process(target=background_process, args=(self.bkg_queue, self.processed_queue))
        self.bkg_proc.start()

    def add_content(self, content: str, keywords: list[str] = None):
        self.add_queue.put(dict(content=content, keywords=keywords))
        time.sleep(0)

    def retrieve(
        self, query: str, k: int, max_distance: float = None,
    ) -> list[RetrievedMemory]:
        self.query_queue.put(dict(
            method="retrieve", kwargs=dict(query=query, k=k, max_distance=max_distance),
        ))
        time.sleep(0)
        mems = self.out_queue.get()
        return mems


class LTMSystem:

    def __init__(
        self, chunk_capacity: int = 50, chunk_overlap_fraction=0, **other_params,
    ):
        self.semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(
            chunk_capacity=chunk_capacity,
            chunk_overlap_fraction=chunk_overlap_fraction,
            **other_params,
        ))
        self.keyword_index = defaultdict(list)

    def is_empty(self) -> bool:
        return self.semantic_memory.is_empty()

    def clear(self):
        self.semantic_memory.clear()

    def state_as_text(self) -> str:
        return json.dumps(dict(
            semantic_memory=self.semantic_memory.state_as_text(),
            keyword_index=self.keyword_index,
        ))

    def set_state(self, state_text: str):
        state = json.loads(state_text)
        self.keyword_index = defaultdict(list, state["keyword_index"])
        self.semantic_memory.set_state(state["semantic_memory"])

    def add_content(self, content: str, keywords: list[str] = None) -> int:
        keywords = keywords or []
        content, metadata = self.content_addition_preprocessing(content, keywords)
        text_key = self.semantic_memory.add_text(content, metadata=metadata)
        self.semantic_memory.add_separator()
        for kw in keywords:
            self.keyword_index[kw].append(text_key)
        return text_key

    def retrieve(
        self, query: str, k: int, max_distance: float = None,
    ) -> list[RetrievedMemory]:
        memories = self.semantic_memory.retrieve(query, k)
        if max_distance is not None:
            memories = [m for m in memories if m.distance <= max_distance]
        return memories

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

    def content_addition_preprocessing(
        self, content: str, keywords: list[str] = None, **other_metadata: Any,
    ) -> tuple[str, dict[str, Any]]:
        metadata = deepcopy(other_metadata)
        metadata["keywords"] = keywords or []
        return content, metadata
