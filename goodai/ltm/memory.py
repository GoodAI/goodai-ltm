import json
import time
import queue
from typing import Any
from copy import deepcopy
from collections import defaultdict
from goodai.ltm.mem.auto import AutoTextMemory, DefaultTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.base import RetrievedMemory, PassageInfo
from multiprocessing import Queue, Process


def memory_server(
    mem_kwargs: dict, time_budget: float, query_queue: Queue, add_queue: Queue,
    bkg_queue: Queue, processed_queue: Queue, out_queue: Queue,
):
    ltm = LTMSystem(**mem_kwargs)
    allowed_methods = {"is_empty", "clear", "state_as_text", "set_state", "retrieve",
                       "retrieve_from_keywords"}
    while True:
        t0_loop = time.time()
        # Attend queries and other priority actions
        try:
            query_dict = query_queue.get(block=False)
            assert query_dict["method"] in allowed_methods
            return_value = getattr(ltm, query_dict["method"])(**query_dict["kwargs"])
            out_queue.put(return_value)
        except queue.Empty:
            pass
        # Give additions and changes some slack. Otherwise, queries might hoard time.
        while True:
            queues_empty = True
            t0_changes = time.time()
            # Add memories and send it to background processing
            # TODO: process these in random order or randomly drop some, to avoid bottlenecks
            try:
                kwargs = add_queue.get(block=False)
                content, metadata = ltm.content_addition_preprocessing(**kwargs)
                text_key = ltm.add_content(**kwargs)
                bkg_kwargs = dict(text=content, metadata=metadata, text_key=text_key)
                bkg_queue.put(bkg_kwargs)
                queues_empty = False
            except queue.Empty:
                pass
            # Take processed memories and update memory database
            try:
                kwargs = processed_queue.get(block=False)
                ltm.semantic_memory.replace_text(**kwargs)
                queues_empty = False
            except queue.Empty:
                pass
            t_changes = time.time() - t0_changes
            # See if there's time for another round
            if queues_empty or time.time() - t0_loop + t_changes >= time_budget:
                break


def background_process(bkg_queue: Queue, processed_queue: Queue):
    # TODO: implement limits & include configuration
    while True:
        kwargs = bkg_queue.get()
        kwargs["metadata"]["processed"] = True
        processed_queue.put(kwargs)


class RealTimeLTMSystem:
    def __init__(self, time_budget: float = 1):
        self.query_queue = Queue()
        self.out_queue = Queue()
        self.add_queue = Queue()
        self.bkg_queue = Queue()
        self.processed_queue = Queue()
        self.mem_server = Process(daemon=True, target=memory_server, kwargs=dict(
            mem_kwargs=dict(), time_budget=time_budget, query_queue=self.query_queue,
            out_queue=self.out_queue, add_queue=self.add_queue,
            bkg_queue=self.bkg_queue, processed_queue=self.processed_queue,
        ))
        self.mem_server.start()
        self.bkg_proc = Process(daemon=True, target=background_process,
                                args=(self.bkg_queue, self.processed_queue))
        self.bkg_proc.start()

    def _sync_call(self, method: str, **kwargs):
        self.query_queue.put(dict(method=method, kwargs=kwargs))
        return self.out_queue.get()

    def is_empty(self) -> bool:
        return self._sync_call("is_empty")

    def clear(self):
        return self._sync_call("clear")

    def state_as_text(self) -> str:
        return self._sync_call("state_as_text")

    def set_state(self, state_text: str):
        return self._sync_call("set_state", state_text=state_text)

    def add_content(
        self, content: str, timestamp: float = None, keywords: list[str] = None,
        **metadata: Any,
    ):
        self.add_queue.put(
            dict(content=content, timestamp=timestamp, keywords=keywords) | metadata
        )

    def retrieve(
        self, query: str, k: int, max_distance: float = None,
    ) -> list[RetrievedMemory]:
        return self._sync_call("retrieve", query=query, k=k, max_distance=max_distance)

    def retrieve_from_keywords(self, keywords: list[str]) -> list[RetrievedMemory]:
        assert len(keywords) > 0
        return self._sync_call("retrieve_from_keywords", keywords=keywords)

    def __end__(self):
        self.mem_server.terminate()


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

    def add_content(
        self, content: str, timestamp: float = None, keywords: list[str] = None,
        **metadata: Any,
    ) -> int:
        keywords = keywords or []
        content, metadata = self.content_addition_preprocessing(content, keywords, **metadata)
        text_key = self.semantic_memory.add_text(
            content, timestamp=timestamp, metadata=metadata,
        )
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
            if len(text_keys) == 0:
                break
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
        assert "keywords" not in other_metadata
        metadata = deepcopy(other_metadata)
        metadata["keywords"] = keywords or []
        return content, metadata
