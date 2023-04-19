from typing import List, Iterable, Any, TypeVar, Callable

_T_item = TypeVar('_T_item')


def remove_duplicates(x: Iterable[_T_item], key_fn: Callable[[_T_item], Any]) -> List[Any]:
    result = []
    visited = set()
    for item in x:
        key = key_fn(item)
        if key not in visited:
            result.append(item)
            visited.add(key)
    return result


def num_visited_to_get_expected_count(x: List[_T_item], expected_count: int,
                                      adjacent_ok: bool, key_fn: Callable[[_T_item], int], ) -> int:
    result_count = 0
    visited = set()
    for i, item in enumerate(x):
        index = key_fn(item)
        extra_cond = True if adjacent_ok else (index + 1) not in visited and (index - 1) not in visited
        if index not in visited and extra_cond:
            result_count += 1
            if result_count >= expected_count:
                return i + 1
            visited.add(index)
    return len(x)


def get_non_adjacent(x: List[_T_item], key_fn: Callable[[_T_item], int]) -> List[_T_item]:
    result = []
    visited = set()
    for item in x:
        index = key_fn(item)
        if index not in visited and (index + 1) not in visited and (index - 1) not in visited:
            result.append(item)
            visited.add(index)
    return result
