import builtins
import codecs
import dataclasses
import enum
import importlib
import json
import re
from json import JSONDecodeError, JSONEncoder, JSONDecoder
from typing import Any, Type, Optional
import numpy as np

_javascript_re = re.compile(r"^.*```javascript(.*)```.*$", re.MULTILINE | re.DOTALL)
_json_re = re.compile(r"^.*```(?:json)?(.*)```.*$", re.MULTILINE | re.DOTALL)


def sanitize_and_parse_json(content: str) -> Any:
    try:
        return json.loads(content)
    except JSONDecodeError:
        return _step1_sanitize_and_parse_json(content)


def _step1_sanitize_and_parse_json(content: str) -> Any:
    match_json = _json_re.search(content)
    match_js = _javascript_re.search(content)
    if match_js:
        # Check Javascript first
        stripped_json = match_js.group(1)
    elif match_json:
        stripped_json = match_json.group(1)
    else:
        stripped_json = content
    countdown_start = max(5, len(stripped_json) // 50)
    return _step2_sanitize_and_parse_json(stripped_json, countdown=countdown_start)


def _step2_sanitize_and_parse_json(content: str, countdown: int) -> Any:
    if countdown <= 0:
        raise ValueError(f"Unable to sanitize JSON after a number of iterations: {content}")
    content = content.strip()
    try:
        return json.loads(content)
    except JSONDecodeError as error:
        msg = error.msg
        if msg.startswith("Extra data"):
            return _step2_sanitize_and_parse_json(content[:error.pos], countdown=countdown - 1)
        elif msg.startswith("Expecting value"):
            if error.pos == 0:
                return _handle_extra_beginning_text(content, countdown=countdown)
            else:
                remain = content[error.pos:].lstrip()
                if remain.startswith("//"):
                    return _handle_extra_line_comment(content, error.pos, countdown=countdown)
                elif remain.startswith("..."):
                    return _handle_ellipsis(content, error.pos, countdown=countdown)
                elif remain.startswith("]"):
                    return _handle_comma_before_brace(content, error.pos, countdown=countdown)
                else:
                    return _handle_missing_seq(content, error.pos, '"', countdown=countdown)
        elif msg.startswith("Expecting property name enclosed in double quotes"):
            remain = content[error.pos:].lstrip()
            if remain.startswith("..."):
                return _handle_ellipsis(content, error.pos, countdown=countdown)
            elif remain.startswith("}"):
                return _handle_comma_before_brace(content, error.pos, countdown=countdown)
            elif remain.startswith("//"):
                return _handle_extra_line_comment(content, error.pos, countdown=countdown)
            else:
                raise
        elif msg.startswith("Expecting ',' delimiter"):
            return _handle_missing_seq(content, error.pos, ",", countdown=countdown)
        elif msg.startswith("Expecting ':' delimiter"):
            remain = content[error.pos:].lstrip()
            if remain.startswith("}") or remain.startswith(","):
                return _handle_missing_seq(content, error.pos, ": null", countdown=countdown)
            else:
                raise
        elif msg.startswith("Invalid control character at"):
            cc = content[error.pos]
            if ord(cc) in [10]:
                return _handle_extra_char(content, error.pos, countdown=countdown)
            else:
                raise
        elif msg.startswith("Expecting property name enclosed in double quotes"):
            remain = content[error.pos:]
            if remain.startswith("//"):
                return _handle_remove_to_line_break(content, error.pos, countdown=countdown)
            elif remain.startswith("}"):
                return _handle_comma_before_brace(content, error.pos, countdown=countdown)
            else:
                raise
        else:
            raise


def _handle_ellipsis(content: str, error_pos: int, countdown: int) -> Any:
    expected_text = "..."
    idx_ellipsis = content.find(expected_text, error_pos)
    if idx_ellipsis == -1:
        raise ValueError(f"Expected ellipsis at position {error_pos}: {content}")
    new_content = content[:idx_ellipsis] + content[idx_ellipsis + len(expected_text):]
    return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_extra_line_comment(content: str, error_pos: int, countdown: int) -> Any:
    idx_comment = content.find("//", error_pos)
    if idx_comment == -1:
        raise ValueError(f"Expected line comment marker at position {error_pos}: {content}")
    idx_line_break = content.find('\n', idx_comment)
    if idx_line_break == -1:
        raise ValueError(f"Expected line break after position {idx_comment}: {content}")
    new_content = content[:idx_comment] + content[idx_line_break+1:]
    return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_comma_before_brace(content: str, error_pos: int, countdown: int) -> Any:
    comma_idx = content.rfind(",", 0, error_pos + 1)
    if comma_idx == -1:
        raise ValueError(f"Unable to resolve JSON error at position {error_pos}: {content}")
    else:
        new_content = content[:comma_idx] + content[comma_idx + 1:]
        return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_remove_to_line_break(content: str, error_pos: int, countdown: int) -> Any:
    lb_pos = content.find('\n', error_pos)
    if lb_pos == -1:
        raise ValueError(f"Unable to resolve JSON error at position {error_pos}: {content}")
    else:
        new_content = content[:error_pos] + content[lb_pos+1:]
        return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_missing_seq(content: str, error_pos: int, sub: str, countdown: int) -> Any:
    new_content = content[:error_pos] + sub + content[error_pos:]
    return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_extra_char(content: str, error_pos: int, countdown: int) -> Any:
    new_content = content[:error_pos] + content[(error_pos + 1):]
    return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_extra_beginning_text(content: str, countdown: int) -> Any:
    idx_brace = content.find('{')
    idx_bracket = content.find('[')
    if idx_brace != -1 and idx_bracket != -1:
        start_at = idx_brace if idx_brace < idx_bracket else idx_bracket
    elif idx_brace != -1:
        start_at = idx_brace
    elif idx_bracket != -1:
        start_at = idx_bracket
    else:
        raise ValueError("Content provided is not JSON!")
    return _step2_sanitize_and_parse_json(content[start_at:], countdown=countdown - 1)


def load_json(file_path: str, charset='utf-8'):
    with codecs.open(file_path, 'r', charset) as fd:
        return json.load(fd)


class SimpleJSONEncoder(JSONEncoder):
    @staticmethod
    def _full_name(a_type: Type):
        return a_type.__module__ + "." + a_type.__name__

    @staticmethod
    def asdict_filtered(obj):
        return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)
                if field.init}

    def default(self, o):
        if dataclasses.is_dataclass(o):
            result = self.asdict_filtered(o)
            result["__class__"] = self._full_name(type(o))
            return result
        elif isinstance(o, enum.Enum):
            return dict(__class__=self._full_name(type(o)), value=o.name)
        elif isinstance(o, np.ndarray):
            return dict(__class__=self._full_name(np.ndarray), value=o.tolist())
        else:
            return super().default(o)


class SimpleJSONDecoder(JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self._from_dict)

    @staticmethod
    def _load_type(full_name: str) -> Optional[Type]:
        dot_idx = full_name.rfind('.')
        module_name = None if dot_idx == -1 else full_name[:dot_idx]
        class_name = full_name if dot_idx == -1 else full_name[dot_idx + 1:]
        try:
            if module_name is None:
                return getattr(builtins, class_name, None)
            else:
                module = importlib.import_module(module_name) if module_name else None
                return getattr(module, class_name, None)
        except (ModuleNotFoundError, AttributeError):
            return None

    @classmethod
    def _from_dict(cls, d: dict) -> Any:
        _class = d.get("__class__", "")
        _type = cls._load_type(_class) if _class else None
        if _type is None:
            return d
        elif dataclasses.is_dataclass(_type):
            d_copy = dict(d)
            del d_copy["__class__"]
            return _type(**d_copy)
        elif issubclass(_type, enum.Enum):
            value = d.get("value")
            return _type[value]
        elif _type == np.ndarray:
            value = d.get("value")
            return np.array(value)
        else:
            return d
