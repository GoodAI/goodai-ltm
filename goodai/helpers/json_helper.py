import codecs
import json


def load_json(file_path: str, charset='utf-8'):
    with codecs.open(file_path, 'r', charset) as fd:
        return json.load(fd)
