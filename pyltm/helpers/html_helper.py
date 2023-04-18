import html
import re


def text_to_html(text: str) -> str:
    h_text = html.escape(text)
    h_text = re.sub(r'\r?\n\r?\n', '<p>', h_text)
    h_text = re.sub(r'\r?\n', '<br>', h_text)
    return h_text
