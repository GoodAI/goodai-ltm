import codecs
import os
import unittest
from pathlib import Path
from pyltm.data.query_passage.auto_data_source import AutoQueryPassageDataSource
from transformers import PreTrainedTokenizer, AutoTokenizer
import numpy as np
from pyltm.helpers.html_helper import text_to_html


class TestQueryPassageDataSource(unittest.TestCase):
    def test_show_examples(self):
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        random = np.random.RandomState(1)
        ds_names = ['adversarial_qa', 'coqa', 'squad_v2']
        for ds_name in ds_names:
            print(f'Processing {ds_name}...')
            self._show_examples(random, ds_name, tokenizer)

    @staticmethod
    def _show_examples(random: np.random.RandomState, ds_name: str, tokenizer: PreTrainedTokenizer):
        train_ds, test_ds = AutoQueryPassageDataSource.create(ds_name, random, tokenizer,
                                                              max_query_tokens=40, min_passage_tokens=24,
                                                              max_passage_tokens=36)

        # Sample 100 examples from the training data source
        examples = train_ds.sample_items(100)

        # Sort examples by match value
        examples.sort(key=lambda x: x.match, reverse=True)

        # Generate HTML table
        table_rows = ['<tr><th>Match</th><th>Query</th><th>Passage</th></tr>']
        for example in examples:
            match_str = 'Yes' if example.match else 'No'
            query_str = tokenizer.decode(example.queryIds, skip_special_tokens=True)
            passage_str = tokenizer.decode(example.passageIds, skip_special_tokens=True)
            query_html = text_to_html(query_str)
            passage_html = text_to_html(passage_str)
            table_rows.append(f'<tr><td>{match_str}</td><td>{query_html}</td>' +
                              f'<td>{passage_html}</td></tr>')
        html_table = '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n' +\
                     '<html><body>' +\
                     '<table border=1>' + ''.join(table_rows) + '</table>' +\
                     '</body></html>'

        # Save HTML file to local data directory
        data_dir = Path('data')
        os.makedirs(data_dir, exist_ok=True)
        html_path = str(data_dir / f'qp_datasource_{ds_name}_examples.html')
        with codecs.open(html_path, 'w', 'utf-8') as f:
            f.write(html_table)
