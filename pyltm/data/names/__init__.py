import json
import os
from typing import List

import numpy as np

from pyltm.helpers.json_helper import load_json

_instance_attr = '__instance'
_names_dir = os.path.join(os.path.dirname(__file__))
_names_path = os.path.join(_names_dir, 'wikidata-names.json')


class NameSource:
    def __init__(self):
        def _filter(names: List[str]):
            return [n for n in names if len(n) <= 10]

        names_obj = load_json(_names_path)
        self.male_names = np.array(_filter(names_obj['MALE']))
        self.female_names = np.array(_filter(names_obj['FEMALE']))
        self.family_names = np.array(_filter(names_obj['FAMILY']))

    @classmethod
    def get_instance(cls):
        try:
            return getattr(cls, _instance_attr)
        except AttributeError:
            instance = cls()
            setattr(cls, _instance_attr, instance)
            return instance

    def get_male_names(self) -> List[str]:
        return list(self.male_names)

    def get_female_names(self) -> List[str]:
        return list(self.female_names)

    def get_family_names(self) -> List[str]:
        return list(self.family_names)

    def sample_male_names(self, random: np.random.RandomState, count: int):
        return random.choice(self.male_names, size=count, replace=False)

    def sample_female_names(self, random: np.random.RandomState, count: int):
        return random.choice(self.female_names, size=count, replace=False)

    def sample_family_names(self, random: np.random.RandomState, count: int):
        return random.choice(self.family_names, size=count, replace=False)

    def sample_first_names(self, random: np.random.RandomState, count: int):
        male_names = self.sample_male_names(random, count)
        female_names = self.sample_female_names(random, count)
        is_male = random.choice([True, False], size=count, replace=True)
        return [m if im else f for m, f, im in zip(male_names, female_names, is_male)]
