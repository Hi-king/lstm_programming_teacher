# -*- coding: utf-8 -*-
from __future__ import print_function
import pickle

class CharaEncoder(object):
    def __init__(self):
        self.charaset = set()
        self.index_dict = {}

    def __call__(self, chara):
        if not chara in self.charaset:
            self.charaset.add(chara)
            self.index_dict[chara] = len(self.charaset)-1
        return self.index_dict[chara]

    def save(self, filename):
        with open(filename, "w+") as f:
            pickle.dump(self.index_dict, f)

    def load(self, filename):
        print(filename)
        with open(filename, "rb") as f:
            self.index_dict = pickle.load(f)
        self.charaset = set(self.index_dict.keys())
