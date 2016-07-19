# -*- coding: utf-8 -*-
from __future__ import print_function
import chainer

class CodeModel(chainer.Chain):
    def __init__(self, vocab_size, midsize, output_feature_size):
        super(CodeModel, self).__init__(
            word_embed=chainer.functions.EmbedID(vocab_size, midsize),
            lstm0=chainer.links.connection.lstm.LSTM(midsize, midsize),
            lstm1=chainer.links.connection.lstm.LSTM(midsize, midsize),
            out_layer=chainer.functions.Linear(midsize, output_feature_size)
        )

    def __call__(self, x):
        h = self.word_embed(x)
        if hasattr(self, "lstm0"):
            h = self.lstm0(h)
        if hasattr(self, "lstm1"):
            h = self.lstm1(h)
        if hasattr(self, "lstm2"):
            h = self.lstm2(h)
        feature = self.out_layer(h)
        return feature

    def reset_state(self):
        if hasattr(self, "lstm0"):
            self.lstm0.reset_state()
        if hasattr(self, "lstm1"):
            self.lstm1.reset_state()
        if hasattr(self, "lstm2"):
            self.lstm2.reset_state()