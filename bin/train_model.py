# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import argparse
import random
import numpy
import chainer
import chainer.optimizers
import pipe
import itertools
import subprocess
import tempfile

rootpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(rootpath)
import lstm_programming_teacher

parser = argparse.ArgumentParser()
parser.add_argument("contest", help="e.g. abc041")
parser.add_argument("stage", help="e.g. a")
parser.add_argument("language", help="e.g. python2_2.7.6")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--testnum", type=int, default=10)
args = parser.parse_args()


def data_list(dirpath, testnum=10):
    all_pathes = [os.path.join(dirpath, base) for base in os.listdir(dirpath)]
    random.shuffle(all_pathes)
    return all_pathes[:testnum], all_pathes[testnum:]
positive_test, positive_train = data_list(os.path.join(rootpath, "data", args.contest, args.stage, args.language, "AC"), testnum=args.testnum)
negative_test, negative_train = data_list(os.path.join(rootpath, "data", args.contest, args.stage, args.language, "WA"), testnum=args.testnum)

class WholeProgramPredictor(object):
    def __init__(self, model, chara_encoder, xp=numpy):
        self.model = model
        self.chara_encoder = chara_encoder
        self.xp = xp

    def __call__(self, program):
        self.model.reset_state()
        for chara in program:
            predicted = self.model(chainer.Variable(self.xp.array([self.chara_encoder(chara)], self.xp.int32)))
        return predicted

class CharaEncoder(object):
    def __init__(self):
        self.charaset = set()
        self.index_dict = {}

    def __call__(self, chara):
        if not chara in self.charaset:
            self.charaset.add(chara)
            self.index_dict[chara] = len(self.charaset)-1
        return self.index_dict[chara]


def data_loader(file_pathes):
    for file_path in file_pathes:
        print(file_path)
        if (not args.language == "gpp_5.3.0") or check_compilable(replaced):
            yield open(file_path).read().lower()

def check_compilable(code):
    f = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".cpp")
    f.write(code)
    f.flush()
    f.close()
    process = subprocess.Popen([
        "g++",
        "-std=c++11",
        f.name],
        stderr=open(os.devnull, 'wb'))
    return_code = process.wait()
    return return_code == 0


@pipe.Pipe
def augmentation(iterable, num_augment=10):
    charas = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    for code in iterable:
        replacements = list(itertools.product(charas, repeat=2))
        random.shuffle(replacements)
        augmented = 0
        for chara_from, chara_to in replacements:
            replaced = code.replace(chara_from, chara_to)
            if (not args.language == "gpp_5.3.0") or check_compilable(replaced):
                yield replaced
                augmented += 1
            if augmented >= num_augment:
                break


def train(postivies, negatives, predictor, optimizer, batch=10, epoch_size=10000000):
    xp = predictor.xp
    num_sample = min(len(postivies), len(negatives), epoch_size)
    data_loader_positive = data_loader(postivies)
    data_loader_negative = data_loader(negatives)
    for batch_start in range(0, num_sample, batch):
        loss = chainer.Variable(xp.zeros((), dtype=xp.float32))
        for _i in range(batch):
            postive_predicted = predictor(data_loader_positive.next())
            loss += chainer.functions.softmax_cross_entropy(postive_predicted, chainer.Variable(xp.array([1], xp.int32)))
            negative_predicted = predictor(data_loader_negative.next())
            loss += chainer.functions.softmax_cross_entropy(negative_predicted, chainer.Variable(xp.array([0], xp.int32)))
        optimizer.zero_grads()
        loss.backward()
        optimizer.update()

def test(postivies, negatives, predictor, head=1000):
    data_loader_positive = data_loader(postivies)
    data_loader_negative = data_loader(negatives)
    num_correct = 0
    num_total = 0
    for sample in data_loader_positive | pipe.take(head):
        predicted = chainer.functions.softmax(predictor(sample))
        if numpy.argmax(predicted.data[0]) == 1:
            num_correct += 1
        num_total += 1
    for sample in data_loader_negative | pipe.take(head):
        predicted = chainer.functions.softmax(predictor(sample))
        if numpy.argmax(predicted.data[0]) == 0:
            num_correct += 1
        num_total += 1
    print("correct: {}, total: {}".format(num_correct, num_total))


model = lstm_programming_teacher.models.CodeModel(vocab_size=100, midsize=10, output_feature_size=2)
predictor = WholeProgramPredictor(model, chara_encoder=CharaEncoder(), xp=numpy)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
for epoch in range(args.epoch):
    train(positive_train, negative_train, predictor, optimizer, batch=1, epoch_size=10)
    test(positive_train, negative_train, predictor, head=10)
    test(positive_test, negative_test, predictor)

