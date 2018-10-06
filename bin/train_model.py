# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import argparse
import random
import numpy
import chainer
import chainer.optimizers
import chainer.serializers
import pipe
import itertools
import subprocess
import tempfile
import time
import logging
import pickle

rootpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(rootpath)
import lstm_programming_teacher

savepath = os.path.join(rootpath, "output", str(time.time()))
if not os.path.exists(savepath):
    os.makedirs(savepath)
logging.basicConfig(filename=os.path.join(savepath, "log.txt"),level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("contest", help="e.g. abc041")
parser.add_argument("stage", help="e.g. a")
parser.add_argument("language", help="e.g. python2_2.7.6")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--testnum", type=int, default=10)
parser.add_argument("--batch", type=int, default=10)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--max_length", type=int, default=300)
parser.add_argument("--augmentation", type=int, default=1, help="how much data augmentation")
args = parser.parse_args()
logging.info(args)

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

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

    # def predicted_concate(self, program):

def data_loader(file_pathes):
    for file_path in file_pathes:
        logging.info(file_path)
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
def max_length_filter(iterable):
    for code in iterable:
        if (not args.language == "gpp_5.3.0") or check_compilable(code):
            yield code

@pipe.Pipe
def compilable_filter(iterable):
    for code in iterable:
        if len(code) < args.max_length:
            yield code

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
                # print("augmented: {}".format(augmented))
            if augmented >= num_augment:
                break


def train(postivies, negatives, predictor, optimizer, batch=10, epoch_size=10000000, num_augment=1):
    xp = predictor.xp
    # num_sample = min(len(postivies), len(negatives), epoch_size)
    data_loader_positive = data_loader(postivies) | max_length_filter | compilable_filter | pipe.take(epoch_size)
    data_loader_negative = data_loader(negatives) | max_length_filter | compilable_filter | pipe.take(epoch_size)
    if num_augment > 1:
        data_loader_positive = data_loader_positive | augmentation(num_augment=num_augment)
        data_loader_negative = data_loader_negative | augmentation(num_augment=num_augment)
    while True:
        loss = chainer.Variable(xp.zeros((), dtype=xp.float32))
        try:
            for _i in range(batch):
                postive_predicted = predictor(data_loader_positive.next())
                loss += chainer.functions.softmax_cross_entropy(postive_predicted, chainer.Variable(xp.array([1], xp.int32)))
                negative_predicted = predictor(data_loader_negative.next())
                loss += chainer.functions.softmax_cross_entropy(negative_predicted, chainer.Variable(xp.array([0], xp.int32)))
        except StopIteration:
            optimizer.zero_grads()
            loss.backward()
            optimizer.update()
            break
        optimizer.zero_grads()
        loss.backward()
        optimizer.update()


def test(postivies, negatives, predictor, head=1000, prefix=""):
    data_loader_positive = data_loader(postivies) | max_length_filter
    data_loader_negative = data_loader(negatives) | max_length_filter
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
    logging.info("{}correct: {}, total: {}".format(prefix, num_correct, num_total))
    logging.info("{}accuracy: {}".format(prefix, float(num_correct)/num_total))

model = lstm_programming_teacher.models.CodeModel(vocab_size=100, midsize=10, output_feature_size=2)
if args.gpu >= 0:
    model.to_gpu()
predictor = WholeProgramPredictor(model, chara_encoder=lstm_programming_teacher.utils.CharaEncoder(), xp=xp)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
for epoch in range(args.epoch):
    logging.info("epoch: {}".format(epoch))
    train(positive_train, negative_train, predictor, optimizer, batch=args.batch, epoch_size=10, num_augment=args.augmentation)
    test(positive_train, negative_train, predictor, head=10, prefix="train ")
    test(positive_test, negative_test, predictor)
    chainer.serializers.save_npz(os.path.join(savepath, "{}_model.npz".format(epoch)), model)
    chainer.serializers.save_npz(os.path.join(savepath, "{}_optimizer.npz".format(epoch)), optimizer)
    predictor.chara_encoder.save(os.path.join(savepath, "{}_chara_encoder.dump".format(epoch)))
