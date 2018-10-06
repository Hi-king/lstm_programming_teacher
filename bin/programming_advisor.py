# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import sys
import numpy
import chainer
import chainer.serializers
rootpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(rootpath)
import lstm_programming_teacher


parser = argparse.ArgumentParser()
parser.add_argument("modelpath")
parser.add_argument("encoderpath")
parser.add_argument("target_code")
parser.add_argument("--outfile", default="out.html")
parser.add_argument("--gpu", type=int, default=-1)
args = parser.parse_args()

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

model = lstm_programming_teacher.models.CodeModel(vocab_size=100, midsize=10, output_feature_size=2)
chainer.serializers.load_npz(args.modelpath, model)

chara_encoder = lstm_programming_teacher.utils.CharaEncoder()
chara_encoder.load(args.encoderpath)

model.reset_state()
values = []
for chara in open(args.target_code).read().lower():
    predicted = model(chainer.Variable(xp.array([chara_encoder(chara)], xp.int32)))
    print(chara)
    print(predicted.data)
    values.append(chainer.functions.softmax(predicted).data[0, 1])
    # values.append(predicted.data[0, 0] / (predicted.data[0, 0] + predicted.data[0, 1]))
    # lastvalue = chainer.functions.softmax(predicted).data[0, 1]

# maxvalue = max(values)
# values = [value/maxvalue for value in values]
with open(args.outfile, "w+") as f:
    f.write("<html>")
    #f.write("<h1>あなたのコードは{}点</h1>".format(int(100*(values[-1]))))
    f.write("<h1>あなたのコードはおそらく{}</h1>".format("正解" if values[-1] > 0.5 else "不正解"))
    for chara, value in zip(open(args.target_code).read(), values):
        if chara == "\n":
            f.write("<br />")
        else:
            f.write('<span style="color:rgb(0, 0, {})">{}</span>'.format(int((value) * 255), chara))
    f.write("</html>")


