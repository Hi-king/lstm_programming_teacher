import pylab
import sys
import argparse
import numpy

parser = argparse.ArgumentParser()
args = parser.parse_args()

values0 = []
values1 = []
for line in sys.stdin:
    if line.find("INFO:root:accuracy") >= 0:
        value = float(line.split()[1])
        values0.append(value)
    if line.find("INFO:root:train accuracy") >= 0:
        value = float(line.split()[2])
        values1.append(value)

pylab.plot(values0, label="test")
pylab.plot(values1, label="train")
pylab.ylim(0, 1.1)
pylab.ylabel("accuracy")
pylab.legend()
pylab.show()
