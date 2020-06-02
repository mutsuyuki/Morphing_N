import numpy as np
import math
import pylab

import chainer
from chainer import serializers
from chainer import Variable


import chainer.functions as F
import chainer.links as L

import numpy

nz = 100

epochIndex = 24

model_file = 'fonts_out_models/170831/dcgan_model_gen_' + str(epochIndex) + '.h5'
out_file = 'output' + str(epochIndex) + '.png'

print(model_file)


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z=L.Linear(nz, 6 * 6 * 512),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1),
            bn0l=L.BatchNormalization(6 * 6 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z))), (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = (self.dc4(h))
        return x


def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))


xp = numpy

gen = Generator()

serializers.load_hdf5(model_file, gen)

pylab.rcParams['figure.figsize'] = (22.0, 22.0)
pylab.clf()
batchsize = 100
z = (xp.random.uniform(-1, 1, (batchsize, nz)).astype(np.float32))
z = Variable(z)
x = gen(z, test=True)
x = x.data
for i_ in range(batchsize):
    tmp = ((np.vectorize(clip_img)(x[i_, :, :, :]) + 1) / 2).transpose(1, 2, 0)
    pylab.subplot(10, 10, i_ + 1)
    pylab.imshow(tmp)
    pylab.axis('off')
pylab.savefig(out_file)
