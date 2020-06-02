import math
import chainer
import chainer.functions as F
import chainer.links as L

nz = 100

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

