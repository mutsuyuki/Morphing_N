import numpy as np
from PIL import Image
import os

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import math
import pylab

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable

import chainer.functions as F
import chainer.links as L

image_dir = './fonts'
out_image_dir = './out_images'
out_model_dir = './out_models'

nz = 100  # # of dim for Z
batchsize = 100
number_of_epoch = 10000
number_of_train = 200000
image_save_interval = 50000

# read all images

fs = os.listdir(image_dir)
print(len(fs))
dataset = []
for fn in fs:
    f = open('%s/%s' % (image_dir, fn), 'rb')
    img_bin = f.read()
    dataset.append(img_bin)
    f.close()
print(len(dataset))


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z=L.Linear(nz, 6 * 6 * 512, wscale=0.02 * math.sqrt(nz)),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            bn0l=L.BatchNormalization(6 * 6 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 3)),
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            c3=L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            l4l=L.Linear(6 * 6 * 512, 2, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x, test=False):
        h = F.elu(self.c0(x))  # no bn because images from generator will katayotteru?
        h = F.elu(self.bn1(self.c1(h), test=test))
        h = F.elu(self.bn2(self.c2(h), test=test))
        h = F.elu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l


def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))


def train_dcgan_labeled(__gen, __dis, __epoch0=0):
    optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_gen.setup(__gen)
    optimizer_dis.setup(__dis)
    optimizer_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    zvis = (xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))

    for epoch in range(__epoch0, number_of_epoch):
        print("epoch --> ", epoch, "(", __epoch0, " to ", number_of_epoch)

        total_loss_dis = np.float32(0)
        total_loss_gen = np.float32(0)

        for i in range(0, number_of_train, batchsize):
            print("i --> ", i, "train -->", number_of_train, "bachsize -->", batchsize)
            # discriminator
            # 0: from dataset
            # 1: from noise

            # print "load image start ", i
            x2 = np.zeros((batchsize, 3, 96, 96), dtype=np.float32)
            for j in range(batchsize):
                try:
                    rnd = np.random.randint(len(dataset))
                    img = np.asarray(Image.open(StringIO(dataset[rnd])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)

                    if np.random.randint(2) == 0:
                        x2[j, :, :, :] = (img[:, :, ::-1] - 128.0) / 128.0
                    else:
                        x2[j, :, :, :] = (img[:, :, :] - 128.0) / 128.0
                except:
                    print
                    'read image error occured', fs[rnd]

            # print "load image done"

            # train generator
            z = Variable(xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))
            x = __gen(z)
            yl = __dis(x)
            loss_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
            loss_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))

            # train discriminator

            x2 = Variable(cuda.to_gpu(x2))
            yl2 = __dis(x2)
            loss_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(batchsize, dtype=np.int32)))

            # print "forward done"

            optimizer_gen.zero_grads()
            loss_gen.backward()
            optimizer_gen.update()

            optimizer_dis.zero_grads()
            loss_dis.backward()
            optimizer_dis.update()

            total_loss_gen += loss_gen.data.get()
            total_loss_dis += loss_dis.data.get()

            # print "backward done"

            if i % image_save_interval == 0:
                pylab.rcParams['figure.figsize'] = (16.0, 16.0)
                pylab.clf()
                vissize = 100
                z = zvis
                z[50:, :] = (xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
                z = Variable(z)
                x = __gen(z, test=True)
                x = x.data.get()
                for i_ in range(100):
                    tmp = ((np.vectorize(clip_img)(x[i_, :, :, :]) + 1) / 2).transpose(1, 2, 0)
                    pylab.subplot(10, 10, i_ + 1)
                    pylab.imshow(tmp)
                    pylab.axis('off')
                pylab.savefig('%s/vis_%d_%d.png' % (out_image_dir, epoch, i))

        serializers.save_hdf5("%s/dcgan_model_dis_%d.h5" % (out_model_dir, epoch), __dis)
        serializers.save_hdf5("%s/dcgan_model_gen_%d.h5" % (out_model_dir, epoch), __gen)
        serializers.save_hdf5("%s/dcgan_state_dis_%d.h5" % (out_model_dir, epoch), optimizer_dis)
        serializers.save_hdf5("%s/dcgan_state_gen_%d.h5" % (out_model_dir, epoch), optimizer_gen)
        print ('epoch end', epoch, total_loss_gen / number_of_train, total_loss_dis / number_of_train)


xp = cuda.cupy
cuda.get_device(0).use()

gen = Generator()
dis = Discriminator()
gen.to_gpu()
dis.to_gpu()

try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
except:
    pass

train_dcgan_labeled(gen, dis)
