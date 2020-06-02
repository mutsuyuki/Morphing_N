# import numpy as np
import cupy as np
from .Generator import Generator
from chainer import serializers
from chainer import Variable
from chainer import cuda
import datetime
import time



class ModelLoader:
    def __init__(self, __path):
        print("model loader initialized")
        self.generator = Generator()
        self.generator.to_gpu()
        serializers.load_hdf5(__path, self.generator)

    def generate(self, __noise):
        unix = time.time()

        z = Variable(__noise)

        print(1, (time.time() - unix) * 10000)
        unix = time.time()

        x = self.generator(z)

        print(2, (time.time() - unix) * 10000)
        unix = time.time()

        images = np.array([])
        for i in range(0, x.shape[0]):
            x_clipped = (x.data[i, :, :, :] + 1) / 2

            image = (x_clipped[0] + x_clipped[1] + x_clipped[2]) / 3 * 255
            image = np.clip(image, 0, 255)
            image = image.flatten()
            images = np.concatenate((images, image),axis=0)

        return images

    def clip_img(self, x):
        return np.float32(-1 if x < -1 else (1 if x > 1 else x))
