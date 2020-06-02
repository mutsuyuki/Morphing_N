# import numpy as np
import cupy as np
import math
from flask import Flask
from flask_cors import CORS
from predictor.ModelLoader import ModelLoader
import time
import cupy

app = Flask(__name__)
CORS(app)


model_loader = ModelLoader('./model/gen.h5')


@app.route('/', methods=['GET'])
def random():
    noise = (np.random.uniform(-1, 1, (3, 100)).astype(np.float32))
    image = model_loader.generate(noise)


    return make_response(image)


@app.route('/api/<key>', methods=['GET'])
def get(key):
    values = key.split(",")

    noise = (np.random.uniform(0, 0, (3, 100)).astype(np.float32))

    for i in range(0, len(values)):
        noise[0, (i + 1) * 3:(i + 2) * 3] = float( values[i] )
        noise[1, (i + 1) * 6:(i + 2) * 6] = float( values[i] )
        noise[2, (i + 1) * 9:(i + 2) * 9] = float( values[i] )

    image = model_loader.generate(noise)

    return make_response(image)


@app.route('/sin/<key>', methods=['GET'])
def loop(key):

    noise = (np.random.uniform(0, 0, (3, 100)).astype(np.float32))

    for i in range(0, 100):
        value = math.sin(i / 100 * 3.14 * 2 + float(key) / 100 * 3.14 * 2)
        noise[0, i] = value

        value = math.sin((i + 30) / 100 * 3.14 * 3 + float(key) / 100 * 3.14 * 3)
        noise[1, i] = value

        value = math.sin((i + 60) / 100 * 3.14 * 4.2 + float(key) / 100 * 3.14 * 4.2)
        noise[2, i] = value

    image = model_loader.generate(noise)

    return make_response(image)


def make_response(__image):
    unix = time.time()
    
    if(hasattr(np, "asnumpy")):
        __image = np.asnumpy(__image)

    flatten = __image.astype(np.int).astype("str")

    print(6, (time.time() - unix) * 10000)
    unix = time.time()

    result = ",".join(flatten)

    print(7, (time.time() - unix) * 10000)
    unix = time.time()

    return result[:-1]


if __name__ == '__main__':
    app.run(debug=True)
