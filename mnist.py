import numpy as np
import struct
import gzip

def load_idx_ubyte(f):
    magic = f.read(4)
    if magic[0:3] != b'\x00\x00\x08':
        raise ValueError('not an idx ubyte file')
    dims = magic[3]
    if dims > 4:
        raise ValueError('too many dims')
    shape = []
    for i in range(dims):
        shape.append(struct.unpack('>I', f.read(4))[0])
    array = np.frombuffer(f.read(), dtype=np.uint8)
    array.shape = tuple(shape)
    return array

def load_mnist_gz(
        images_fname='data/train-images-idx3-ubyte.gz',
        labels_fname='data/train-labels-idx1-ubyte.gz'):
    with open(images_fname, 'rb') as f:
        images = load_idx_ubyte(gzip.open(f))
    with open(labels_fname, 'rb') as f:
        labels = load_idx_ubyte(gzip.open(f))
    return images, labels
