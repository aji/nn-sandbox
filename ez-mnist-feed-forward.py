import numpy as np
import mnist
import random
import nn

def random_subsets(xs, n):
    indices = list(range(n))
    ys = []
    while True:
        random.shuffle(indices)
        for idx in indices:
            ys.append(xs[idx])
            if len(ys) >= n:
                yield ys
                ys = []

if __name__ == '__main__':
    images, labels = mnist.load_mnist_gz()
    images.shape = (images.shape[0], 28 * 28)
    images = images / 255.

    def to_onehot(i):
        return np.where(i == np.arange(10), 1., -1.)
    def fr_onehot(x):
        return np.argmax(x)

    net = nn.Layered(nn.SumOfSquares(), 0.001)
    net.add(nn.Linear(28 * 28, 200))
    net.add(nn.LeakyReLU(200))
    net.add(nn.Linear(200, 200))
    net.add(nn.LeakyReLU(200))
    net.add(nn.Linear(200, 10))
    net.add(nn.Tanh(10))

    all_examples = [(x, to_onehot(y)) for x, y in zip(images, labels)]

    for examples in random_subsets(all_examples, 100):
        cost = net.train(examples)
        print('')
        print('cost: ' + str(cost))

        print('sample:')
        sample, label_vec = random.choice(all_examples)
        guess_vec = net.sample(sample)
        label = fr_onehot(label_vec)
        guess = fr_onehot(guess_vec)
        color = 32 if guess == label else 31
        for i in range(10):
            print(' {} {}{:7.3f} {}\x1b[0m'.format(
                '\x1b[1;{}m--->'.format(color) if i == label else '    ',
                '\x1b[1;{}m'.format(color) if i == guess else '',
                guess_vec[i],
                '<---' if i == guess else '    '))
