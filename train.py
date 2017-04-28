import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

import numpy as np

from make_data import generate_trajectory

from matplotlib import pyplot as plt

class RNN(chainer.Chain):
    def __init__(self):
        n_units = 5
        super().__init__(
            l1 = L.Linear(2, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, 2)
        )
    
    def reset_state(self):
        self.l2.reset_state()

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        y = self.l3(h2)
        return y

class Loss(chainer.Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)

    def __call__(self, x, t):
        x.data = x.data.reshape((-1, len(x.data))).astype(np.float32)
        t.data = t.data.reshape((-1, len(t.data))).astype(np.float32)

        y = self.predictor(x)
        #for i,j in zip(y,t):
        #    print(i.data,j.data)
        loss = F.mean_squared_error(y, t)
        #print(loss.data)
        #report({'loss':loss}, self)
        return loss

dev_id = -1

if dev_id >= 0:
    chainer.cuda.get_device(dev_id).use()

rnn = RNN()
model = Loss(rnn)
optimizer = optimizers.Adam()
optimizer.setup(model)

if dev_id >= 0:
    model.to_gpu()

n = 25000
trk_len = 10

for i in range(n):

    rnn.reset_state()

    track = generate_trajectory(n=trk_len)
    
    train = track[:-1]
    true = track[1:]
    #print(train.shape, true.shape)

    loss = 0.0

    for x, t in zip(train, true):
        if dev_id >= 0:
            x = chainer.cuda.to_gpu(x)
            t = chainer.cuda.to_gpu(t)

        loss += model(chainer.Variable(x), chainer.Variable(t))

    optimizer.target.zerograds()
    loss.backward()
    loss.unchain_backward()
    optimizer.update()
    

    if int(i) % 100 == 0:
        print(i,n,loss.data)


# test the model
n_test = 3

plt.figure()

for test in range(n_test):
    track = generate_trajectory(n=trk_len)

    x = []
    y = []

    for i in track[:-1]:
        i = i.reshape(-1, len(i))
        if dev_id >= 0:
            i = chainer.cuda.to_gpu(i)
        yy = rnn(chainer.Variable(i))

        if dev_id >= 0:
            yy = chainer.cuda.to_cpu(yy)

        x.append(yy[0,0].data)
        y.append(yy[0,1].data)

    plt.plot(track[:,0], track[:,1], ls="dashed")
    plt.plot(x, y)

    rnn.reset_state()

plt.show()
