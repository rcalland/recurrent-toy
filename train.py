import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, serializers

import numpy as np

from make_data import generate_trajectory

from matplotlib import pyplot as plt

class RNN(chainer.Chain):
    def __init__(self):
        n_units = 64
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
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        return loss

dev_id = 0

if dev_id >= 0:
    chainer.cuda.get_device(dev_id).use()

rnn = RNN()
model = Loss(rnn)
optimizer = optimizers.Adam()
optimizer.setup(model)

continue_training = True
if continue_training:
    serializers.load_hdf5("curve.model", rnn)
    serializers.load_hdf5("curve.state", optimizer)

if dev_id >= 0:
    model.to_gpu()

n = 500
batchsize = 32
trk_len = 16

for i in range(n):

    rnn.reset_state()

    # get a batch of training curves
    batch = generate_trajectory(n=trk_len)

    for b in range(batchsize-1):
        batch = np.concatenate((batch, generate_trajectory(n=trk_len)))

    train = batch[:,:-1]
    true = batch[:,1:]

    loss = 0.0

    for ii in range(trk_len-1):
        x = train[:,ii,:]
        t = true[:,ii,:]

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

# save our model
serializers.save_hdf5("curve.model", rnn)
serializers.save_hdf5("curve.state", optimizer)

# test the model
n_test = 1

fig = plt.figure()
ax = fig.add_subplot(111)

occluded = [8,9,10]

for test in range(n_test):
    track = generate_trajectory(n=trk_len)
    rnn.reset_state()

    x = []
    y = []

    for c, i in enumerate(track[0,]):
        i = i.reshape(-1, len(i))

        #if c > trk_len / 2:
        #    print("predicting {}".format(c))
        #    i = np.array([[x[-1], y[-1]]])
        
        # remove some measurements
        if c in occluded:
            print("{} is occluded".format(c))
            i = np.array([[x[-1], y[-1]]])


        if dev_id >= 0:
            i = chainer.cuda.to_gpu(i)
        
        yy = rnn(chainer.Variable(i)).data

        if dev_id >= 0:
            yy = chainer.cuda.to_cpu(yy)
            i = chainer.cuda.to_cpu(i)

        x.append(yy[0,0])
        y.append(yy[0,1])

    # remove occluded elements from plot
    track = np.delete(track, occluded, 1)

    ax.scatter(track[0,:,0], track[0,:,1], c="black", marker="o", s=50)
    ax.plot(x, y, lw=3)

plt.show()
