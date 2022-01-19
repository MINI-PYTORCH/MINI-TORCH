from mnist import MNIST
from numba.core.types.iterators import RangeIteratorType
import minitorch
import visdom
import numpy  as np
from tqdm import tqdm
from minitorch.nn import bce_loss, cross_entropy, mse_loss
from minitorch.optim import SGD,Adam
import math
import random
	
import warnings
warnings.filterwarnings("ignore")

# vis = visdom.Visdom()
mndata = MNIST("data/")
images, labels = mndata.load_training()


# BACKEND = minitorch.make_tensor_backend(minitorch.FastOps)
BACKEND=minitorch.make_tensor_backend(minitorch.CudaOps)

BATCH = 50
N = 500
EVAL_NUM=200
epochs=50

# N = 100
# EVAL_NUM=50
# epochs=30


# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


np.random.seed(101)
random.seed(101)





'''
不同的参数初始化方法
'''
def RParam(*shape):

    # r = 1.0 * (minitorch.rand(shape, backend=BACKEND) - 0.5) #[-0.05,0.05的均匀分布]
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)

def HeParam(*shape):
    r = math.sqrt(6/shape[0]) *2* (minitorch.rand(shape,backend=BACKEND)-0.5) #He初始化
    return minitorch.Parameter(r)

def ZeroParam(*shape):
    r=  minitorch.zeros(shape,backend=BACKEND)
    return minitorch.Parameter(r) #零初始化





class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # self.weights = HeParam(in_size, out_size)
        self.weights = RParam(in_size, out_size)
        # self.bias = ZeroParam(out_size)
        self.bias= RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        return x@self.weights.value+self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)
        # self.bias = ZeroParam(out_channels, 1, 1)

    def forward(self, input):
        return minitorch.fast_conv.conv2d(input,self.weights.value)+self.bias.value



class MLP(minitorch.Module):
    def __init__(self):
        super().__init__()
        self.l1=Linear(28*28,512)
        self.l2=Linear(512,256)
        self.l3=Linear(256,C)


    def forward(self, x):
        x=x.view(x.size//(28*28),28*28)
        x=self.l1(x).relu()
        x=self.l2(x).relu()
        x=self.l3(x)
        return x


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()
        self.mid = Conv2d(1,4,3,3)#36
        self.out = Conv2d(4,8,3,3)#216
        self.l1=Linear(392,128)
        self.l2=Linear(128,C)


    def forward(self, x):
        x=self.mid(x).relu()
        x=self.out(x).relu()
        x=minitorch.nn.maxpool2d(x,[4,4])
        x=x.view(x.size//392,392)
        x=self.l1(x).relu()
        if self.training:
            x=minitorch.nn.dropout(x,0.25)*4/3
        x=self.l2(x)
        return x

def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


X, ys = make_mnist(0, N)
val_x, val_ys = make_mnist(10000, 10000+EVAL_NUM)


# model = Network()
model = MLP()

losses = []
# optimizer=SGD(model.parameters(),lr=0.1)
# optimizer=SGD(model.parameters(),lr=0.01,weight_decay_l2=0.0005)
optimizer=Adam(model.parameters(),lr=0.0001,weight_decay_l2=0.0005)

for epoch in tqdm(range(epochs)):
    total_loss = 0.0
    cur = 0
    cur_y = 0

    model.train()
    
    if epoch==50:
        optimizer.lr*=0.5
        optimizer.weight_decay_l2*=0.5
        
    
    for batch_num, example_num in tqdm(enumerate(range(0, N, BATCH))):
        if N - example_num < BATCH:
            continue
        y = minitorch.tensor_fromlist(
            ys[example_num : example_num + BATCH], backend=BACKEND
        )
        x = minitorch.tensor_fromlist(
            X[example_num : example_num + BATCH], backend=BACKEND
        )
        x=(x-32.0)/77.0
        # Forward
        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)


        loss=bce_loss(out,y)
        # loss=cross_entropy(out,y)
        # loss=mse_loss(out,y)

        optimizer.zero_grad()
        loss.backward()
        total_loss += loss
        losses.append(total_loss)

        # Update
        optimizer.step()

    if (epoch+1)%2 == 0:
        model.eval()
            # Evaluate on 5 held-out batches

        train_correct = 0
        x=minitorch.tensor_fromlist(X,backend=BACKEND).view(N,1,H,W)
        y=minitorch.tensor_fromlist(ys,backend=BACKEND)
        x=(x-32.0)/77.0
        out=model.forward(x).view(N,C)
        for i in range(N):
            m = - 1000000000
            ind = 0
            for j in range(C):
                if out[i, j] > m:
                    ind = j
                    m = out[i, j]
            if y[i, ind] == 1.0:
                train_correct += 1            


        eval_correct = 0
        y = minitorch.tensor_fromlist(
        val_ys[0:EVAL_NUM], backend=BACKEND
        )
        x = minitorch.tensor_fromlist(
        val_x[0:EVAL_NUM], backend=BACKEND
        )
        x=(x-32.0)/77.0
        out = model.forward(x.view(EVAL_NUM, 1, H, W)).view(EVAL_NUM, C)
        for i in range(EVAL_NUM):
            m = -1000
            ind = 0
            for j in range(C):
                if out[i, j] > m:
                    ind = j
                    m = out[i, j]
            if y[i, ind] == 1.0:
                eval_correct += 1

        print(
                "Epoch ",
                epoch,
                " example ",
                example_num,
                " loss ",
                total_loss[0],
                "eval_accuracy ",
                eval_correct / float(EVAL_NUM),
                "train_accuracy ",
                train_correct/float(N)

        )
        total_loss = 0.0
        model.train()


'''
BASELINE: RESULT


'''