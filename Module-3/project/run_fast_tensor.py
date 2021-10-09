import minitorch
import datasets
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--PTS", type=int, default=50, help="number of points")
parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
parser.add_argument("--RATE", type=float, default=0.5, help="learning rate")
parser.add_argument("--BACKEND", default="cpu", help="backend mode")
parser.add_argument("--DATASET", default="xor", help="dataset")
parser.add_argument("--PLOT", default=False, help="dataset")

args = parser.parse_args()


PTS = args.PTS

if args.DATASET == "xor":
    DATASET = datasets.Xor(PTS, vis=True)
elif args.DATASET == "simple":
    DATASET = datasets.Simple(PTS, vis=True)
elif args.DATASET == "split":
    DATASET = datasets.Split(PTS, vis=True)

HIDDEN = int(args.HIDDEN)
RATE = args.RATE


# Change which backend to use

if args.BACKEND == "cpu":
    BACKEND = minitorch.make_tensor_backend(minitorch.FastOps)
elif args.BACKEND == "old":
    # Module-2 backend
    # You can use this to debug, but you will need to add a
    # Matrix multiplication @ operator
    BACKEND = minitorch.TensorFunctions
elif args.BACKEND == "gpu":
    BACKEND = minitorch.make_tensor_backend(minitorch.CudaOps)


def RParam(*shape):
    p = 1.0
    for s in shape:
        p += s
    r = 2 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, HIDDEN)
        self.layer2 = Linear(HIDDEN, HIDDEN)
        self.layer3 = Linear(HIDDEN, 1)

    def forward(self, x):
        # TODO: Implement for Task 3.5.
        x=self.layer1(x).relu()
        x=self.layer2(x).relu()
        return self.layer3(x).sigmoid()
        # raise NotImplementedError('Need to implement for Task 3.5')


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # TODO: Implement for Task 3.5.
        return x@self.weights.value+self.bias.value
        # raise NotImplementedError('Need to implement for Task 3.5')


model = Network()
data = DATASET

X = minitorch.tensor_fromlist(data.X, backend=BACKEND)
y = minitorch.tensor(data.y, backend=BACKEND)


losses = []
for epoch in range(250):
    total_loss = 0.0

    start = time.time()

    # Forward
    out = model.forward(X).view(data.N)
    prob = (out * y) + (out - 1.0) * (y - 1.0)
    loss = -prob.log()
    (loss.sum().view(1)).backward()

    total_loss = loss.sum().view(1)[0]
    losses.append(total_loss)

    # Update
    for p in model.parameters():
        if p.value.grad is not None:
            p.update(p.value - RATE * (p.value.grad / float(data.N)))

    epoch_time = time.time() - start

    # Logging
    if epoch % 10 == 0:
        correct = 0
        for i, lab in enumerate(data.y):
            if lab == 1 and out[i] > 0.5:
                correct += 1
            if lab == 0 and out[i] < 0.5:
                correct += 1

        print(
            "Epoch ",
            epoch,
            " loss ",
            total_loss,
            "correct",
            correct,
            "time",
            epoch_time,
        )
        im = f"Epoch: {epoch}"

        def plot(x):
            return model.forward(minitorch.tensor(x, (1, 2), backend=BACKEND))[0, 0]

        if args.PLOT:
            data.graph(im, plot)
        plt.plot(losses, c="blue")
        data.vis.matplot(plt, win="loss")
