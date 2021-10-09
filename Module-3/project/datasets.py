import numpy
import random
import visdom
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def to_fig(canvas):
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(s, np.uint8).reshape((height, width, 4))


def make_pts(N):
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


class Graph:
    def __init__(self, vis=False, vis_args={}):
        self.gifs = []
        if vis:
            self.vis = visdom.Visdom(**vis_args)
        else:
            self.vis = None
        self.first = True

    def plot(self, outfile, model=None, fig=None):
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        cmap = plt.cm.get_cmap("RdBu")
        if model is not None:
            X = []
            Y = []
            Z = []
            for i in range(51):
                inner = []
                innerx = []
                innery = []
                for j in range(51):
                    x_1 = i / 50.0
                    x_2 = j / 50.0
                    innerx.append(x_1)
                    innery.append(x_2)
                    val = model([x_1, x_2])
                    inner.append(val)
                    Z.append(inner)
                    X.append(innerx)
                    Y.append(innery)

            Z = numpy.array(Z)
            ax.contourf(X, Y, Z, cmap=cmap)

        ax.scatter(
            [p[0] for p in self.X],
            [p[1] for p in self.X],
            c=self.y,
            edgecolors="black",
            cmap=cmap,
        )
        ax.set_title(outfile)
        return fig

    def graph(self, outfile, model=None):
        if self.vis is None:
            return
        fig = self.plot(outfile, model, Figure())
        canvas = FigureCanvas(fig)
        im = to_fig(canvas)
        if self.first:
            self.vis.close(win="Progress")
        self.vis.image(
            im.transpose(2, 0, 1), win="Progress", opts=dict(store_history=True)
        )
        self.first = False


class Simple(Graph):
    def __init__(self, N, vis=False, vis_args={}):
        super().__init__(vis, vis_args)
        self.N = N
        self.X = make_pts(N)
        self.y = []
        for x_1, x_2 in self.X:
            y = 1 if x_1 < 0.5 else 0
            self.y.append(y)


class Split(Graph):
    def __init__(self, N, vis=False, vis_args={}):
        super().__init__(vis, vis_args)
        self.N = N
        self.X = make_pts(N)
        self.y = []
        for x_1, x_2 in self.X:
            y = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
            self.y.append(y)


class Xor(Graph):
    def __init__(self, N, vis=False, vis_args={}):
        super().__init__(vis, vis_args)
        self.N = N
        self.X = make_pts(N)
        self.y = []
        for x_1, x_2 in self.X:
            y = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
            self.y.append(y)
