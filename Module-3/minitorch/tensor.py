"""
Implementation of the core Tensor object for autodifferentiation.
"""

from .autodiff import Variable
from .tensor_data import TensorData
from . import operators


class Tensor(Variable):
    """
    Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    Attributes:
        _tensor (:class:`TensorData`) : the tensor data storage
        backend : backend object used to implement tensor math (see `tensor_functions.py`)
    """

    def __init__(self, v, back=None, name=None, backend=None):
        assert isinstance(v, TensorData)
        assert backend is not None
        super().__init__(back, name=name)
        self._tensor = v
        self.backend = backend

    def to_numpy(self):
        """
        Returns:
             narray : converted to numpy array
        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    # Properties
    @property
    def shape(self):
        """
        Returns:
             tuple : shape of the tensor
        """
        return self._tensor.shape

    @property
    def size(self):
        """
        Returns:
             int : size of the tensor
        """
        return self._tensor.size

    @property
    def dims(self):
        """
        Returns:
             int : dimensionality of the tensor
        """
        return self._tensor.dims

    def _ensure_tensor(self, b):
        "Turns a python number into a tensor with the same backend."
        if isinstance(b, (int, float)):
            b = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
        return b

    # Functions
    def __add__(self, b):
        return self.backend.Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b):
        return self.backend.Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b):
        return self.backend.Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b):
        return self.backend.Mul.apply(
            self, self.backend.Inv.apply(self._ensure_tensor(b))
        )

    def __matmul__(self, b):
        return self.backend.MatMul.apply(self, b)

    def __lt__(self, b):
        return self.backend.LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b):
        return self.backend.EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b):
        return self.backend.LT.apply(self._ensure_tensor(b), self)

    def __neg__(self):
        return self.backend.Neg.apply(self)

    def sigmoid(self):
        return self.backend.Sigmoid.apply(self)

    def relu(self):
        return self.backend.ReLU.apply(self)

    def log(self):
        return self.backend.Log.apply(self)

    def exp(self):
        return self.backend.Exp.apply(self)

    def sum(self, dim=None):
        "Compute the sum over dimension `dim`"
        return self.backend.Sum.apply(self, dim)

    def mean(self, dim=None):
        "Compute the mean over dimension `dim`"
        return self.backend.Mean.apply(self, dim)

    def permute(self, *order):
        "Permute tensor dimensions to *order"
        return self.backend.Permute.apply(self, order)

    def view(self, *shape):
        "Change the shape of the tensor to a new shape with the same size"
        return self.backend.View.apply(self, shape)

    def contiguous(self):
        "Return a contiguous tensor with the same data"
        return self.backend.Copy.apply(self)

    def __repr__(self):
        return self._tensor.to_string()

    def __getitem__(self, key):
        return self._tensor.get(key)

    def __setitem__(self, key, val):
        self._tensor.set(key, val)

    @property
    def grad(self):
        return self.derivative

    # Internal methods used for autodiff.
    def _type_(self, backend):
        self.backend = backend
        if backend.cuda:
            self._tensor.to_cuda_()

    def _new(self, tensor_data):
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(storage, shape, strides=None, backend=None):
        "Create a new tensor from data"
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other):
        "Method used to allow for backprop over reduce."

        if self.shape == other.shape:
            return other
        shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(shape)
        self.backend._id_map(other, out=buf)
        if self.shape == shape:
            return buf
        buf2 = self.zeros(self.shape)
        # START CODE CHANGE
        buf2 = self.backend._add_reduce(buf, out=buf2)
        # END CODE CHANGE
        return buf2

    def zeros(self, shape=None):
        def zero(shape):
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self):
        return self._tensor.tuple()

    def get_data(self):
        return Tensor(self._tensor, backend=self.backend)

    def backward(self, grad_output=None):
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        super().backward(grad_output)