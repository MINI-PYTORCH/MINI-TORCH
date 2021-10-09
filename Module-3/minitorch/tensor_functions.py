"""
Implementation of the autodifferentiation Functions for Tensor.
"""


from minitorch import tensor_data
from .autodiff import FunctionBase
from .tensor_ops import TensorOps
import numpy as np
from . import operators
from .tensor import Tensor
import random



# Constructors
class Function(FunctionBase):
    data_type = Tensor

    @staticmethod
    def variable(data, back):
        return Tensor(data[0], back, backend=data[1])

    @staticmethod
    def data(a):
        return (a._tensor, a.backend)


def make_tensor_backend(tensor_ops, is_cuda=False):
    """
    Dynamically construct a tensor backend based on a `tensor_ops` object
    that implements map, zip, and reduce higher-order functions.

    Args:
        tensor_ops (:class:`TensorOps`) : tensor operations object see `tensor_ops.py`
        is_cuda (bool) : is the operations object CUDA / GPU based

    Returns :
        backend : a collection of tensor functions


    tensor_pos的输入和tensor_function的输入，都是tensor,不是tensor_data.
    我们的functional的实现只用ops(不会有计算图)，而不其他的functional或者重载符，因为可能会错误。
    """
    # Maps
    neg_map = tensor_ops.map(operators.neg)
    sigmoid_map = tensor_ops.map(operators.sigmoid)
    relu_map = tensor_ops.map(operators.relu)
    log_map = tensor_ops.map(operators.log)
    exp_map = tensor_ops.map(operators.exp)
    id_map = tensor_ops.map(operators.id)
    inv_map = tensor_ops.map(operators.inv)

    # Zips
    add_zip = tensor_ops.zip(operators.add)
    mul_zip = tensor_ops.zip(operators.mul)
    lt_zip = tensor_ops.zip(operators.lt)
    eq_zip = tensor_ops.zip(operators.eq)
    is_close_zip = tensor_ops.zip(operators.is_close)
    relu_back_zip = tensor_ops.zip(operators.relu_back)
    log_back_zip = tensor_ops.zip(operators.log_back)
    inv_back_zip = tensor_ops.zip(operators.inv_back)

    # Reduce
    add_reduce = tensor_ops.reduce(operators.add, 0.0)
    mul_reduce = tensor_ops.reduce(operators.mul, 1.0)

    class Backend:
        cuda = is_cuda
        _id_map = id_map
        _add_reduce = add_reduce

        class Neg(Function):
            @staticmethod
            def forward(ctx, t1):
                return neg_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                return neg_map(grad_output)

        class Inv(Function):
            @staticmethod
            def forward(ctx, t1):
                ctx.save_for_backward(t1)
                return inv_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                t1 = ctx.saved_values
                return inv_back_zip(t1, grad_output)

        class Add(Function): #? 如果两个输入的shape是不一样的?他们的grad应该也不一样？为什么不考虑？
            @staticmethod
            def forward(ctx, t1, t2):
                ctx.save_for_backward(t1,t2)
                return add_zip(t1, t2)

            @staticmethod
            def backward(ctx, grad_output):
                t1,t2=ctx.saved_values
                return t1.expand(grad_output), t2.expand(grad_output)

        class Mul(Function):
            @staticmethod
            def forward(ctx, a, b):
                # TODO: Implement for Task 2.3.
                ctx.save_for_backward(a,b)
                return mul_zip(a,b)
                # raise NotImplementedError('Need to implement for Task 2.3')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.4.
                a,b=ctx.saved_values
                return a.expand(mul_zip(b,grad_output)),b.expand(mul_zip(a,grad_output))
                # raise NotImplementedError('Need to implement for Task 2.4')

        class Sigmoid(Function):
            @staticmethod
            def forward(ctx, a):
                # TODO: Implement for Task 2.3.
                sig_a=sigmoid_map(a)
                ctx.save_for_backward(sig_a)
                return sig_a
                # raise NotImplementedError('Need to implement for Task 2.3')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.4.
                sig_a=ctx.saved_values
                return mul_zip(grad_output,mul_zip(sig_a,add_zip(tensor_fromlist([1]),neg_map(sig_a)))) 
                # raise NotImplementedError('Need to implement for Task 2.4')

        class ReLU(Function):
            @staticmethod
            def forward(ctx, a):
                # TODO: Implement for Task 2.3.
                ctx.save_for_backward(a)
                return relu_map(a)
                # raise NotImplementedError('Need to implement for Task 2.3')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.4.
                a=ctx.saved_values
                return relu_back_zip(a,grad_output)
                # raise NotImplementedError('Need to implement for Task 2.4')

        class Log(Function):
            @staticmethod
            def forward(ctx, a):
                # TODO: Implement for Task 2.3.
                ctx.save_for_backward(a)
                return log_map(a)
                # raise NotImplementedError('Need to implement for Task 2.3')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.4.
                a=ctx.saved_values
                return log_back_zip(a,grad_output)
                # raise NotImplementedError('Need to implement for Task 2.4')

        class Exp(Function):
            @staticmethod
            def forward(ctx, a):
                # TODO: Implement for Task 2.3.
                exp_a=exp_map(a)
                ctx.save_for_backward(exp_a)
                return exp_a
                # raise NotImplementedError('Need to implement for Task 2.3')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.4.
                exp_a=ctx.saved_values
                return mul_zip(grad_output,exp_a) 
                # raise NotImplementedError('Need to implement for Task 2.4')

        class Sum(Function):
            @staticmethod
            def forward(ctx, a, dim):
                ctx.save_for_backward(a.shape, dim)
                if dim is not None:
                    return add_reduce(a, dim)
                else:
                    return add_reduce(
                        #用于计算总的item的个数
                        a.contiguous().view(int(operators.prod(a.shape))), 0
                    )

            @staticmethod
            def backward(ctx, grad_output):
                a_shape, dim = ctx.saved_values
                if dim is None:
                    out = grad_output.zeros(a_shape)
                    out._tensor._storage[:] = grad_output[0]
                    return out
                else:
                    # return grad_output #origin
                    #TODO:REQUIRE TO THINK CAREFULLY 
                    #这里为什么可以通过呢？ 明明grad_output广播为a_shape的形状后才能得到真正的梯度？原因是因为accumulate_derivative会自动用零加上，增广为它的形状。
                    # 那为什么加法，乘法等要求一定要返回准确的形状呢？这是因为grad_output是a_shape的广播，刚好与第一种情况相反。
                    # 可以想象到expand当中的三段处理中的第二段就是为了该情况而考虑的。
                    # 但是对于更复杂的情况准确性就不一定了。
                    return grad_output+zeros(a_shape)

        class All(Function):
            @staticmethod
            def forward(ctx, a, dim):
                if dim is not None:
                    return mul_reduce(a, dim)
                else:
                    return mul_reduce(
                        a.contiguous().view(int(operators.prod(a.shape))), 0
                    )
        class Mean(Function):
            @staticmethod
            def forward(ctx, a, dim):
                
                if dim is not None:
                    size=a.shape[dim]
                    ctx.save_for_backward(a,dim,size)
                    return mul_zip(add_reduce(a,dim),tensor_fromlist([1.0/a.shape[dim]]))
                else:
                    size=int(operators.prod(a.shape))
                    ctx.save_for_backward(a,dim,size)
                    return mul_zip(add_reduce(a.contiguous().view(size),0),tensor_fromlist([1.0/size]))
                # raise NotImplementedError('Need to include this file from past assignment.')

            @staticmethod
            def backward(ctx, grad_output):
                a,dim,size=ctx.saved_values
                if dim is not None:
                    ans=zeros(a.shape)
                    ans=add_zip(ans,mul_zip(grad_output,tensor_fromlist([1.0/size])))
                    return ans
                else:
                    ans=zeros(a.shape)
                    ans=add_zip(ans,tensor_fromlist([grad_output[0]/size]))
                    return ans


                # raise NotImplementedError('Need to include this file from past assignment.')
        class LT(Function):
            @staticmethod
            def forward(ctx, a, b):
                # TODO: Implement for Task 2.3.
                ctx.save_for_backward(a.shape,b.shape)
                return lt_zip(a,b)
                # raise NotImplementedError('Need to implement for Task 2.3')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.4.
                ashape,bshape=ctx.saved_values
                return zeros(ashape),zeros(bshape)
                # raise NotImplementedError('Need to implement for Task 2.4')

        class EQ(Function):
            @staticmethod
            def forward(ctx, a, b):
                # TODO: Implement for Task 2.3.
                ctx.save_for_backward(a.shape,b.shape)
                return eq_zip(a,b)
                # raise NotImplementedError('Need to implement for Task 2.3')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.4.
                ashape,bshape=ctx.saved_values
                return zeros(ashape),zeros(bshape)
                # raise NotImplementedError('Need to implement for Task 2.4')

        class IsClose(Function):
            @staticmethod
            def forward(ctx, a, b):
                # TODO: Implement for Task 2.3.
                return is_close_zip(a,b)
                # raise NotImplementedError('Need to implement for Task 2.3')

        class Permute(Function):
            @staticmethod
            def forward(ctx, a, order):
                # TODO: Implement for Task 2.3.
                ctx.save_for_backward(order)
                return Tensor(a._tensor.permute(*order),backend=a.backend)
                # raise NotImplementedError('Need to implement for Task 2.3')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.4.
                order=ctx.saved_values
                re_order=[0 for i in range(len(order))]
                for k,v in enumerate(order):
                    re_order[v]=k
                return Tensor(grad_output._tensor.permute(*re_order),backend=grad_output.backend)

                # raise NotImplementedError('Need to implement for Task 2.4')

        class View(Function):
            @staticmethod
            def forward(ctx, a, shape):
                ctx.save_for_backward(a.shape)
                assert a._tensor.is_contiguous(), "Must be contiguous to view"
                return Tensor.make(a._tensor._storage, shape, backend=a.backend)

            @staticmethod
            def backward(ctx, grad_output):
                original = ctx.saved_values
                return Tensor.make(
                    grad_output._tensor._storage, original, backend=grad_output.backend
                )

        class Copy(Function):
            @staticmethod
            def forward(ctx, a):
                return id_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        class MatMul(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                ctx.save_for_backward(t1, t2)
                return tensor_ops.matrix_multiply(t1, t2)

            @staticmethod
            def backward(ctx, grad_output):
                t1, t2 = ctx.saved_values

                def transpose(a):
                    order = list(range(a.dims))
                    order[-2], order[-1] = order[-1], order[-2]
                    return a._new(a._tensor.permute(*order))

                return (
                    tensor_ops.matrix_multiply(grad_output, transpose(t2)),
                    tensor_ops.matrix_multiply(transpose(t1), grad_output),
                )

    return Backend


TensorFunctions = make_tensor_backend(TensorOps)


# Helpers for Constructing tensors
def zeros(shape, backend=TensorFunctions):
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend

    Returns:
        :class:`Tensor` : new tensor
    """
    return Tensor.make([0] * int(operators.prod(shape)), shape, backend=backend)


def rand(shape, backend=TensorFunctions, requires_grad=False):
    """
    Produce a random tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(ls, shape=None, backend=TensorFunctions, requires_grad=False):
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls (list): data for tensor
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    if not shape:
        shape = (len(ls),)
    tensor = Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor_fromlist(ls, backend=TensorFunctions, requires_grad=False):
    """
    Produce a tensor with data and shape from ls

    Args:
        ls (list): data for tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls):
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls):
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape = shape(ls)
    return tensor(cur, tuple(shape), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(f, *vals, arg=0, epsilon=1e-6, ind=None):
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f, *vals):
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    # print("out.shape:",out.shape)
    out.sum().backward()

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        # # debug
        # print("hello")
        # print("ind",ind)
        # print("x.shape:",x.shape)
        # print("x.grad.shape",x.grad.shape)
        # assert 0
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        np.testing.assert_allclose(x.grad[ind], check, 1e-2, 1e-2)
