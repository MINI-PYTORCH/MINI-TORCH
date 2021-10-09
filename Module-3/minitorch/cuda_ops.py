from numba import cuda
import numba

from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
import numpy as np

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)


def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.3.
        assert len(out) == out_size
        i = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
        if i >= out_size:
            return
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)
        out[index_to_position(out_index, out_strides)] = fn(
            in_storage[index_to_position(in_index, in_strides)])
        # raise NotImplementedError('Need to implement for Task 3.3')

    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)

        # Instantiate and run the cuda kernel.
        threadsperblock = 32
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # TODO: Implement for Task 3.3.
        assert len(out) == out_size
        i = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
        if i >= out_size:
            return
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        out[index_to_position(out_index, out_strides)] = fn(a_storage[index_to_position(
            a_index, a_strides)], b_storage[index_to_position(b_index, b_strides)])
        # raise NotImplementedError('Need to implement for Task 3.3')

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = 32
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_shape,
        reduce_size,
    ):
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError('Need to implement for Task 3.3')
        assert len(out) == out_size
        i = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
        if i >= out_size:
            return
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        reduce_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)  # 注意临时变量都需要这个包起来
        pos = cuda.local.array(2, numba.int32)
        to_index(i, out_shape, out_index)
        pos[0] = index_to_position(out_index, out_strides)
        for j in range(reduce_size):
            to_index(j, reduce_shape, reduce_index)
            # a_index=reduce_index+out_index #直接赋值会重新定义
            for k in range(len(out_shape)):
                a_index[k] = reduce_index[k]+out_index[k]
            pos[1] = index_to_position(a_index, a_strides)
            out[pos[0]] = fn(out[pos[0]], a_storage[pos[1]])

    return cuda.jit()(_reduce)


def reduce(fn, start=0.0):
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dims=None, out=None):
        if isinstance(dims, int):
            dims = [dims]
        old_shape = None
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
        else:
            old_shape = out.shape
            diff = len(a.shape) - len(out.shape)
            out = out.view(*([1] * diff + list(old_shape)))

        # Assume they are the same dim
        assert len(out.shape) == len(a.shape)

        # Create a reduce shape / reduce size
        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if out.shape[i] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)

        threadsperblock = 32
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), np.array(reduce_shape), reduce_size
        )
        # START CODE CHANGE
        if old_shape is not None:
            out = out.view(*old_shape)
        # END CODE CHANGE
        return out

    return ret


@cuda.jit()
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    # TODO: Implement for Task 3.4.
    i=cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if i>=out_size:
        return 
    out_index=cuda.local.array(MAX_DIMS,numba.int32)
    a_index=cuda.local.array(MAX_DIMS,numba.int32)
    b_index=cuda.local.array(MAX_DIMS,numba.int32)
    to_index(i, out_shape, out_index)
    broadcast_index(out_index,out_shape, a_shape, a_index)
    broadcast_index(out_index,out_shape,b_shape,b_index)
    for j in range(a_shape[-1]):
        a_index[len(a_shape)-1]=j #注意这里不能像fast_ops用-1，-2因为index现在为长为32的array。
        b_index[len(b_shape)-2]=j
        out[index_to_position(out_index, out_strides)] += (a_storage[index_to_position(a_index, a_strides)]
                         * b_storage[index_to_position(b_index, b_strides)])

    # raise NotImplementedError('Need to implement for Task 3.4')


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))
    threadsperblock = 32
    blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
