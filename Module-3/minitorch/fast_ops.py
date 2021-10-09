import numpy as np

from minitorch.operators import mul
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,

)
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.1.
        for i in prange(len(out)):
            out_index=out_shape.copy()#等价于copy
            in_index=in_shape.copy()
            ordinal=i+0#因为i是prange的索引，要保证其不能被改变，最好的方法就是将拷贝传入函数。
            to_index(ordinal,out_shape,out_index) # 获得在某个continous下的out_index
            broadcast_index(out_index,out_shape,in_shape,in_index) #转化到某个continous下的in_index
            out[index_to_position(out_index,out_strides)]=fn(in_storage[index_to_position(in_index,in_strides)])#找到对应的strides下的position.
        # raise NotImplementedError('Need to implement for Task 3.1')

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      b = fn_map(a)


    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
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
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        for i in prange(len(out)):
            out_index=out_shape.copy()
            a_index=a_shape.copy()
            b_index=b_shape.copy()
            ordinal=i+0
            to_index(ordinal,out_shape,out_index)
            broadcast_index(out_index,out_shape,a_shape,a_index)
            broadcast_index(out_index,out_shape,b_shape,b_index)
            out[index_to_position(out_index,out_strides)]=fn(
                a_storage[index_to_position(a_index,a_strides)],
                b_storage[index_to_position(b_index,b_strides)]
                )
        # raise NotImplementedError('Need to implement for Task 2.2')

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function.

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape

    Returns:
        None : Fills in `out`

    """

    def _reduce(
        out,  ##array
        out_shape,   
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        reduce_shape,# a list
        reduce_size # a number 
    ):
        # TODO: Implement for Task 3.1.
        tmp_reduce_shape=np.array(reduce_shape)
        for i in prange(len(out)):
            out_index=out_shape.copy()
            a_index=a_shape.copy()
            reduce_index=tmp_reduce_shape.copy()
            reduce_shape=tmp_reduce_shape.copy() 
            ordinal_i=i+0
            to_index(ordinal_i,out_shape,out_index)
            out_pos=index_to_position(out_index,out_strides)
            for j in prange(reduce_size):
                ordinal_j=j+0
                to_index(ordinal_j,reduce_shape,reduce_index)
                a_index=reduce_index+out_index
                a_pos=index_to_position(a_index,a_strides)
                data=fn(out[out_pos],a_storage[a_pos])
                out[out_pos]=data
    #注意我们reduce的实现实际上会保留一的维度。
    # raise NotImplementedError('Need to implement for Task 3.1')
    return njit(parallel=True)(_reduce)



def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::
      fn_reduce = reduce(fn)
      reduced = fn_reduce(a, dims)
    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dims (list, optional): list of dims to reduce
        out (:class:`TensorData`, optional): tensor to reduce into
    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_reduce(njit()(fn))

    # START Code Update
    def ret(a, dims=None, out=None):
        if isinstance(dims,int):
            dims=[dims]
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

        # Apply
        f(*out.tuple(), *a.tuple(), reduce_shape, reduce_size)

        if old_shape is not None:
            out = out.view(*old_shape)
        return out

    return ret
    # END Code Update


@njit(parallel=True)
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    # TODO: Implement for Task 3.2.
    # basic implementation
    for i in prange(len(out)):
        out_index=out_shape.copy()
        ordinal_i=i+0
        to_index(ordinal_i,out_shape,out_index)
        out_pos=index_to_position(out_index,out_strides)
        for j in prange(a_shape[-1]):
            a_index=a_shape.copy()
            b_index=b_shape.copy() #这里不能把定义那到前面去，因为可能会发生并行问题。
            ordinal_j=j+0
            a_tmp_index=out_index.copy()
            a_tmp_index[-1]=ordinal_j
            b_tmp_index=out_index.copy()
            b_tmp_index[-2]=ordinal_j
            broadcast_index(a_tmp_index,out_shape,a_shape,a_index) #注意这里out_shape并不是a_shape的广播形式，但是由于我们的实现，仍然是正确的。
            broadcast_index(b_tmp_index,out_shape,b_shape,b_index)
            out[out_pos]+=(a_storage[index_to_position(a_index,a_strides)]*b_storage[index_to_position(b_index,b_strides)])

        

    # raise NotImplementedError('Need to implement for Task 3.2')


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

    # Create out shape
    # START CODE CHANGE
    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    # END CODE CHANGE
    out = a.zeros(tuple(ls))

    # Call main function
    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
