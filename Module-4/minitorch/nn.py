from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(ipt, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = ipt.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height=height//kh
    new_width=width//kw
    #view 会返回一个连续存储的tensor
    # TODO: Implement for Task 4.3. #tile的实现需要顶层的思考。
    # ipt=ipt.contiguous().view(batch,channel,new_height,kh,new_width,kw).permute(0,1,2,4,3,5).contiguous().view(batch,channel,new_height,new_width,kh*kw)
    ipt=ipt.contiguous()
    ipt=ipt.view(batch,channel,new_height,kh,new_width,kw)
    ipt=ipt.permute(0,1,2,4,3,5)
    ipt=ipt.contiguous()
    ipt=ipt.view(batch,channel,new_height,new_width,kw*kh)
    return ipt,new_height,new_width
    # raise NotImplementedError('Need to implement for Task 4.3')


def avgpool2d(ipt, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = ipt.shape
    # TODO: Implement for Task 4.3.
    ipt,new_height,new_width=tile(ipt,kernel)
    ipt=ipt.mean(4)
    ipt=ipt.view(batch,channel,new_height,new_width)
    return ipt
    
    # raise NotImplementedError('Need to implement for Task 4.3')


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, [dim])# 如果某一列的值全都一样呢？
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        ctx.save_for_backward(input,dim)
        return max_reduce(input,dim)
        # raise NotImplementedError('Need to implement for Task 4.4')

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        ipt,dim=ctx.saved_values
        return grad_output*argmax(ipt,dim)
        # raise NotImplementedError('Need to implement for Task 4.4')


max = Max.apply


def softmax(ipt, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    # TODO: Implement for Task 4.4.
    ipt=ipt.exp()
    tmp=ipt.sum(dim)
    return ipt/tmp

    # raise NotImplementedError('Need to implement for Task 4.4')


def logsoftmax(ipt, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    return softmax(ipt,dim).log()
    # raise NotImplementedError('Need to implement for Task 4.4')


def maxpool2d(ipt, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = ipt.shape
    # TODO: Implement for Task 4.4.
    ipt,new_height,new_width=tile(ipt,kernel)
    ipt=max(ipt,4)
    ipt=ipt.view(batch,channel,new_height,new_width)
    return ipt
    # raise NotImplementedError('Need to implement for Task 4.4')


def dropout(ipt, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if ignore:
        return ipt
    else:
        is_not_drop=rand(ipt.shape,backend=ipt.backend)>rate
        return is_not_drop*ipt 

    # raise NotImplementedError('Need to implement for Task 4.4')
