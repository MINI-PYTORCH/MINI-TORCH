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
    ipt=ipt.contiguous()
    ipt=ipt.view(batch,channel,new_height,kh,new_width,kw)
    ipt=ipt.permute(0,1,2,4,3,5)
    ipt=ipt.contiguous()
    ipt=ipt.view(batch,channel,new_height,new_width,kw*kh)
    return ipt,new_height,new_width


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
    ipt,new_height,new_width=tile(ipt,kernel)
    ipt=ipt.mean(4)
    ipt=ipt.view(batch,channel,new_height,new_width)
    return ipt


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
    out = max_reduce(input, [dim])
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        ctx.save_for_backward(input,dim)
        return max_reduce(input,dim)

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        ipt,dim=ctx.saved_values
        return grad_output*argmax(ipt,dim)


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
    ipt=ipt.exp()
    tmp=ipt.sum(dim)
    return ipt/tmp

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
    return softmax(ipt,dim).log()


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
    ipt,new_height,new_width=tile(ipt,kernel)
    ipt=max(ipt,4)
    ipt=ipt.view(batch,channel,new_height,new_width)
    return ipt


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
    if ignore:
        return ipt
    else:
        is_not_drop=rand(ipt.shape,backend=ipt.backend)>rate
        return is_not_drop*ipt 


'''
三种不同的损失函数
'''

def cross_entropy(out,y):
    out=logsoftmax(out,1)
    prob = (out * y).sum(1)
    loss = -prob.mean()
    return loss

def bce_loss(out,y):
    out  = out.sigmoid()
    prob = (out.log() * y) + ( (1-y)* (1-out).log()).sum(1)
    loss = -prob.mean()
    return loss

def mse_loss(out,y):
    loss=((out-y)*(out-y)).sum(1).mean()
    return loss

