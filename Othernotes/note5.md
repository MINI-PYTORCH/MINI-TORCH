关于view和reshape的区别。

view 函数只能用于 contiguous 后的 tensor 上，也就是只能用于内存中连续存储的 tensor。如果对 tensor **调用过 transpose, permute 等操作的话会使该 tensor 在内存中变得不再连续**，此时就不能再调用 view 函数。因此，需要先使用 contiguous 来返回一个 contiguous copy。
reshape 则不需要依赖目标 tensor 是否在内存中是连续的。

原文链接：https://blog.csdn.net/HuanCaoO/article/details/104794075/

> 加黑处原因，因为strides会被改变,storage实际上没改变。