import minitorch
from minitorch.tensor_functions import tensor as tensor_from_list

a=tensor_from_list([1],requires_grad=True)
b=tensor_from_list([[1,1]],requires_grad=True)

c=(a*b).sum()
print(b[0,0]) # True 只实现了完整的索引操作。
print(b[0]) #False 没有实现层次索引。
print(b[:]) #False 没有实现切片索引。

c.backward()
print(c.shape)
print(a.shape,a.grad.shape)
print(b.shape,b.grad.shape)