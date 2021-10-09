```python
    strides=strides_from_shape(shape)
    for i in range(len(strides)-1):
        out_index[i]=ordinal//strides[i]
    out_index[len(shape)-1]=ordinal%shape[-1]
```
