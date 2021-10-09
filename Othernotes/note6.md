module-2当中很有收获的就是明白了tensor之间的运算的简单实现、包括storage和strides的区别，以及map、zip和reduce的实现，并利用这些实现了计算函数的forward和backward，进而重载tensor的函数，实现前向和反向传播。

并且当反向传播出现广播时，还需要进行梯度形状的修正。



> 有时间可以详细写一些。

