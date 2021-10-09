记录一个bug
task1_2过不去

使用打印日志的方法很快锁定了是relu和neg的类型问题。

发现relu的问题是因为没有使用max(a,0.0) 而是max(a,0)
发现neg的问题是sub的时候需要用到neg(b),而b的value没有保证为float。
那么为什么其他的没有出现这样的问题呢？因为其他的都是self或者跟self有运算的，会自动进行类型转换。
