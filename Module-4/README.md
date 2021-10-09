

## RESULT

MNIST数据集选取100个训练数据，30个测试数据，最终的运行结果如下。

最大的缺陷是尽管使用了jit实现卷积算子等，但运行速度实在太慢，导致不得不使用很小的训练集和测试集。


```python

Epoch  4  example  0  loss  227.99535603676892 eval_accuracy  0.26666666666666666 train_accuracy  0.22

Epoch  9  example  0  loss  217.2792879507466 eval_accuracy  0.5 train_accuracy  0.42

Epoch  14  example  0  loss  177.3316376447844 eval_accuracy  0.5 train_accuracy  0.56

Epoch  19  example  0  loss  199.0772699261184 eval_accuracy  0.6333333333333333 train_accuracy  0.56

Epoch  24  example  0  loss  114.2096354918763 eval_accuracy  0.36666666666666664 train_accuracy  0.5

Epoch  34  example  0  loss  91.29269839573625 eval_accuracy  0.6333333333333333 train_accuracy  0.75

Epoch  39  example  0  loss  38.71350300038923 eval_accuracy  0.6666666666666666 train_accuracy  0.94

Epoch  44  example  0  loss  38.24167122192205 eval_accuracy  0.7 train_accuracy  0.87

Epoch  49  example  0  loss  48.32263035197364 eval_accuracy  0.6333333333333333 train_accuracy  0.94

Epoch  54  example  0  loss  71.06039867447983 eval_accuracy  0.5666666666666667 train_accuracy  0.77

Epoch  59  example  0  loss  81.40645907953355 eval_accuracy  0.6333333333333333 train_accuracy  0.79

Epoch  64  example  0  loss  56.79745140569936 eval_accuracy  0.7 train_accuracy  0.92

Epoch  69  example  0  loss  21.338493807760262 eval_accuracy  0.7333333333333333 train_accuracy  0.99

Epoch  74  example  0  loss  15.727859416071178 eval_accuracy  0.6666666666666666 train_accuracy  0.98

Epoch  79  example  0  loss  14.35703247717639 eval_accuracy  0.7333333333333333 train_accuracy  0.99

Epoch  84  example  0  loss  13.785507672261227 eval_accuracy  0.7666666666666667 train_accuracy  1.0

Epoch  89  example  0  loss  11.552276920917441 eval_accuracy  0.7 train_accuracy  1.0

Epoch  94  example  0  loss  11.647459155089415 eval_accuracy  0.7 train_accuracy  1.0

Epoch  99  example  0  loss  5.689393917524065 eval_accuracy  0.7666666666666667 train_accuracy  1.0

Epoch  114  example  0  loss  1.8656490900953642 eval_accuracy  0.7333333333333333 train_accuracy  1.0

```

